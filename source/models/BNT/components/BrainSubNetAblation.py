import math
import re
import zipfile
import xml.etree.ElementTree as ET
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def _read_subnet_labels(subnet_path: str) -> np.ndarray:
    """Read subnet labels from the existing atlas file."""
    try:
        df = pd.read_excel(subnet_path, header=None)
    except Exception:
        try:
            df = pd.read_csv(subnet_path, header=None)
        except Exception:
            return _read_xlsx_second_column_with_stdlib(subnet_path)

    if df.shape[1] < 2:
        raise ValueError("subnet file must contain at least two columns")

    return df.iloc[:, 1].values


def _read_xlsx_second_column_with_stdlib(subnet_path: str) -> np.ndarray:
    """Minimal xlsx reader used when openpyxl is unavailable."""
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    with zipfile.ZipFile(subnet_path) as workbook:
        shared_strings = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            for si in root.findall("main:si", ns):
                parts = [node.text or "" for node in si.findall(".//main:t", ns)]
                shared_strings.append("".join(parts))

        sheet = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))

    values = []
    for cell in sheet.findall(".//main:c", ns):
        cell_ref = cell.attrib.get("r", "")
        if re.sub(r"\d+", "", cell_ref) != "B":
            continue

        value_node = cell.find("main:v", ns)
        if value_node is None:
            continue

        raw_value = value_node.text or ""
        if cell.attrib.get("t") == "s":
            raw_value = shared_strings[int(raw_value)]
        values.append(raw_value)

    return np.array(values)


class AblationSubnetGraphFlattenTokenizer(nn.Module):
    """
    Base tokenizer for subnet partition ablations.

    It keeps the same forward behavior and output shape as
    SubnetGraphFlattenTokenizer: input (bz, N, N), output (bz, 8, embed_dim).
    Only the subnet node indices are different.
    """

    def __init__(
        self,
        subnet_indices: Iterable[Iterable[int]],
        num_nodes: int = 200,
        embed_dim: int = 128,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim

        self.subnet_indices: List[torch.Tensor] = []
        self.projectors = nn.ModuleList()

        for indices in subnet_indices:
            idx = torch.tensor(list(indices), dtype=torch.long)
            if idx.numel() == 0:
                raise ValueError("subnet indices cannot be empty")
            if torch.any(idx < 0) or torch.any(idx >= num_nodes):
                raise ValueError("subnet indices out of range")

            self.subnet_indices.append(idx)
            self.projectors.append(nn.Linear(idx.numel() * num_nodes, embed_dim))

    @property
    def subnet_sizes(self) -> List[int]:
        return [idx.numel() for idx in self.subnet_indices]

    def forward(self, global_feature: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_feature: (bz, N, N), log-mapped whole-brain feature matrix.

        Returns:
            sequence_matrix: (bz, 8, embed_dim).
        """
        bz = global_feature.size(0)
        device = global_feature.device
        subnet_tokens = []

        for idx_tensor, projector in zip(self.subnet_indices, self.projectors):
            curr_indices = idx_tensor.to(device)
            n_prime = curr_indices.numel()

            x = global_feature[:, curr_indices, :]
            a = x[:, :, curr_indices]

            ax = torch.bmm(a, x)
            ax_scaled = ax / math.sqrt(n_prime)
            aggregated_x = x + 0.1 * ax_scaled

            x_flattened = aggregated_x.reshape(bz, -1)
            subnet_tokens.append(projector(x_flattened))

        return torch.stack(subnet_tokens, dim=1)


class RandomSubnetGraphFlattenTokenizer(AblationSubnetGraphFlattenTokenizer):
    """
    Randomly partition nodes into 8 subnets.

    The subnet sizes are copied from the biological prior partition in
    subnet_path, so this ablation changes only membership, not token dimension.
    """

    def __init__(
        self,
        subnet_path: str,
        num_nodes: int = 200,
        embed_dim: int = 128,
        seed: Optional[int] = 42,
    ):
        labels = _read_subnet_labels(subnet_path)
        unique_subnets = sorted(np.unique(labels))
        biological_sizes = [int(np.sum(labels == subnet_id)) for subnet_id in unique_subnets]

        if sum(biological_sizes) != num_nodes:
            raise ValueError(
                f"biological subnet sizes sum to {sum(biological_sizes)}, "
                f"but num_nodes={num_nodes}"
            )

        rng = np.random.RandomState(seed)
        shuffled_nodes = rng.permutation(num_nodes)

        subnet_indices = []
        start = 0
        for size in biological_sizes:
            end = start + size
            subnet_indices.append(shuffled_nodes[start:end].tolist())
            start = end

        self.seed = seed
        self.biological_sizes = biological_sizes
        super().__init__(subnet_indices, num_nodes=num_nodes, embed_dim=embed_dim)


class SequentialEqualSubnetGraphFlattenTokenizer(AblationSubnetGraphFlattenTokenizer):
    """
    Sequentially split 200 nodes into 8 equal-size subnets.

    For num_nodes=200 and num_subnets=8, the groups are 0-24, 25-49, ...,
    175-199.
    """

    def __init__(
        self,
        num_nodes: int = 200,
        embed_dim: int = 128,
        num_subnets: int = 8,
    ):
        if num_nodes % num_subnets != 0:
            raise ValueError("num_nodes must be divisible by num_subnets")

        group_size = num_nodes // num_subnets
        subnet_indices = [
            range(start, start + group_size)
            for start in range(0, num_nodes, group_size)
        ]

        super().__init__(subnet_indices, num_nodes=num_nodes, embed_dim=embed_dim)

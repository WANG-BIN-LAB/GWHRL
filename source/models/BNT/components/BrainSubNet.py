import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import List
import random
import math
import re
import zipfile
import xml.etree.ElementTree as ET
import torch.nn.functional as F




class SubnetGraphFlattenTokenizer(nn.Module):

    def __init__(self, subnet_path, num_nodes=200, embed_dim=128):
        super(SubnetGraphFlattenTokenizer, self).__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim

        df = pd.read_excel(subnet_path, header=None)
        self.labels = df.iloc[:, 1].values
        self.unique_subnets = sorted(np.unique(self.labels))

        self.subnet_indices = []
        self.gcn_weights = nn.ModuleList()
        self.residual_alphas = nn.ParameterList()
        self.projectors = nn.ModuleList()

        for subnet_id in self.unique_subnets:
            idx = np.where(self.labels == subnet_id)[0]
            self.subnet_indices.append(torch.tensor(idx, dtype=torch.long))

            N_prime = len(idx)

            self.gcn_weights.append(
                nn.Sequential(
                    nn.Linear(num_nodes, num_nodes),
                    nn.GELU(),
                    nn.Linear(num_nodes, num_nodes)))

            self.residual_alphas.append(nn.Parameter(torch.tensor(0.1)))

            # self.projectors.append(nn.Linear(N_prime * num_nodes, embed_dim))
            self.projectors.append(nn.Sequential(
                nn.Linear(N_prime * num_nodes, embed_dim),
                nn.LayerNorm(embed_dim)
            ))
    def forward(self, global_feature):

        bz = global_feature.size(0)
        device = global_feature.device

        subnet_tokens = []

        for i, idx_tensor in enumerate(self.subnet_indices):
            curr_indices = idx_tensor.to(device)
            N_prime = len(curr_indices)

            gcn_weight = self.gcn_weights[i]
            alpha = self.residual_alphas[i]
            projector = self.projectors[i]

            X = global_feature[:, curr_indices, :]
            A = X[:, :, curr_indices]

            AX = torch.bmm(A, X)
            AX_scaled = AX / math.sqrt(N_prime)
            aggregated_X = X + alpha * AX_scaled

            aggregated_X = gcn_weight(aggregated_X)

            X_flattened = aggregated_X.view(bz, -1)
            subnet_token = projector(X_flattened)
            subnet_tokens.append(subnet_token)

        sequence_matrix = torch.stack(subnet_tokens, dim=1)

        return sequence_matrix


from typing import Tuple
class VectorDistanceAttention(nn.Module):
    def __init__(self, vec_dim: int, d_out: int):

        super(VectorDistanceAttention, self).__init__()
        self.d_in = vec_dim
        self.d_out = d_out
        self.alpha = nn.Parameter(torch.tensor(1.0))

        self.q_proj = nn.Linear(vec_dim, d_out, bias=False)
        self.k_proj = nn.Linear(vec_dim, d_out, bias=False)
        self.v_proj = nn.Linear(vec_dim, d_out, bias=False)

    def transform(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return q, k, v

    def forward(self, x: torch.Tensor, heads_num: int = 1) -> torch.Tensor:
        # (bz, n, vec)
        bz, n, _ = x.shape

        # 1. (bz, n, d_out)
        q, k, v = self.transform(x)

        assert self.d_out % heads_num == 0, "d_out must be divisible by heads_num"
        head_dim = self.d_out // heads_num

        q = q.view(bz, n, heads_num, head_dim).transpose(1, 2)
        k = k.view(bz, n, heads_num, head_dim).transpose(1, 2)
        v = v.view(bz, n, heads_num, head_dim).transpose(1, 2)

        q_sq = torch.sum(q ** 2, dim=-1, keepdim=True)
        k_sq = torch.sum(k ** 2, dim=-1).unsqueeze(-2)
        qk = torch.matmul(q, k.transpose(-1, -2))
        dist_sq = torch.relu(q_sq + k_sq - 2 * qk)

        tau = head_dim ** 0.5
        attn_scores = -dist_sq / tau

        atten_prob = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(atten_prob, v)
        out = out.transpose(1, 2).contiguous().view(bz, n, self.d_out)

        return out
















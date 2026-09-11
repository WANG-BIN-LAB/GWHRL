import numpy as np
import torch

from .preprocess import StandardScaler
from omegaconf import DictConfig, open_dict



def some_csr_matrix_object(matrix, sparsity_ratio=0.7):
    B, N, _ = matrix.shape  # Get batch size and matrix dimensions
    sparse_matrix = np.zeros_like(matrix)  # Initialize output matrix

    for i in range(B):  # Iterate over each matrix
        flat_matrix = matrix[i].flatten()  # Flatten current matrix to 1D
        k = int(len(flat_matrix) * sparsity_ratio)  # Calculate 70th percentile position
        threshold = np.partition(flat_matrix, k)[k]  # Get 70th percentile value

        # Set elements below 70th percentile to 0, keep original values of elements above or equal
        sparse_matrix[i] = np.where(matrix[i] >= threshold, matrix[i], 0.0)

    return sparse_matrix



def load_abide_data(cfg: DictConfig):

    data = np.load(cfg.dataset.path, allow_pickle=True).item()   # numpy 数组
    # data = data.item()
    final_timeseires = data["timeseires"]  # (1009,200,100)
    final_pearson = data["corr"]  # (1009,200,200)
    labels = data["label"] # (1009,)
    site = data['site']  # (1009,)

    scaler = StandardScaler(mean=np.mean(
        final_timeseires), std=np.std(final_timeseires))

    final_timeseires = scaler.transform(final_timeseires)

    final_timeseires, final_pearson, labels = [torch.from_numpy(
        data).float() for data in (final_timeseires, final_pearson, labels)]

    with open_dict(cfg):
        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = final_pearson.shape[1:]
        cfg.dataset.timeseries_sz = final_timeseires.shape[2]



    return final_timeseires, final_pearson, labels, site





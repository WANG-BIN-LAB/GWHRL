import torch
import torch.nn as nn
from omegaconf import DictConfig
from ..base import BaseModel
import numpy as np
from .components.BrainSubNet import VectorDistanceAttention, SubnetGraphFlattenTokenizer



class BrainNetworkTransformer(BaseModel):

    def __init__(self, config: DictConfig, use_random_subnet=False):

        super().__init__()

        subnetLabel_path = "./figure/cc200.csv"
        self.epsilon = 1e-6
        SPD_output_vector_dim = 128
        self.converter = SubnetGraphFlattenTokenizer(subnet_path=subnetLabel_path, num_nodes=200, embed_dim=SPD_output_vector_dim)
        self.num_subnets = len(self.converter.subnet_indices)
        self.epsilon = 1e-6

        self.fusion_layer = nn.Sequential(
            nn.Linear(SPD_output_vector_dim, SPD_output_vector_dim),
            nn.LayerNorm(SPD_output_vector_dim),
            nn.GELU()
        )

        self.attention = VectorDistanceAttention(vec_dim=SPD_output_vector_dim, d_out=SPD_output_vector_dim)
        self.attention_linear = nn.Linear(SPD_output_vector_dim, SPD_output_vector_dim)

        # in_dim = num_nodes * (num_nodes + 1) // 2
        self.global_extractor = nn.Sequential(
            nn.Linear(200*201//2, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, SPD_output_vector_dim)
        )

        self.fc = nn.Sequential(
            nn.Linear(self.num_subnets * SPD_output_vector_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 32),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )


    def forward(self,
                time_seires: torch.tensor,
                node_feature: torch.tensor,
                label: torch.tensor):

        bz, N, _, = node_feature.shape


        whole_brain_mapping = self.log_eig_mapping(self.make_spd(node_feature))

        sub_spds_vec = self.converter(whole_brain_mapping)

        global_token = self.global_extractor(self.vectorize_symmetric_matrix(whole_brain_mapping)) # (bz, vec)

        global_expanded = global_token.unsqueeze(1).expand(-1, sub_spds_vec.size(1), -1)
        concat_feat = sub_spds_vec + global_expanded  # (bz, 8, vec)
        subnet_global_vec = self.fusion_layer(concat_feat)

        att_output = self.attention(subnet_global_vec)  # (bz, 8, vec)
        seq_output = self.attention_linear(att_output)  # (bz, 8, vec)
        final_feat = seq_output.reshape(bz, -1)  # (bz, 8*vec)

        return self.fc(final_feat) ,final_feat


    def make_spd(self, fc_batch):
        x_sym = (fc_batch + fc_batch.transpose(-1, -2)) / 2.0
        trace = torch.diagonal(x_sym, dim1=-1, dim2=-2).sum(-1, keepdim=True)
        x_norm = x_sym / (trace.unsqueeze(-1) + 1e-8)  # 防止除0
        identity = torch.eye(x_norm.shape[-1], device=x_norm.device)
        identity = identity.unsqueeze(0).expand(x_norm.shape[0], -1, -1)
        eps = 1e-3
        spd_matrix = x_norm + eps * identity

        return spd_matrix


    def log_eig_mapping(self, spd_matrix):
        L, Q = torch.linalg.eigh(spd_matrix)
        L_clipped = torch.clamp(L, min=self.epsilon)
        log_L = torch.log(L_clipped)
        log_mapped = Q @ torch.diag_embed(log_L) @ Q.transpose(-1, -2)
        return log_mapped

    def vectorize_symmetric_matrix(self, sym_matrix):
        bz, d, _ = sym_matrix.shape
        row_idx, col_idx = torch.triu_indices(d, d)
        vectorized = sym_matrix[:, row_idx, col_idx]
        return vectorized






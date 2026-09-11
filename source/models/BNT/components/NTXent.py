import torch
import torch.nn as nn



def loss_cal(self, z1, z2):
    T = 0.01
    batch_size, _ = z1.size()
    z1_abs = z1.norm(dim=1)
    z2_abs = z2.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', z1, z2) / torch.einsum('i,j->ij', z1_abs, z2_abs)

    neg_sim_matrix = sim_matrix.clone()
    neg_sim_matrix[range(batch_size), range(batch_size)] = 0
    neg_sim_matrix = torch.exp(neg_sim_matrix / T)

    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    pos_sim = torch.exp(pos_sim / T)

    loss = pos_sim / (neg_sim_matrix.sum(dim=1) + pos_sim)
    loss = - torch.log(loss).mean()
    return loss



class LogContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(LogContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, features, labels):

        bz = features.size(0)
        if bz < 2:
            return torch.tensor(0.0, requires_grad=True, device=features.device)

        dist_matrix = torch.cdist(features, features, p=2.0)


        d_log = torch.log(1.0 + dist_matrix)

        labels_row = labels.unsqueeze(1)  # (bz, 1)
        labels_col = labels.unsqueeze(0)  # (1, bz)

        is_same_class = (labels_row == labels_col).float()
        y_ij = 1.0 - is_same_class  # 转换为论文中 y_ij 的定义

        loss_similar = (1.0 - y_ij) * (d_log ** 2)
        loss_dissimilar = y_ij * (torch.clamp(self.margin - d_log, min=0.0) ** 2)
        loss_matrix = loss_similar + loss_dissimilar

        mask = torch.eye(bz, device=features.device).bool()
        loss_matrix = loss_matrix.masked_fill(mask, 0.0)

        num_valid_pairs = bz * (bz - 1)
        loss = loss_matrix.sum() / num_valid_pairs

        return loss








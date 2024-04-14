import torch
import torch.nn as nn
from .._math.geodesics import compute_distance

__all__ = ["GDR"]


class GDR(nn.Module):
    def __init__(self, adjacency_matrix, target_dim, mid_channel):
        """
        Initialize the module

        Parameters
        ----------
        adjacency_matrix : (Genes, Genes) array-like
            Store the adjacency matrix of the gene co-expression network with only 0, 1 and -1.
        """
        super(GDR, self).__init__()
        self.adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.int8)
        self.degrees = torch.abs(self.adjacency_matrix).sum(dim=1)
        self.adjacency_matrix = self.adjacency_matrix.float() / torch.sqrt(
            self.degrees.unsqueeze(1) * self.degrees.unsqueeze(0)
        )
        self.degrees = None

        self.in_features = self.adjacency_matrix.shape[0]

        self.dim_reduction = nn.Sequential(
            nn.Linear(self.in_features, mid_channel * target_dim),
            nn.ReLU(),
            nn.Linear(mid_channel * target_dim, target_dim),
            nn.Tanh(),
        )
        self.recon = nn.Sequential(
            nn.Linear(target_dim, mid_channel * target_dim),
            nn.ReLU(),
            nn.Linear(mid_channel * target_dim, self.in_features),
        )

    def forward(self, x):
        x = self.dim_reduction(x @ self.adjacency_matrix)
        y = self.recon(x) @ self.adjacency_matrix
        return [x, y]

    def fn_reduction(self, x):
        return self.dim_reduction(x @ self.adjacency_matrix)

    def fn_recon(self, x):
        return torch.nn.functional.softplus(self.recon(x) @ self.adjacency_matrix)


class ReconLoss(nn.Module):
    def __init__(self):
        super(ReconLoss, self).__init__()

    @staticmethod
    def forward(batch, recon_batch):
        diff = batch - recon_batch
        diff = torch.norm(diff, p=2, dim=1)
        diff = torch.mean(diff)
        return diff


class DimReductionLoss(nn.Module):
    def __init__(self):
        super(DimReductionLoss, self).__init__()

    @staticmethod
    def forward(batch, dr_batch, param, alpha, K, method):
        dis_mat = compute_distance(batch, param, alpha, K, method)
        dis_mat_dr = dr_batch[:, None, :] - dr_batch[None, :, :]
        dis_mat_dr = torch.norm(dis_mat_dr, p=2, dim=2)
        return torch.nn.functional.mse_loss(dis_mat, dis_mat_dr)


class joint_loss(nn.Module):
    def __init__(self):
        super(joint_loss, self).__init__()

    def forward(
        self, batch, dr_batch, recon_batch, joint_coef, param, alpha, K, method
    ):
        loss_recon = ReconLoss.forward(batch, recon_batch)
        loss_dr = DimReductionLoss.forward(batch, dr_batch, param, alpha, K, method)
        return loss_recon + joint_coef * loss_dr

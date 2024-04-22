import torch
import torch.nn as nn
import torch_geometric.nn as pygnn

from .._math.geodesics import compute_distance

__all__ = ["GDR", "JointLoss"]


class GDR(nn.Module):
    def __init__(self, in_genes, batch_size, adjacency_mat, target_dims, mid_channel):
        super().__init__()

        def get_edge_index(in_matrix):
            adj_matrix = torch.tensor(in_matrix)
            rows, cols = torch.nonzero(adj_matrix, as_tuple=True)
            return torch.stack([rows, cols], dim=0)

        self.edge_index = get_edge_index(adjacency_mat)
        self.conv1 = pygnn.CGConv(batch_size, mid_channel * batch_size)
        self.conv2 = pygnn.CGConv(mid_channel * batch_size, batch_size)
        self.dr = nn.Sequential(
            nn.CELU(),
            nn.Linear(in_genes, mid_channel * target_dims),
            nn.CELU(),
            nn.Linear(mid_channel * target_dims, target_dims),
        )
        self.recon = nn.Sequential(
            nn.Linear(target_dims, mid_channel * target_dims),
            nn.CELU(),
            nn.Linear(mid_channel * target_dims, in_genes),
            nn.Softplus(),
        )
        self.linear1 = nn.Linear(in_genes, in_genes)

    def forward(self, x):
        x = x.t()
        x = self.conv1(x, self.edge_index)
        x = nn.functional.celu(x)
        x = self.conv2(x, self.edge_index)
        x = x.t()
        x = self.dr(x)
        y = self.recon(x)
        return x, y


class ReconLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(batch, recon_batch):
        diff = batch - recon_batch
        diff = torch.norm(diff, p=2, dim=1)
        diff = torch.mean(diff)
        return diff


class DimReductionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(batch, dr_batch, param, alpha, K, method):
        dis_mat = compute_distance(batch, param, alpha, K, method)
        dis_mat_dr = dr_batch[:, None, :] - dr_batch[None, :, :]
        dis_mat_dr = torch.norm(dis_mat_dr, p=2, dim=2)
        return torch.nn.functional.mse_loss(dis_mat, dis_mat_dr)


class JointLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        batch,
        initial_batch,
        dr_batch,
        recon_batch,
        joint_coef,
        param,
        alpha,
        K,
        method,
    ):
        loss_recon = ReconLoss.forward(initial_batch, recon_batch)
        loss_dr = DimReductionLoss.forward(batch, dr_batch, param, alpha, K, method)
        return loss_recon + joint_coef * loss_dr, loss_recon, loss_dr

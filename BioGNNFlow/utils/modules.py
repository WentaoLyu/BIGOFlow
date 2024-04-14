import torch
import torch.nn as nn

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
            nn.Linear(self.in_features, mid_channel),
            nn.ReLU(),
            nn.Linear(mid_channel, target_dim),
            nn.Tanh(),
        )
        self.recon = nn.Sequential(
            nn.Linear(target_dim, mid_channel),
            nn.ReLU(),
            nn.Linear(mid_channel, self.in_features),
        )

    def forward(self, x):
        x = self.dim_reduction(x @ self.adjacency_matrix)
        y = self.recon(x) @ self.adjacency_matrix
        return [x, y]

    def fn_reduction(self, x):
        return self.dim_reduction(x @ self.adjacency_matrix)

    def fn_recon(self, x):
        return self.recon(x) @ self.adjacency_matrix

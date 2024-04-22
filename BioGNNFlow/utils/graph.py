import accelerate as _accelerate
import numpy as _np
import tasklogger as _tasklogger
import torch as _torch
from torch.utils.data import DataLoader as _DataLoader
from torch.utils.data import TensorDataset as _TensorDataset

from .modules import *
from .mtx import *

_logger = _tasklogger.get_tasklogger("graphlogger")
_logger.set_level(1)


# TODO finish the GCE class.
class GCENetwork:
    def __init__(self, data, ground_truth=None) -> None:
        """
        Initialize the gene co-expression network.

        Parameters
        ----------
        data : (..., Genes) array-like
            The scRNA-seq data, the last dimension of which is genes expression.
        ground_truth : (Genes, Genes) array-like, optional
            The ground truth provided by user, if None(default), the initial graph is initialized with 0,
            the diagonal units shall be `1`.
        """
        self.adjacency_matrix = _init_graph(data, ground_truth)  # adjacency matrix
        self.data = data
        self.n = self.adjacency_matrix.shape[0]
        self.cut_off_admatrix = None
        self.cut_index = None
        self.cut_data = None
        self.nm_cut_data = None
        self.module = None
        self.accelerator = None
        self.cut_off_normalized = False
        self.loss_fn = None
        self.accelerator = None
        self.dataloader = None
        self.module = None
        self.optimizer = None

    def compute_admatrix(
        self, method: str = "Pearson", noise=False, scale_factor: float = 0.01
    ) -> None:
        """
        Compute the adjacency matrix of the gene co-expression network.

        Parameters
        ----------
        method : str, optional
            The method of computing correlation coefficient, default is 'Pearson'.
        noise : bool, optional
            Whether to add noise to the RNA seq Data, default is False.
        scale_factor : float, optional
            The scale factor of the noise, default is 0.01.
        """
        _logger.start_task("adjacency matrix")
        if noise:
            temp_data = _np.array(self.data)
            scales = _np.std(temp_data, axis=0)
            scales[_np.isclose(scales, 0)] = scale_factor
            temp_data += _np.random.normal(
                loc=0, scale=scale_factor * scales, size=temp_data.shape
            )
        else:
            temp_data = _np.array(self.data)
        if method == "Pearson":
            self.adjacency_matrix = _np.corrcoef(temp_data, rowvar=False)
        elif method == "Spearman":
            # TODO complete the computation of spearman correlation coefficient.
            pass
        else:
            _logger.complete_task("adjacency matrix")
            raise ValueError(
                "The method of computing correlation coefficient is not supported."
            )
        _logger.complete_task("adjacency matrix")

    def cut_off(
        self,
        positive_threshold: float = 0.3,
        negative_threshold: float = 0.3,
        cut_off_genes: bool = True,
    ) -> None:
        _logger.start_task("cut off")
        positive = (self.adjacency_matrix >= positive_threshold).astype(int)
        negative = -1 * (self.adjacency_matrix < negative_threshold).astype(int)
        self.cut_off_admatrix = positive + negative
        if cut_off_genes:
            row_sum = _np.sum(_np.abs(self.cut_off_admatrix), axis=1)
            arg_sort = _np.argsort(row_sum)
            row_sum = row_sum[arg_sort]
            cut_index = _np.where(row_sum > 1)[0][0]
            _logger.log_info(
                f"Cut off {cut_index} genes, since they have rare connection with other genes."
            )
            _logger.log_info(
                f"{self.adjacency_matrix.shape[0] - cut_index} genes remains to be analyzed."
            )
            cut_index = arg_sort[cut_index:]
            self.cut_off_admatrix = self.cut_off_admatrix[_np.ix_(cut_index, cut_index)]
            self.cut_index = cut_index
            self.cut_data = _np.array(self.data)[:, cut_index]
            det = _np.linalg.det(self.cut_off_admatrix)
            if not _np.isclose(det, 0):
                _logger.log_info(
                    f"The cut off matrix is invertible, the determinant is {det}."
                )
            else:
                _logger.log_warning("The cut off matrix is not invertible.")
        _logger.complete_task("cut off")

    def normalize_cut_off(self):
        if self.cut_off_normalized:
            return
        else:
            degrees = _np.sum(_np.abs(self.cut_off_admatrix), axis=1)
            self.cut_off_admatrix = self.cut_off_admatrix / _np.sqrt(
                degrees[:, None] * degrees[None, :]
            )
            self.cut_off_normalized = True

    def get_module(self, target_dim=32, mid_channel=4, batch_size=64, lr=1e-3):
        if not self.cut_off_normalized:
            raise ValueError("Adjacency matrix not normalized!")
        self.accelerator = _accelerate.Accelerator()
        in_feature = self.cut_off_admatrix.shape[0]
        self.module = GDR(
            in_feature,
            batch_size,
            adjacency_mat=self.cut_off_admatrix,
            target_dims=target_dim,
            mid_channel=mid_channel,
        )
        self.loss_fn = JointLoss()
        dataset = _TensorDataset(
            _torch.tensor(self.cut_data, dtype=_torch.float32),
        )
        self.dataloader = _DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.optimizer = _torch.optim.Adam(
            self.module.parameters(), lr=lr, amsgrad=True
        )

        (
            self.module,
            self.optimizer,
            self.dataloader,
            self.loss_fn,
        ) = self.accelerator.prepare(
            self.module,
            self.optimizer,
            self.dataloader,
            self.loss_fn,
        )

    def train(self, joint_coef, param, alpha, K, method="Gaussian", epochs=100):
        self.module.train()
        recon_ls = []
        dr_ls = []
        _logger.start_task("training")
        for epoch in range(epochs):
            batch_num = 0
            average_recon = 0
            average_dr = 0
            for data in self.dataloader:
                self.optimizer.zero_grad()
                dr_batch, recon_batch = self.module(data[0])
                loss, loss_recon, loss_dr = self.loss_fn(
                    data[0],
                    data[0],
                    dr_batch,
                    recon_batch,
                    joint_coef=joint_coef,
                    param=param,
                    alpha=alpha,
                    K=K,
                    method=method,
                )
                self.accelerator.backward(loss)
                self.optimizer.step()
                average_recon = (average_recon * batch_num + loss_recon.item()) / (
                    batch_num + 1
                )
                average_dr = (average_dr * batch_num + loss_dr.item()) / (batch_num + 1)
                batch_num += 1
                dr_ls.append(loss_dr.item())
                recon_ls.append(loss_recon.item())
            _logger.log_info(
                f"training on epoch {epoch}\t loss_recon: {average_recon}\t loss_dr: {average_dr}"
            )
            with open("/root/Dynamics-Notebook/training_log", "a") as f:
                f.write(
                    f"training on epoch {epoch}\t loss_recon: {average_recon}\t loss_dr: {average_dr}\n"
                )
        _logger.complete_task("training")
        return dr_ls, recon_ls

import numpy as _np
import tasklogger as _tasklogger

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
        positive = (self.adjacency_matrix > positive_threshold).astype(int)
        negative = -1 * (self.adjacency_matrix < negative_threshold).astype(int)
        self.cut_off_admatrix = positive + negative
        if cut_off_genes:
            row_sum = _np.sum(_np.abs(self.cut_off_admatrix))
            arg_sort = _np.argsort(row_sum)
            row_sum = row_sum[arg_sort]
            cut_index = _np.where(row_sum > 1)[0][0]
            _logger.info(
                f"Cut off {cut_index} genes, since they have rare connection with other genes."
            )
            cut_index = arg_sort[cut_index:]
            self.cut_off_admatrix = self.cut_off_admatrix[_np.ix_(cut_index, cut_index)]
            self.cut_index = cut_index
            self.cut_data = _np.array(self.data)[:, cut_index]

        _logger.complete_task("cut off")

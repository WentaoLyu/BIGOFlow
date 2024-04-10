import numpy as _np
import tasklogger as _tasklogger

from .preprocess import to_numpy

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
        self.admatrix = _init_graph(data, ground_truth)  # adjacency matrix
        self.data = data
        self.n = self.admatrix.shape[0]

    def compute_admatrix(
        self, method: str = "Pearson", noise=True, scale_factor: float = 0.01
    ) -> None:
        """
        Compute the adjacency matrix of the gene co-expression network.

        Parameters
        ----------
        method : str, optional
            The method of computing correlation coefficient, default is 'Pearson'.
        noise : bool, optional
            Whether to add noise to the RNA seq Data, default is True.
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
        if method == "Pearson":
            self.admatrix = _np.corrcoef(temp_data, rowvar=False)
        elif method == "Spearman":
            # TODO complete the computation of spearman correlation coefficient.
            pass
        else:
            _logger.complete_task("adjacency matrix")
            raise ValueError(
                "The method of computing correlation coefficient is not supported."
            )
        _logger.complete_task("adjacency matrix")

    def clustering(self) -> None:
        pass


@to_numpy()
def _init_graph(data, ground_truth=None) -> _np.ndarray:
    """
    Initialize the gene co-expression network.

    Parameters
    ----------
    data : (..., Genes) array-like
        The scRNA-seq data, the last dimension of which is genes expression.
    ground_truth : (Genes, Genes) array-like, optional
        The ground truth provided by user, if None(default), the initial graph is initialized with eye matrix,
        the diagonal units shall be `1`.

    Returns
    -------
    (Genes, Genes) array
        The initial graph of the gene co-expression network.
    """
    if ground_truth is not None:
        initial_graph = ground_truth
    else:
        initial_graph = _np.eye(data.shape[-1])
    return initial_graph

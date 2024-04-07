import numpy as _np

from preprocess import to_numpy


# TODO finish the GCE class.
class GCENetwork:
    @to_numpy()
    def __init__(self, data, ground_truth=None) -> None:
        """
        Initialize the gene co-expression network.

        Parameters
        ----------
        data : (..., Genes) array-like
            The scRNA-seq data, the last dimension of which is genes expression.
        ground_truth : (Genes, Genes) array-like, optional
            The ground truth provided by user, if None(default), the initial graph is initialized with gaussian noise,
            the diagonal units shall be `1`.
        """
        self.admatrix = _init_graph(data, ground_truth)  # adjacency matrix
        self.data = data
        self.n = self.admatrix.shape[0]

    def compute_admatrix(self, method: str = "pearson") -> None:
        """
        Compute the adjacency matrix of the gene co-expression network.

        Parameters
        ----------
        method : str, optional
            The method of computing correlation coefficient, default is 'pearson'.
        """
        if method == "pearson":
            self.admatrix = _np.corrcoef(self.data, rowvar=False)
        elif method == "spearman":
            # TODO complete the computation of spearman correlation coefficient.
            pass


def _init_graph(data: _np.ndarray, ground_truth: _np.ndarray = None) -> _np.ndarray:
    """
    Initialize the gene co-expression graph, add unitary distributed noise to data.

    Parameters
    ----------
    data : (..., Genes) array
        The scRNA-seq data, the last dimension of which is genes expression.
    ground_truth : (Genes, Genes) array, optional
        The ground truth provided by user, if None(default), the initial graph is initialized with gaussian noise,
        the diagonal units shall be `1`.

    Returns
    -------
    initial_graph : (Genes, Genes) array
        The initial graph
    """
    initial_graph = _np.zeros((data.shape[-1], data.shape[-1]))
    noise = _np.triu(2 * (_np.random.random(size=initial_graph.shape) - 0.5), k=1)
    noise += noise.T
    if ground_truth is not None:
        initial_graph += ground_truth
        initial_graph += noise
    else:
        _np.fill_diagonal(noise, 1)
        initial_graph = noise
    return initial_graph

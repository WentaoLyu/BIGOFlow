import numpy as _np
import numpy.typing as _npt

from preprocess import to_numpy, get_shape


# TODO finish the GCE class.
class GCENetwork:

    def __init__(self, data, ground_truth=None) -> None:
        self.admatrix = _init_graph(data, ground_truth)  # adjacency matrix
        self.n = self.admatrix.shape[0]


@to_numpy()
def _init_graph(
    data: _npt.ArrayLike, ground_truth: _npt.ArrayLike = None
) -> _np.ndarray:
    """
    Initialize the gene co-expression graph, add unitary distributed noise to data.

    Parameters
    ----------
    data : (..., Genes) array-like
        The scRNA-seq data, the last dimension of which is genes expression.
    ground_truth : (Genes, Genes) array-like, optional
        The ground truth provided by user, if None(default), the initial graph is initialized with gaussian noise,
        the diagonal units shall be `1`.

    Returns
    -------
    initial_graph : (Genes, Genes) array
        The initial graph
    """
    initial_graph = _np.zeros((get_shape(data)[-1], get_shape(data)[-1]))
    noise = _np.triu(2 * (_np.random.random(size=initial_graph.shape) - 0.5), 1)
    noise += noise.T
    if ground_truth is not None:
        initial_graph += ground_truth
        initial_graph += noise
    else:
        _np.fill_diagonal(noise, 1)
        initial_graph = noise
    return initial_graph

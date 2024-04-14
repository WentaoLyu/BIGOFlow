import numpy as _np

from .preprocess import *

__all__ = ["_init_graph"]


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

import os

import pandas as _pd
from scipy.io import mmread as _mmread
from scipy.io import mmwrite as _mmwrite

_MTX_FILE = "matrix.mtx"
_BARCODE_FILE = "barcodes.tsv"
_GENES_FILE = "genes.tsv"


def load_data(data_dir: str) -> _pd.DataFrame:
    """
    Load the scRNA-seq data from the folder directory.

    Parameters
    ----------
    data_dir : str
        The directory of the folder containing the data.

    Returns
    -------
    data : DataFrame
        The data loaded from the folder.
    """
    mtx = _mmread(os.path.join(data_dir, _MTX_FILE))
    barcodes = _pd.read_csv(
        os.path.join(data_dir, _BARCODE_FILE), header=None
    ).squeeze()
    genes = _pd.read_csv(os.path.join(data_dir, _GENES_FILE), header=None).squeeze()
    return _pd.DataFrame.sparse.from_spmatrix(data=mtx, index=barcodes, columns=genes)


def save_data(data_dir: str, dataset: _pd.DataFrame) -> None:
    """
    Save the scRNA-seq data to the folder directory.

    Parameters
    ----------
    data_dir : str
        The directory of the folder containing the data.
    dataset : _pd.DataFrame
        The data to be saved.
    """
    mtx = dataset.values
    _mmwrite(os.path.join(data_dir, _MTX_FILE), mtx)
    dataset.index.to_series().to_csv(
        os.path.join(data_dir, _BARCODE_FILE), sep="\n", index=False, header=False
    )
    dataset.columns.to_series().to_csv(
        os.path.join(data_dir, _GENES_FILE), sep="\n", index=False, header=False
    )

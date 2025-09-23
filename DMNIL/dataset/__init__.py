from __future__ import absolute_import
import warnings

from .U1652_dor import U1652_dor
from .U1652_sat import U1652_sat
from .SUES_dor import SUES_dor
from .SUES_sat import SUES_sat
from .DenseUAV_dor import DenseUAV_dor
from .DenseUAV_sat import DenseUAV_sat

__factory = {
    'U1652_dor':U1652_dor,
    'U1652_sat':U1652_sat,
    'SUES_dor':SUES_dor,
    'SUES_sat':SUES_sat,
    'DenseUAV_dor': DenseUAV_dor,
    'DenseUAV_sat':DenseUAV_sat,
}

def names():
    return sorted(__factory.keys())


def create(name, root,trial=0, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name.
    root : str
        The path to the dataset directory.
    split_id : int, optional
        The index of data split. Default: 0
    num_val : int or float, optional
        When int, it means the number of validation identities. When float,
        it means the proportion of validation to all the trainval. Default: 100
    download : bool, optional
        If True, will download the dataset. Default: False
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, trial=trial, *args, **kwargs)


def get_dataset(name, root, *args, **kwargs):
    warnings.warn("get_dataset is deprecated. Use create instead.")
    return create(name, root, *args, **kwargs)

import numpy as np
import torch


def assert_numpy_no_nan_no_inf(x):
    assert np.all(~np.isnan(x))
    assert np.all(~np.isinf(x))


def assert_torch_no_nan_no_inf(x):
    assert torch.all(~torch.isnan(x))
    assert torch.all(~torch.isinf(x))

import numpy as np


def assert_no_nan_no_inf(x):
    assert np.all(~np.isnan(x))
    assert np.all(~np.isinf(x))

import numpy as np
import torch


def assert_numpy_no_nan_no_inf(x):
    assert np.all(~np.isnan(x))
    assert np.all(~np.isinf(x))


def assert_torch_no_nan_no_inf(x):
    assert torch.all(~torch.isnan(x))
    assert torch.all(~torch.isinf(x))


def numpy_logits_to_probs(logits):
    probs = 1. / (1. + np.exp(-logits))
    return probs


def numpy_probs_to_logits(probs):
    logits = - np.log(1. / probs - 1.)
    return logits


def torch_logits_to_probs(logits):
    probs = 1. / (1. + torch.exp(-logits))
    return probs


def torch_probs_to_logits(probs):
    logits = - torch.log(1. / probs - 1.)
    return logits

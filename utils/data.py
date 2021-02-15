import numpy as np


def generate_mixture_of_gaussians(num_gaussians: int = 3,
                                  gaussian_dim: int = 2,
                                  gaussian_mean_prior_cov_scaling: float = 3.,
                                  gaussian_cov_scaling: float = 0.3):
    # sample Gaussians' means from prior = N(0, rho * I)
    means = np.random.multivariate_normal(
        mean=np.zeros(gaussian_dim),
        cov=gaussian_mean_prior_cov_scaling * np.eye(gaussian_dim),
        size=num_gaussians)

    # all Gaussians have same covariance
    cov = gaussian_cov_scaling * np.eye(gaussian_dim)
    covs = np.repeat(cov[np.newaxis, :, :],
                     repeats=num_gaussians,
                     axis=0)

    mixture_of_gaussians = dict(means=means, covs=covs)

    return mixture_of_gaussians


def sample_sequence_from_crp(T: int,
                             alpha: float):
    assert alpha > 0
    table_counts = np.zeros(shape=T, dtype=np.int)
    sampled_tables = np.zeros(shape=T, dtype=np.int)

    # the first customer always goes at the first table
    table_counts[0] = 1

    for t in range(1, T):
        max_k = np.argmin(table_counts)  # first zero index
        freq = table_counts.copy()
        freq[max_k] = alpha
        probs = freq / np.sum(freq)
        z_t = np.random.choice(np.arange(max_k + 1), p=probs[:max_k + 1])
        sampled_tables[t] = z_t
        table_counts[z_t] += 1
    return table_counts, sampled_tables


vectorized_sample_sequence_from_crp = np.vectorize(sample_sequence_from_crp,
                                                   otypes=[np.ndarray])


def sample_sequence_from_mixture_of_gaussians(seq_len: int = 100,
                                              class_sampling: str = 'Uniform',
                                              alpha: float = None,
                                              num_gaussians: int = None,
                                              gaussian_params: dict = dict()):
    """
    Draw sample from mixture of Gaussians, using either uniform sampling or
    CRP sampling.

    Exactly one of alpha and num_gaussians must be specified.

    :param seq_len: desired sequence length
    :param class_sampling:
    :param alpha:
    :param num_gaussians:
    :param gaussian_params:
    :return:
        assigned_table_seq: NumPy array with shape (seq_len,) of (integer) sampled classes
        gaussian_samples_seq: NumPy array with shape (seq_len, dim of Gaussian) of
                                Gaussian samples
    """

    if class_sampling == 'Uniform':
        assert num_gaussians is not None
        assigned_table_seq = np.random.choice(np.arange(num_gaussians),
                                              replace=True,
                                              size=seq_len)

    elif class_sampling == 'CRP':
        assert alpha is not None
        table_counts, assigned_table_seq = sample_sequence_from_crp(T=seq_len,
                                                                    alpha=alpha)
        num_gaussians = np.max(assigned_table_seq) + 1
    else:
        raise ValueError(f'Impermissible class sampling: {class_sampling}')

    mixture_of_gaussians = generate_mixture_of_gaussians(
        num_gaussians=num_gaussians,
        **gaussian_params)

    # create sequence of Gaussian samples from (mean_t, cov_t)
    gaussian_samples_seq = np.array([
        np.random.multivariate_normal(mean=mixture_of_gaussians['means'][assigned_table],
                                      cov=mixture_of_gaussians['covs'][assigned_table])
        for assigned_table in assigned_table_seq])

    # convert assigned table sequence to one-hot codes
    assigned_table_seq_one_hot = np.zeros((seq_len, seq_len))
    assigned_table_seq_one_hot[np.arange(seq_len), assigned_table_seq] = 1.

    result = dict(
        mixture_of_gaussians=mixture_of_gaussians,
        assigned_table_seq=assigned_table_seq,
        assigned_table_seq_one_hot=assigned_table_seq_one_hot,
        gaussian_samples_seq=gaussian_samples_seq
    )

    return result

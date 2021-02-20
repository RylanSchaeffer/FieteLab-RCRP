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


def generate_mixture_of_unigrams(num_topics: int,
                                 vocab_dim: int,
                                 dp_concentration_param: float = 5.7,
                                 prior_over_topic_parameters: float = 0.8):

    assert dp_concentration_param > 0
    assert prior_over_topic_parameters > 0

    beta_samples = np.random.beta(1, dp_concentration_param, size=num_topics)
    stick_weights = np.zeros(shape=num_topics, dtype=np.float64)
    remaining_stick = 1.
    for topic_index in range(num_topics):
        stick_weights[topic_index] = beta_samples[topic_index] * remaining_stick
        remaining_stick *= (1. - beta_samples[topic_index])
    # ordinarily, we'd set the last stick weight so that the total stick weights sum to 1.
    # However, floating-point errors can make this last value negative (yeah, I was surprised too)
    # so I only do this if the sum of the weights isn't sufficiently close to 1
    try:
        assert np.allclose(np.sum(stick_weights), 1.)
    except AssertionError:
        stick_weights[-1] = 1 - np.sum(stick_weights[:-1])
    assert np.alltrue(stick_weights >= 0.)
    assert np.allclose(np.sum(stick_weights), 1.)

    # sort stick weights to be descending in magnitude
    stick_weights = np.sort(stick_weights)[::-1]

    topics_parameters = np.zeros((num_topics, vocab_dim))
    for topic_index in range(num_topics):
        gam_t = np.random.gamma(prior_over_topic_parameters, size=vocab_dim)
        topics_parameters[topic_index, :] = gam_t / np.sum(gam_t)

    mixture_of_unigrams = dict(stick_weights=stick_weights,
                               topics_parameters=topics_parameters)

    return mixture_of_unigrams


def sample_sequence_from_crp(T: int,
                             alpha: float):
    assert alpha > 0
    table_occupancies = np.zeros(shape=T, dtype=np.int)
    sampled_tables = np.zeros(shape=T, dtype=np.int)

    # the first customer always goes at the first table
    table_occupancies[0] = 1

    for t in range(1, T):
        max_k = np.argmin(table_occupancies)  # first zero index
        freq = table_occupancies.copy()
        freq[max_k] = alpha
        probs = freq / np.sum(freq)
        z_t = np.random.choice(np.arange(max_k + 1), p=probs[:max_k + 1])
        sampled_tables[t] = z_t
        table_occupancies[z_t] += 1
    return table_occupancies, sampled_tables


vectorized_sample_sequence_from_crp = np.vectorize(sample_sequence_from_crp,
                                                   otypes=[np.ndarray, np.ndarray])


def sample_sequence_from_mixture_of_gaussians(seq_len: int = 100,
                                              class_sampling: str = 'Uniform',
                                              alpha: float = None,
                                              num_gaussians: int = None,
                                              gaussian_params: dict = {}):
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

    elif class_sampling == 'DP':
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


def sample_sequence_from_mixture_of_unigrams(seq_len: int = 450,
                                             num_topics: int = 10,
                                             vocab_dim: int = 8,
                                             doc_len: int = 120,
                                             unigram_params: dict = {}):

    mixture_of_unigrams = generate_mixture_of_unigrams(
        num_topics=num_topics,
        vocab_dim=vocab_dim,
        **unigram_params)

    # create sequence of Multinomial samples from (num_topics, vocab_dim)
    # draw a Sample of num_docs documents each of size doc_len
    assigned_table_seq_one_hot = np.random.multinomial(
        n=1,
        pvals=mixture_of_unigrams['stick_weights'],
        size=seq_len)
    # convert assigned table sequence to one-hot codes
    assigned_table_seq = np.argmax(assigned_table_seq_one_hot, axis=1)

    doc_samples_seq = np.zeros(shape=(seq_len, vocab_dim))
    for doc_idx in range(seq_len):
        topic_idx = assigned_table_seq[doc_idx]
        doc = np.random.multinomial(n=doc_len,
                                    pvals=mixture_of_unigrams['topics_parameters'][topic_idx, :],
                                    size=1)
        doc_samples_seq[doc_idx, :] = doc

    result = dict(
        mixture_of_unigrams=mixture_of_unigrams,
        assigned_table_seq=assigned_table_seq,
        assigned_table_seq_one_hot=assigned_table_seq_one_hot,
        doc_samples_seq=doc_samples_seq,
    )

    return result

from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.infer
import numpyro.distributions

from utils.helpers import assert_no_nan_no_inf


def bayesian_recursion(docs,
                       alpha: float,
                       likelihood_fn,
                       update_parameters_fn,
                       epsilon: float = 10.):
    assert alpha > 0
    assert epsilon > 0
    num_docs, vocab_dim = docs.shape

    table_assignment_priors = np.zeros((num_docs, num_docs), dtype=np.float64)
    table_assignment_priors[0, 0] = 1.
    table_assignment_posteriors = np.zeros((num_docs, num_docs), dtype=np.float64)
    table_assignment_posteriors_running_sum = np.zeros_like(table_assignment_posteriors)
    num_table_posteriors = np.zeros(shape=(num_docs, num_docs))

    parameters = dict(
        topics_concentrations=np.empty(shape=(0, vocab_dim)),  # need small value to ensure non-zero
        epsilon=epsilon)

    likelihoods_bookeeping = np.zeros(shape=(num_docs, num_docs))

    for doc_idx, doc in enumerate(docs):
        likelihoods, parameters = likelihood_fn(doc=doc, parameters=parameters)
        likelihoods_bookeeping[doc_idx, :len(likelihoods)] = likelihoods

        # if doc_idx == 8:
        #     plt.imshow(likelihoods_bookeeping[:doc_idx, :doc_idx])
        #     plt.show()
        #     print(19)

        if doc_idx == 0:
            # first customer has to go at first table
            table_assignment_posterior = np.array([1.])
            table_assignment_posteriors[doc_idx, 0] = table_assignment_posterior
            num_table_posteriors[0, 0] = 1.
        else:
            table_assignment_prior = np.copy(table_assignment_posteriors_running_sum[doc_idx - 1, :len(likelihoods)])
            # we don't subtract 1 because Python uses 0-based indexing
            assert np.allclose(np.sum(table_assignment_prior), doc_idx)
            # right shift table posterior by 1
            table_assignment_prior[1:] += alpha * np.copy(num_table_posteriors[doc_idx - 1, :len(likelihoods) - 1])
            table_assignment_prior /= (alpha + doc_idx)
            assert np.allclose(np.sum(table_assignment_prior), 1.)

            # record latent prior
            table_assignment_priors[doc_idx, :len(table_assignment_prior)] = table_assignment_prior

            unnormalized_table_assignment_posterior = np.multiply(likelihoods, table_assignment_prior)
            table_assignment_posterior = unnormalized_table_assignment_posterior / np.sum(
                unnormalized_table_assignment_posterior)
            assert np.allclose(np.sum(table_assignment_posterior), 1.)

            # record latent posterior
            table_assignment_posteriors[doc_idx, :len(table_assignment_posterior)] = table_assignment_posterior

            # update posterior over number of tables
            for k1, p_z_t_equals_k1 in enumerate(table_assignment_posteriors[doc_idx, :doc_idx + 1]):
                for k2, p_prev_num_tables_equals_k2 in enumerate(num_table_posteriors[doc_idx - 1, :doc_idx + 1]):
                    # exclude cases of placing customer at impossible table
                    if k1 > k2 + 1:
                        continue
                    # customer allocated to previous table
                    elif k1 <= k2:
                        num_table_posteriors[doc_idx, k2] += p_z_t_equals_k1 * p_prev_num_tables_equals_k2
                    # create new table
                    elif k1 == k2 + 1:
                        num_table_posteriors[doc_idx, k1] += p_z_t_equals_k1 * p_prev_num_tables_equals_k2
                    else:
                        raise ValueError
            num_table_posteriors[doc_idx, :] /= np.sum(num_table_posteriors[doc_idx, :])
            assert np.allclose(np.sum(num_table_posteriors[doc_idx, :]), 1.)

        # update running sum of posteriors
        table_assignment_posteriors_running_sum[doc_idx, :] = table_assignment_posteriors_running_sum[doc_idx - 1, :] + \
                                                              table_assignment_posteriors[doc_idx, :]
        assert np.allclose(np.sum(table_assignment_posteriors_running_sum[doc_idx, :]),
                           doc_idx + 1)

        parameters = update_parameters_fn(
            doc=doc,
            table_assignment_posteriors_running_sum=table_assignment_posteriors_running_sum[doc_idx,
                                                    :len(table_assignment_posterior)],
            table_assignment_posterior=table_assignment_posterior,
            parameters=parameters)

    bayesian_recursion_results = dict(
        table_assignment_priors=table_assignment_priors,
        table_assignment_posteriors=table_assignment_posteriors,
        table_assignment_posteriors_running_sum=table_assignment_posteriors_running_sum,
        num_table_posteriors=num_table_posteriors,
        parameters=parameters,
    )

    return bayesian_recursion_results


def expectation_maximization(docs,
                             alpha: float,
                             num_iter: int,
                             epsilon: float = 1e-5):
    assert alpha > 0
    assert epsilon > 0
    assert num_iter > 0
    assert isinstance(num_iter, int)
    num_docs, vocab_dim = docs.shape
    num_topics = 10  # num_docs

    table_assignment_priors = np.zeros((num_docs, num_docs), dtype=np.float64)
    # table_assignment_posteriors = np.zeros((num_docs, num_docs), dtype=np.float64)
    num_table_posteriors = np.zeros(shape=(num_docs, num_docs))

    # # randomly initialize concentration parameters
    # topic_probabilities = np.random.dirichlet(alpha=np.full(shape=vocab_dim,
    #                                                         fill_value=alpha),
    #                                           size=num_topics)
    # topic_probabilities += epsilon
    # topic_probabilities = np.divide(
    #     topic_probabilities,
    #     np.sum(topic_probabilities, axis=1)[:, np.newaxis]
    # )
    # assert np.all(topic_probabilities > 0)
    # assert np.allclose(np.sum(topic_probabilities, axis=1), 1.)
    # assert topic_probabilities.shape == (num_topics, vocab_dim)
    #
    # for iter_idx in range(num_iter):
    #     # shape (num_docs, num_topics)
    #     log_like = docs @ np.log(topic_probabilities).T
    #     like_norm = np.exp(log_like)
    #     soft_assign = like_norm / like_norm.sum(axis=1)[:, np.newaxis]  # normalized distribution
    #     assert_no_nan_no_inf(soft_assign)
    #     hard_assign = np.argmax(soft_assign, axis=1)
    #     print('num_clusters =', len(np.unique(hard_assign)))
    #
    #     # import scipy.stats
    #     # like_scipy = scipy.stats.multinomial.pmf(x=docs[0, :],
    #     #                                          iter_idx=np.sum(docs[0, :]).astype(np.int),
    #     #                                          p=topic_probabilities[0, :])
    #
    #     soft_counts = soft_assign.T @ docs
    #     new_topic_probabilities = np.divide(
    #         soft_counts,
    #         np.sum(soft_counts, axis=1)[:, np.newaxis])
    #     # if dividing by 0/0, replace with epsilon
    #     new_topic_probabilities[np.isnan(new_topic_probabilities)] = epsilon
    #     new_topic_probabilities = np.divide(
    #         new_topic_probabilities,
    #         np.sum(new_topic_probabilities, axis=1)[:, np.newaxis]
    #     )
    #     assert_no_nan_no_inf(new_topic_probabilities)
    #     diff = np.linalg.norm(topic_probabilities - new_topic_probabilities)
    #     assert_no_nan_no_inf(diff)
    #     print(diff)
    #     topic_probabilities = new_topic_probabilities

    # parameters = dict(topics_concentrations=topic_probabilities)
    # table_assignment_posteriors = np.copy(soft_assign)
    # assert_no_nan_no_inf(table_assignment_posteriors)
    # table_assignment_posteriors_running_sum = np.cumsum(table_assignment_posteriors,
    #                                                     axis=0)

    P = np.zeros((num_topics, vocab_dim))
    for k in range(num_topics):
        arr = np.random.gamma(alpha, size=vocab_dim)
        arr = arr / arr.sum()
        P[k, :] = arr

    for n in range(num_iter):
        # num_docs x num_topics
        log_like = docs @ np.log(P).T
        like_norm = np.exp(log_like)
        soft_assign = like_norm / like_norm.sum(axis=1)[:, np.newaxis]  # normalized distribution
        counts_soft = soft_assign.T @ docs
        P = counts_soft + alpha
        P = P / P.sum(axis=1)[:, np.newaxis]

    # these are the MAP probability vectors for each topic
    parameters = dict(topics_concentrations=P)
    table_assignment_posteriors = np.copy(soft_assign)
    table_assignment_posteriors_running_sum = np.cumsum(table_assignment_posteriors,
                                                        axis=0)

    expectation_maximization_results = dict(
        table_assignment_priors=table_assignment_priors,
        table_assignment_posteriors=table_assignment_posteriors,
        table_assignment_posteriors_running_sum=table_assignment_posteriors_running_sum,
        num_table_posteriors=num_table_posteriors,
        parameters=parameters,
    )

    return expectation_maximization_results


def sampling_hmc_gibbs(docs,
                       num_samples: int,
                       alpha: float,
                       epsilon: float = 10,
                       sampling_max_num_clusters=None,
                       burn_fraction: float = 0.25):
    num_docs, vocab_dim = docs.shape
    doc_len = np.sum(docs, axis=1)[0].astype(int)
    if sampling_max_num_clusters is None:
        # multiply by 2 as heuristic
        sampling_max_num_clusters = 2 * int(np.ceil(alpha * np.log(1 + num_docs / alpha)))

    def mix_weights(beta):
        beta1m_cumprod = jnp.cumprod(1 - beta, axis=-1)
        term1 = jnp.pad(beta, (0, 1), mode='constant', constant_values=1.)
        term2 = jnp.pad(beta1m_cumprod, (1, 0), mode='constant', constant_values=1.)
        return jnp.multiply(term1, term2)

    def model(obs):
        with numpyro.plate('beta_plate', sampling_max_num_clusters - 1):
            beta = numpyro.sample(
                'beta',
                numpyro.distributions.Beta(1, alpha))

        with numpyro.plate('word_prob_vectors_plate', sampling_max_num_clusters):
            topics_word_prob_vectors = numpyro.sample(
                'topics_word_prob_vectors',
                numpyro.distributions.Dirichlet(
                    concentration=jnp.full(shape=vocab_dim, fill_value=epsilon)))

        with numpyro.plate('data', num_docs):
            z = numpyro.sample(
                'z',
                numpyro.distributions.Categorical(mix_weights(beta=beta)).mask(False))  # TODO: is mask necessary?
            numpyro.sample(
                'obs',
                numpyro.distributions.Multinomial(
                    total_count=doc_len,
                    probs=topics_word_prob_vectors[z]),
                obs=obs)

    hmc_kernel = numpyro.infer.NUTS(model)
    kernel = numpyro.infer.DiscreteHMCGibbs(inner_kernel=hmc_kernel)
    mcmc = numpyro.infer.MCMC(kernel, num_warmup=100, num_samples=num_samples, progress_bar=True)
    mcmc.run(random.PRNGKey(0), obs=docs)
    # mcmc.print_summary()
    samples = mcmc.get_samples()

    # shape (num samples, num centroids, obs dim)
    # only keep last 1000 samples
    parameters = dict(
        topics_word_prob_vectors=np.mean(np.array(samples['topics_word_prob_vectors'][-1000:, :, :]),
                                         axis=0),
        mixture_weights=np.array(samples['beta']))

    # shape (num samples, num obs)
    sampled_table_assignments = np.array(samples['z'])
    # convert sampled cluster assignments from (num samples, num obs) to (num obs, num clusters)
    bins = np.arange(0, 2 + np.max(sampled_table_assignments))
    table_assignment_posteriors = np.stack([
        np.histogram(sampled_table_assignments[-1000:, obs_idx].astype(np.float),
                     bins=bins,
                     density=True)[0]
        for obs_idx in range(num_docs)])
    table_assignment_posteriors_running_sum = np.cumsum(table_assignment_posteriors,
                                                        axis=0)

    # plt.imshow(sampled_table_assignments, cmap='jet')
    # plt.show()
    import seaborn as sns
    fig, axes = plt.subplots(nrows=1, ncols=2)
    sns.heatmap(docs, cmap='jet', ax=axes[0])
    sns.heatmap(table_assignment_posteriors, cmap='jet', ax=axes[1])
    fig.suptitle(f'Num Samples = {num_samples}')
    plt.savefig(f'exp_02_mixture_of_unigrams/plots/hmc_gibbs_alpha={alpha}_num_samples={num_samples}.png')
    # plt.show()
    plt.close()

    # returns classes assigned and centroids of corresponding classes
    hmc_gibbs_results = dict(
        table_assignment_posteriors=table_assignment_posteriors,
        table_assignment_posteriors_running_sum=table_assignment_posteriors_running_sum,
        parameters=parameters,
    )

    return hmc_gibbs_results

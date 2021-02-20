import numpy as np

from utils.helpers import assert_no_nan_no_inf


def bayesian_recursion(docs,
                       alpha: float,
                       likelihood_fn,
                       update_parameters_fn,
                       epsilon: float = 1e-5):
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

    for doc_idx, doc in enumerate(docs):
        likelihoods, parameters = likelihood_fn(doc=doc, parameters=parameters)
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

            # # truncate to speed up. otherwise too slow.
            # if table_assignment_posterior[-1] > 1e-10:
            #     num_topics_idx += 1
            # else:
            #     # assign posterior to zero
            #     table_assignment_posterior[-1] = 0
            #     # reset this parameter
            #     inferred_topic_dirichlet_parameters[num_topics_idx, :] = epsilon

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
            table_assignment_posteriors_running_sum=table_assignment_posteriors_running_sum[doc_idx, :len(table_assignment_posterior)],
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

import numpy as np


def bayesian_recursion(observations,
                       alpha: float,
                       likelihood_fn,
                       update_parameters_fn):
    assert alpha > 0
    num_obs, obs_dim = observations.shape

    # The inference algorithm does not require recording the full history of priors/posteriors
    # We record the full history for subsequent analysis
    table_assignment_priors = np.zeros((num_obs, num_obs), dtype=np.float64)
    table_assignment_priors[0, 0] = 1.
    table_assignment_posteriors = np.zeros((num_obs, num_obs), dtype=np.float64)

    table_assignment_posteriors_running_sum = np.zeros_like(table_assignment_posteriors)

    num_table_posteriors = np.zeros(shape=(num_obs, num_obs))

    parameters = dict(
        means=np.empty(shape=(0, obs_dim)),
        covs=np.empty(shape=(0, obs_dim, obs_dim)))

    for obs_idx, observation in enumerate(observations):
        likelihood, parameters = likelihood_fn(observation=observation,
                                               parameters=parameters)
        if obs_idx == 0:
            # first customer has to go at first table
            table_assignment_posterior = np.array([1.])
            table_assignment_posteriors[obs_idx, 0] = table_assignment_posterior
            num_table_posteriors[0, 0] = 1.
        else:
            table_assignment_prior = np.copy(table_assignment_posteriors_running_sum[obs_idx - 1, :len(likelihood)])
            # we don't subtract 1 because Python uses 0-based indexing
            assert np.allclose(np.sum(table_assignment_prior), obs_idx)
            # right shift by 1
            table_assignment_prior[1:] += alpha * np.copy(num_table_posteriors[obs_idx - 1, :len(likelihood) - 1])
            table_assignment_prior /= (alpha + obs_idx)
            assert np.allclose(np.sum(table_assignment_prior), 1.)

            # record latent prior
            table_assignment_priors[obs_idx, :len(table_assignment_prior)] = table_assignment_prior

            unnormalized_table_assignment_posterior = np.multiply(likelihood, table_assignment_prior)
            table_assignment_posterior = unnormalized_table_assignment_posterior / np.sum(
                unnormalized_table_assignment_posterior)
            assert np.allclose(np.sum(table_assignment_posterior), 1.)

            # record latent posterior
            table_assignment_posteriors[obs_idx, :len(table_assignment_posterior)] = table_assignment_posterior

            # update posterior over number of tables
            for k1, p_z_t_equals_k1 in enumerate(table_assignment_posteriors[obs_idx, :obs_idx + 1]):
                for k2, p_prev_num_tables_equals_k2 in enumerate(num_table_posteriors[obs_idx - 1, :obs_idx + 1]):
                    # exclude cases of placing customer at impossible table
                    if k1 > k2 + 1:
                        continue
                    # customer allocated to previous table
                    elif k1 <= k2:
                        num_table_posteriors[obs_idx, k2] += p_z_t_equals_k1 * p_prev_num_tables_equals_k2
                    # create new table
                    elif k1 == k2 + 1:
                        num_table_posteriors[obs_idx, k1] += p_z_t_equals_k1 * p_prev_num_tables_equals_k2
                    else:
                        raise ValueError
            num_table_posteriors[obs_idx, :] /= np.sum(num_table_posteriors[obs_idx, :])
            assert np.allclose(np.sum(num_table_posteriors[obs_idx, :]), 1.)

        # update running sum of posteriors
        table_assignment_posteriors_running_sum[obs_idx, :] = table_assignment_posteriors_running_sum[obs_idx - 1, :] + \
                                                              table_assignment_posteriors[obs_idx, :]
        assert np.allclose(np.sum(table_assignment_posteriors_running_sum[obs_idx, :]), obs_idx + 1)

        # update parameters
        parameters = update_parameters_fn(
            observation=observation,
            table_assignment_posterior=table_assignment_posterior,
            table_assignment_posteriors_running_sum=table_assignment_posteriors_running_sum[
                                                    obs_idx, :len(table_assignment_posterior)],
            parameters=parameters)

    bayesian_recursion_results = dict(
        table_assignment_priors=table_assignment_priors,
        table_assignment_posteriors=table_assignment_posteriors,
        table_assignment_posteriors_running_sum=table_assignment_posteriors_running_sum,
        num_table_posteriors=num_table_posteriors,
        parameters=parameters,
    )

    return bayesian_recursion_results


def DP_means_online(class_seq, gaussian_samples_seq, lambd):
    # dimensionality of points
    dim = gaussian_samples_seq.shape[1]
    num_clusters = 1

    # centroids of clusters
    mu = np.zeros((num_clusters, dim))

    # initial cluster = first data point
    mu[0, :] = gaussian_samples_seq[0, :]

    # empirical online classification labels
    empirical_cluster_indicator = np.zeros(1)
    empirical_cluster_indicator[0] = 1

    # history of all samples seen till current time
    history = gaussian_samples_seq[0, :]
    for iter in range(1, len(gaussian_samples_seq)):

        # add new sample to history:
        history = np.concatenate((history, gaussian_samples_seq[iter, :]), axis=0)

        # compute distance of new sample from previous centroids:
        distances = np.linalg.norm(gaussian_samples_seq[iter, :] - mu)

        # if smallest distance greater than cutoff lambda, create new cluster:
        if (np.min(distances) > lambd):

            # increment number of clusters by 1:
            num_clusters += 1

            # centroid of new cluster = new sample
            mu = np.concatenate((mu, gaussian_samples_seq[iter, :]), axis=0)
            empirical_cluster_indicator = np.concatenate((empirical_cluster_indicator, num_clusters))

        else:

            # If the smallest distance is less than the cutoff lambda, assign point
            # to one of the older clusters (add 1 because numbering starts at 0):
            assigned_cluster = np.argmin(distances)
            empirical_cluster_indicator = np.concatenate((empirical_cluster_indicator, assigned_cluster + 1))

            # get indices of all older points assigned to that cluster:
            older_points_in_assigned_cluster_indices = empirical_cluster_indicator == assigned_cluster
            older_points_in_assigned_cluster = history[older_points_in_assigned_cluster_indices, :]

            # recompute centroid incorporating this new sample:
            mu[assigned_cluster, :] = np.mean(
                np.concatenate((gaussian_samples_seq[iter, :], older_points_in_assigned_cluster), axis=0))

            # returns classes assigned and centroids of corresponding classes
        return empirical_cluster_indicator, mu
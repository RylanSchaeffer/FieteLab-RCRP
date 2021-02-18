import numpy as np
import pymc3 as pm
from pymc3.math import logsumexp
from scipy.spatial.distance import cdist
from sklearn.mixture import BayesianGaussianMixture
from theano import tensor as tt
from theano.tensor.nlinalg import det


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


def dp_means_online(observations: np.ndarray,
                    lambd: float):
    assert lambd > 0

    # dimensionality of points
    num_obs, obs_dim = observations.shape
    max_num_clusters = num_obs
    num_clusters = 1

    # centroids of clusters
    means = np.zeros(shape=(max_num_clusters, obs_dim))

    # initial cluster = first data point
    means[0, :] = observations[0, :]

    # empirical online classification labels
    table_assignments = np.zeros((max_num_clusters, max_num_clusters))
    table_assignments[0, 0] = 1

    for obs_idx in range(1, len(observations)):

        # compute distance of new sample from previous centroids:
        distances = np.linalg.norm(observations[obs_idx, :] - means[:num_clusters, :],
                                   axis=1)
        assert len(distances) == num_clusters

        # if smallest distance greater than cutoff lambda, create new cluster:
        if np.min(distances) > lambd:

            # increment number of clusters by 1:
            num_clusters += 1

            # centroid of new cluster = new sample
            means[num_clusters - 1, :] = observations[obs_idx, :]
            table_assignments[obs_idx, num_clusters - 1] = 1.

        else:

            # If the smallest distance is less than the cutoff lambda, assign point
            # to one of the older clusters (add 1 because numbering starts at 0):
            assigned_cluster = np.argmin(distances)
            table_assignments[obs_idx, assigned_cluster] = 1.

            # get indices of all older points assigned to that cluster:
            older_points_in_assigned_cluster_indices = table_assignments[:, assigned_cluster] == 1
            older_points_in_assigned_cluster = observations[older_points_in_assigned_cluster_indices, :]

            assert older_points_in_assigned_cluster.shape[0] > 1

            # recompute centroid incorporating this new sample
            diff = observations[obs_idx, :] - means[assigned_cluster, :]
            num_points_in_assigned_cluster_indices = np.sum(older_points_in_assigned_cluster_indices)
            means[assigned_cluster, :] += diff / num_points_in_assigned_cluster_indices

            # means[assigned_cluster, :] = np.mean(older_points_in_assigned_cluster,
            #                                      axis=0)

    table_assignment_posteriors_running_sum = np.cumsum(np.copy(table_assignments), axis=0)

    # returns classes assigned and centroids of corresponding classes
    dp_means_online_results = dict(
        # table_assignment_priors=np.full_like(table_assignments, fill_value=np.nan),
        table_assignment_posteriors=table_assignments,
        table_assignment_posteriors_running_sum=table_assignment_posteriors_running_sum,
        parameters=dict(means=means),
    )
    return dp_means_online_results


def dp_means_offline(observations,
                     num_passes: int,
                     lambd: float):
    assert lambd > 0

    # dimensionality of points
    num_obs, obs_dim = observations.shape
    max_num_clusters = num_obs
    num_clusters = 1

    # centroids of clusters
    means = np.zeros(shape=(max_num_clusters, obs_dim))

    # initial cluster = first data point
    means[0, :] = observations[0, :]

    # empirical online classification labels
    table_assignments = np.zeros((max_num_clusters, max_num_clusters))
    table_assignments[0, 0] = 1

    for pass_idx in range(num_passes):
        for obs_idx in range(1, len(observations)):

            # compute distance of new sample from previous centroids:
            distances = np.linalg.norm(observations[obs_idx, :] - means[:num_clusters, :],
                                       axis=1)
            assert len(distances) == num_clusters

            # if smallest distance greater than cutoff lambda, create new cluster:
            if np.min(distances) > lambd:

                # increment number of clusters by 1:
                num_clusters += 1

                # centroid of new cluster = new sample
                means[num_clusters - 1, :] = observations[obs_idx, :]
                table_assignments[obs_idx, num_clusters - 1] = 1.

            else:

                # If the smallest distance is less than the cutoff lambda, assign point
                # to one of the older clusters
                assigned_cluster = np.argmin(distances)
                table_assignments[obs_idx, assigned_cluster] = 1.

        for cluster_idx in range(num_clusters):
            # get indices of all observations assigned to that cluster:
            indices_of_points_in_assigned_cluster = table_assignments[:, cluster_idx] == 1

            # get observations assigned to that cluster
            points_in_assigned_cluster = observations[indices_of_points_in_assigned_cluster, :]

            assert points_in_assigned_cluster.shape[0] >= 1

            # recompute centroid incorporating this new sample
            means[cluster_idx, :] = np.mean(points_in_assigned_cluster,
                                            axis=0)

    table_assignment_posteriors_running_sum = np.cumsum(np.copy(table_assignments), axis=0)

    # returns classes assigned and centroids of corresponding classes
    dp_means_offline_results = dict(
        table_assignment_posteriors=table_assignments,
        table_assignment_posteriors_running_sum=table_assignment_posteriors_running_sum,
        parameters=dict(means=means),
    )
    return dp_means_offline_results


def nuts_sampling(observations,
                  num_samples: int,
                  alpha: float,
                  gaussian_cov_scaling: int,
                  gaussian_mean_prior_cov_scaling: float,
                  burn_fraction: float = 0.25):
    assert alpha > 0

    num_obs, obs_dim = observations.shape
    max_num_clusters = num_obs
    table_assignments = np.zeros((max_num_clusters, max_num_clusters))
    table_assignments[0, 0] = 1

    # https://docs.pymc.io/notebooks/dp_mix.html
    def stick_breaking(beta):
        portion_remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])
        return beta * portion_remaining

    # https://docs.pymc.io/notebooks/gaussian-mixture-model-advi.html
    # Log likelihood of normal distribution
    def logp_normal(mu, tau, value):
        # log probability of individual samples
        k = tau.shape[0]
        delta = lambda mu: value - mu
        return (-1 / 2.0) * (
                k * tt.log(2 * np.pi)
                + tt.log(1.0 / det(tau))
                + (delta(mu).dot(tau) * delta(mu)).sum(axis=1)
        )

    # https://docs.pymc.io/notebooks/gaussian-mixture-model-advi.html
    # Log likelihood of Gaussian mixture distribution
    def logp_gmix(cluster_means, cluster_weights, tau):
        def logp_(value):
            logps = [tt.log(cluster_weights[i]) + logp_normal(mu, tau, value) for i, mu in enumerate(cluster_means)]
            return tt.sum(logsumexp(tt.stacklists(logps)[:, :num_obs], axis=0))

        return logp_

    # https://docs.pymc.io/notebooks/gaussian-mixture-model-advi.html
    with pm.Model() as model:
        pm_beta = pm.Beta("beta", 1.0, alpha, shape=max_num_clusters)
        pm_w = pm.Deterministic("w", stick_breaking(pm_beta))
        pm_cluster_means = [
            pm.MvNormal(f'cluster_mean_{cluster_idx}',
                        mu=pm.floatX(np.zeros(obs_dim)),
                        cov=pm.floatX(gaussian_mean_prior_cov_scaling * np.eye(obs_dim)),
                        shape=(obs_dim,))
            for cluster_idx in range(max_num_clusters)]
        pm_obs = pm.DensityDist('obs',
                                logp_gmix(cluster_means=pm_cluster_means,
                                          cluster_weights=pm_w,
                                          tau=gaussian_cov_scaling * np.eye(obs_dim)),
                                observed=observations)

        pm_trace = pm.sample(draws=num_samples,
                             tune=500,
                             chains=4,
                             target_accept=0.9,
                             random_seed=1)

    # TODO: figure out number of clusters

    # figure out which point belongs to which cluster
    # shape (max_num_clusters, num_samples, obs_dim)
    cluster_means_samples = np.stack([pm_trace[f'cluster_mean_{cluster_idx}']
                                      for cluster_idx in range(max_num_clusters)])

    # burn the first samples
    cluster_means_samples = cluster_means_samples[:, int(burn_fraction * num_samples):, :]
    cluster_means = np.mean(cluster_means_samples, axis=1)

    distance_obs_to_cluster_means = cdist(observations, cluster_means)
    table_assignment_posteriors = np.exp(-distance_obs_to_cluster_means / 2)
    table_assignment_posteriors *= np.power(2. * np.pi, -obs_dim / 2.)

    # normalize to get posterior distributions
    table_assignment_posteriors = np.divide(
        table_assignment_posteriors,
        np.sum(table_assignment_posteriors, axis=1)[:, np.newaxis])
    assert np.allclose(np.sum(table_assignment_posteriors, axis=1), 1.)

    table_assignment_posteriors_running_sum = np.cumsum(table_assignment_posteriors,
                                                        axis=0)

    params = dict(means=cluster_means,
                  covs=np.repeat(np.eye(obs_dim)[np.newaxis, :, :],
                                 repeats=num_obs,
                                 axis=0))

    gibbs_sampling_results = dict(
        table_assignment_posteriors=table_assignment_posteriors,
        table_assignment_posteriors_running_sum=table_assignment_posteriors_running_sum,
        parameters=params)

    return gibbs_sampling_results


def variational_bayes(observations,
                      alpha: float,
                      gaussian_mean_prior_cov_scaling: float,
                      max_iter: int = 100,
                      n_init: int = 5):
    assert alpha > 0

    num_obs, obs_dim = observations.shape
    var_dp_gmm = BayesianGaussianMixture(
        n_components=num_obs,
        covariance_type='tied',
        max_iter=max_iter,
        n_init=n_init,
        init_params='random',
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=alpha,
        mean_precision_prior=1. / gaussian_mean_prior_cov_scaling,
        random_state=0,
    )
    var_dp_gmm.fit(observations)
    table_assignment_posteriors = var_dp_gmm.predict_proba(observations)
    table_assignment_posteriors_running_sum = np.cumsum(table_assignment_posteriors,
                                                        axis=0)

    params = dict(means=var_dp_gmm.means_,
                  covs=np.repeat(var_dp_gmm.covariances_[np.newaxis, :, :],
                                 repeats=num_obs,
                                 axis=0))

    # returns classes assigned and centroids of corresponding classes
    variational_results = dict(
        table_assignment_posteriors=table_assignment_posteriors,
        table_assignment_posteriors_running_sum=table_assignment_posteriors_running_sum,
        parameters=params,
    )
    return variational_results

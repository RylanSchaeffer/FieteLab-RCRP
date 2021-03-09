from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.infer
import numpyro.distributions
# import pymc3 as pm
# from pymc3.math import logsumexp
import pyro
import pyro.distributions
import pyro.infer
from scipy.spatial.distance import cdist
from sklearn.mixture import BayesianGaussianMixture
from theano import tensor as tt
from theano.tensor.nlinalg import det
import torch
import torch.nn.functional as F

torch.set_default_tensor_type('torch.DoubleTensor')


# torch.set_default_tensor_type('torch.FloatTensor')


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
            table_assignment_posteriors_running_sum=table_assignment_posteriors_running_sum[obs_idx,
                                                    :len(table_assignment_posterior)],
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


# why sampling is so hard: https://mc-stan.org/users/documentation/case-studies/identifying_mixture_models.html
def sampling_hmc_gibbs(observations,
                       num_samples: int,
                       alpha: float,
                       gaussian_cov_scaling: int,
                       gaussian_mean_prior_cov_scaling: float,
                       sampling_max_num_clusters=None,
                       burn_fraction: float = 0.25):
    num_obs, obs_dim = observations.shape

    if sampling_max_num_clusters is None:
        # multiply by 2 for safety
        sampling_max_num_clusters = 2 * int(np.ceil(alpha * np.log(1 + num_obs / alpha)))

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

        with numpyro.plate('mean_plate', sampling_max_num_clusters):
            mean = numpyro.sample(
                'mean',
                numpyro.distributions.MultivariateNormal(
                    jnp.zeros(obs_dim),
                    gaussian_mean_prior_cov_scaling * jnp.eye(obs_dim)))

        with numpyro.plate('data', num_obs):
            z = numpyro.sample(
                'z',
                numpyro.distributions.Categorical(mix_weights(beta=beta)).mask(False))
            numpyro.sample(
                'obs',
                numpyro.distributions.MultivariateNormal(
                    mean[z],
                    gaussian_cov_scaling * jnp.eye(obs_dim)),
                obs=obs)

    hmc_kernel = numpyro.infer.NUTS(model)
    kernel = numpyro.infer.DiscreteHMCGibbs(inner_kernel=hmc_kernel)
    mcmc = numpyro.infer.MCMC(kernel, num_warmup=100, num_samples=num_samples, progress_bar=True)
    mcmc.run(random.PRNGKey(0), obs=observations)
    # mcmc.print_summary()
    samples = mcmc.get_samples()

    # shape (num samples, num centroids, obs dim)
    params = dict(means=np.mean(np.array(samples['mean'][-1000:, :, :]), axis=0))
    # shape (num samples, num obs)
    sampled_table_assignments = np.array(samples['z'])
    # convert sampled cluster assignments from (num samples, num obs) to (num obs, num clusters)
    bins = np.arange(0, 2 + np.max(sampled_table_assignments))
    table_assignment_posteriors = np.stack([
        np.histogram(sampled_table_assignments[-1000:, obs_idx], bins=bins, density=True)[0]
        for obs_idx in range(num_obs)])
    table_assignment_posteriors_running_sum = np.cumsum(table_assignment_posteriors,
                                                        axis=0)

    # plt.imshow(sampled_table_assignments, cmap='jet')
    # plt.show()
    plt.scatter(x=observations[:, 0],
                y=observations[:, 1],
                c=np.argmax(table_assignment_posteriors, axis=1))
    plt.title(f'Num Samples = {num_samples}')
    plt.savefig(f'exp_01_mixture_of_gaussians/plots/hmc_gibbs_num_samples={num_samples}.png')
    # plt.show()
    plt.close()

    # returns classes assigned and centroids of corresponding classes
    hmc_gibbs_results = dict(
        table_assignment_posteriors=table_assignment_posteriors,
        table_assignment_posteriors_running_sum=table_assignment_posteriors_running_sum,
        parameters=params,
    )

    return hmc_gibbs_results


def variational_bayes(observations,
                      alpha: float,
                      max_iter: int = 8,  # same as DP-Means
                      n_init: int = 1):
    assert alpha > 0

    num_obs, obs_dim = observations.shape
    var_dp_gmm = BayesianGaussianMixture(
        n_components=num_obs,
        max_iter=max_iter,
        n_init=n_init,
        init_params='random',
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=alpha)
    var_dp_gmm.fit(observations)
    table_assignment_posteriors = var_dp_gmm.predict_proba(observations)
    table_assignment_posteriors_running_sum = np.cumsum(table_assignment_posteriors,
                                                        axis=0)
    params = dict(means=var_dp_gmm.means_,
                  covs=var_dp_gmm.covariances_)

    # returns classes assigned and centroids of corresponding classes
    variational_results = dict(
        table_assignment_posteriors=table_assignment_posteriors,
        table_assignment_posteriors_running_sum=table_assignment_posteriors_running_sum,
        parameters=params,
    )
    return variational_results

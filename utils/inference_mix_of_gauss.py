from jax import random
import jax.numpy as jnp
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
            table_assignment_posteriors_running_sum=table_assignment_posteriors_running_sum[obs_idx, :len(table_assignment_posterior)],
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
    # torch_observations = torch.from_numpy(observations)
    # torch_alpha = torch.from_numpy(np.array(alpha))
    # torch_gaussian_cov_scaling = torch.from_numpy(
    #     np.array(gaussian_cov_scaling))
    # torch_gaussian_mean_prior_cov_scaling = torch.from_numpy(
    #     np.array(gaussian_mean_prior_cov_scaling))

    # http://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.hmc_gibbs.DiscreteHMCGibbs

    # def model(probs, locs):
    #     c = numpyro.sample("c", numpyro.distributions.Categorical(probs))
    #     numpyro.sample("x", numpyro.distributions.Normal(locs[c], 0.5))

    # probs = jnp.array([0.15, 0.3, 0.3, 0.25])
    # locs = jnp.array([-2, 0, 2, 4])
    # kernel = numpyro.infer.DiscreteHMCGibbs(numpyro.infer.NUTS(model))
    # mcmc = numpyro.infer.MCMC(kernel, num_warmup=10, num_samples=11, progress_bar=True)
    # mcmc.run(random.PRNGKey(0), probs, locs)
    # mcmc.print_summary()
    # samples = mcmc.get_samples() #["x"]
    # assert abs(jnp.mean(samples['x']) - 1.3) < 0.1
    # assert abs(jnp.var(samples['x']) - 4.36) < 0.5

    if sampling_max_num_clusters is None:
        # multiply by 2 for safety
        sampling_max_num_clusters = 2 * int(alpha * np.log(1 + num_obs / alpha))

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

    # def model():
    #     x = numpyro.sample("x", numpyro.distributions.Normal(0.0, 2.0))
    #     z = numpyro.sample("z", numpyro.distributions.Normal(0.0, 2.0))
    #     numpyro.sample("obs", numpyro.distributions.Normal(x + z, 1.0), obs=jnp.array([1.0]))

    # def gibbs_fn(rng_key, gibbs_sites, hmc_sites):
    #     z = hmc_sites['z']
    #     new_x = numpyro.distributions.Normal(0.8 * (1 - z), jnp.sqrt(0.8)).sample(rng_key)
    #     return {'x': new_x}

    hmc_kernel = numpyro.infer.NUTS(model)
    # kernel = numpyro.infer.HMCGibbs(hmc_kernel, gibbs_fn=gibbs_fn, gibbs_sites=['z'])
    kernel = numpyro.infer.DiscreteHMCGibbs(inner_kernel=hmc_kernel)
    mcmc = numpyro.infer.MCMC(kernel, num_warmup=100, num_samples=89, progress_bar=True)
    mcmc.run(random.PRNGKey(0), obs=observations)
    mcmc.print_summary()
    samples = mcmc.get_samples() #["z"]
    print(10)


    # def mix_weights(beta):
    #     beta1m_cumprod = (1 - beta).cumprod(-1)
    #     return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)
    #
    # def gibbs_fn():
    #     pass
    #
    # def model(obs):
    #     with pyro.plate('beta_plate', sampling_max_num_clusters - 1):
    #         beta = pyro.sample(
    #             'beta',
    #             pyro.distributions.Beta(1, torch_alpha))
    #
    #     with pyro.plate('mean_plate', sampling_max_num_clusters):
    #         mean = pyro.sample(
    #             'mean',
    #             pyro.distributions.MultivariateNormal(
    #                 torch.zeros(obs_dim),
    #                 torch_gaussian_mean_prior_cov_scaling * torch.eye(obs_dim)))
    #
    #     with pyro.plate('data', num_obs):
    #         z = pyro.sample(
    #             'z',
    #             pyro.distributions.Categorical(mix_weights(beta=beta)).mask(False))
    #         pyro.sample(
    #             'obs',
    #             pyro.distributions.MultivariateNormal(
    #                 mean[z],
    #                 torch_gaussian_cov_scaling * torch.eye(obs_dim)),
    #             obs=obs)


def sampling_nuts(observations,
                  num_samples: int,
                  alpha: float,
                  gaussian_cov_scaling: int,
                  gaussian_mean_prior_cov_scaling: float,
                  variational_max_num_clusters=None,
                  burn_fraction: float = 0.25):

    assert alpha > 0

    num_obs, obs_dim = observations.shape

    if variational_max_num_clusters is None:
        # multiply by 2 for safety
        variational_max_num_clusters = 2*int(alpha * np.log(1 + num_obs / alpha))

    table_assignments = np.zeros((variational_max_num_clusters,
                                  variational_max_num_clusters))
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

    # https://docs.pymc.io/notebooks/sampling_compound_step.html
    with pm.Model() as model:
        pm_beta = pm.Beta("beta", 1.0, alpha, shape=variational_max_num_clusters)
        pm_w = pm.Deterministic("w", stick_breaking(pm_beta))
        pm_cluster_means = [
            pm.MvNormal(f'cluster_mean_{cluster_idx}',
                        mu=pm.floatX(np.zeros(obs_dim)),
                        cov=pm.floatX(gaussian_mean_prior_cov_scaling * np.eye(obs_dim)),
                        shape=(obs_dim,))
            for cluster_idx in range(variational_max_num_clusters)]
        pm_obs = pm.DensityDist('obs',
                                logp_gmix(cluster_means=pm_cluster_means,
                                          cluster_weights=pm_w,
                                          tau=gaussian_cov_scaling * np.eye(obs_dim)),
                                observed=observations)

        pm_trace = pm.sample(draws=num_samples,
                             tune=2500,
                             chains=4,
                             target_accept=0.9,
                             random_seed=1)

    # TODO: figure out number of clusters

    # figure out which point belongs to which cluster
    # shape (max_num_clusters, num_samples, obs_dim)
    cluster_means_samples = np.stack([pm_trace[f'cluster_mean_{cluster_idx}']
                                      for cluster_idx in range(variational_max_num_clusters)])

    # burn the first samples
    cluster_means_samples = cluster_means_samples[:, int(burn_fraction * num_samples):, :]
    cluster_means = np.mean(cluster_means_samples, axis=1)

    distance_obs_to_cluster_means = cdist(observations, cluster_means)
    table_assignment_posteriors = np.exp(-np.square(distance_obs_to_cluster_means) / 2)
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

    nuts_sampling_results = dict(
        table_assignment_posteriors=table_assignment_posteriors,
        table_assignment_posteriors_running_sum=table_assignment_posteriors_running_sum,
        parameters=params)

    return nuts_sampling_results


def sampling_nuts_pyro(observations,
                       num_samples: int,
                       alpha: float,
                       gaussian_cov_scaling: int,
                       gaussian_mean_prior_cov_scaling: float,
                       sampling_max_num_clusters=None,
                       burn_fraction: float = 0.25):

    pyro.enable_validation(True)
    pyro.set_rng_seed(0)

    num_obs, obs_dim = observations.shape
    torch_observations = torch.from_numpy(observations)
    torch_alpha = torch.from_numpy(np.array(alpha))
    torch_gaussian_cov_scaling = torch.from_numpy(
        np.array(gaussian_cov_scaling))
    torch_gaussian_mean_prior_cov_scaling = torch.from_numpy(
        np.array(gaussian_mean_prior_cov_scaling))

    if sampling_max_num_clusters is None:
        # multiply by 2 for safety
        sampling_max_num_clusters = 2 * int(alpha * np.log(1 + num_obs / alpha))

    def mix_weights(beta):
        beta1m_cumprod = (1 - beta).cumprod(-1)
        # first step: add a 1 to the end of beta
        # second step: add a 1 to the start of beta1m_cumprod
        # then multiply the two
        return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)

    def model(obs):
        with pyro.plate('beta_plate', sampling_max_num_clusters - 1):
            beta = pyro.sample(
                'beta',
                pyro.distributions.Beta(1, torch_alpha))

        with pyro.plate('mean_plate', sampling_max_num_clusters):
            mean = pyro.sample(
                'mean',
                pyro.distributions.MultivariateNormal(
                    torch.zeros(obs_dim),
                    torch_gaussian_mean_prior_cov_scaling * torch.eye(obs_dim)))

        with pyro.plate('data', num_obs):
            z = pyro.sample(
                'z',
                pyro.distributions.Categorical(mix_weights(beta=beta)).mask(False))
            pyro.sample(
                'obs',
                pyro.distributions.MultivariateNormal(
                    mean[z],
                    torch_gaussian_cov_scaling * torch.eye(obs_dim)),
                obs=obs)

    # serving_model = pyro.infer.infer_discrete(model, first_available_dim=-1)
    nuts_kernel = pyro.infer.NUTS(model=model, adapt_step_size=True)
    mcmc = pyro.infer.MCMC(
        kernel=nuts_kernel,
        num_samples=num_samples,
        warmup_steps=3)
    mcmc.run(torch_observations)
    samples = mcmc.get_samples()

    import matplotlib.pyplot as plt
    for cluster_idx in range(sampling_max_num_clusters):
        plt.scatter(samples['mean'][-1, cluster_idx, 0],
                 samples['mean'][-1, cluster_idx, 1],
                 label=cluster_idx)
    plt.legend()
    plt.show()
    print(10)


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

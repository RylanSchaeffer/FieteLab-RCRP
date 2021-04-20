import jax.numpy as jnp
import jax.random
import numpy as np
import numpyro
import numpyro.infer
import numpyro.distributions
import scipy
import scipy.special
import sklearn.mixture
import torch
import torch.distributions
import torch.nn
import torch.nn.functional
import torch.optim

from utils.helpers import assert_torch_no_nan_no_inf, torch_logits_to_probs, torch_probs_to_logits

torch.set_default_tensor_type('torch.DoubleTensor')


def beta(a, b):
    result = torch.exp(log_beta(a, b))
    return result


def create_new_cluster_params_bernoulli(torch_observation,
                                        obs_idx,
                                        cluster_parameters):
    # data is necessary to not break backprop
    # see https://stackoverflow.com/questions/53819383/how-to-assign-a-new-value-to-a-pytorch-variable-without-breaking-backpropagation
    raise NotImplementedError


def create_new_cluster_params_continuous_bernoulli(torch_observation,
                                                   obs_idx,
                                                   cluster_parameters):
    # data is necessary to not break backprop
    # see https://stackoverflow.com/questions/53819383/how-to-assign-a-new-value-to-a-pytorch-variable-without-breaking-backpropagation
    assert_torch_no_nan_no_inf(torch_observation)
    torch_obs_as_logits = torch_probs_to_logits(torch_observation)
    assert_torch_no_nan_no_inf(torch_obs_as_logits)
    cluster_parameters['logits'].data[obs_idx, :] = torch_obs_as_logits
    assert_torch_no_nan_no_inf(cluster_parameters['logits'].data[:obs_idx + 1])


def create_new_cluster_params_dirichlet_multinomial(torch_observation,
                                                    obs_idx,
                                                    cluster_parameters):
    # data is necessary to not break backprop
    # see https://stackoverflow.com/questions/53819383/how-to-assign-a-new-value-to-a-pytorch-variable-without-breaking-backpropagation
    assert_torch_no_nan_no_inf(torch_observation)
    epsilon = 10.
    cluster_parameters['topics_concentrations'].data[obs_idx, :] = torch_observation + epsilon


def create_new_cluster_params_multivariate_normal(torch_observation,
                                                  obs_idx,
                                                  cluster_parameters):
    # data is necessary to not break backprop
    # see https://stackoverflow.com/questions/53819383/how-to-assign-a-new-value-to-a-pytorch-variable-without-breaking-backpropagation
    assert_torch_no_nan_no_inf(torch_observation)
    cluster_parameters['means'].data[obs_idx, :] = torch_observation
    cluster_parameters['stddevs'].data[obs_idx, :, :] = torch.eye(torch_observation.shape[0])


def dp_means(observations,
             concentration_param: float,
             likelihood_model: str,
             learning_rate: float,
             num_passes: int):
    # if num_passes = 1, then this is "online."
    # if num_passes > 1, then this if "offline"
    assert concentration_param > 0
    assert isinstance(num_passes, int)
    assert num_passes > 0

    # set learning rate to 0; unused
    learning_rate = 0

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

            # if smallest distance greater than cutoff concentration_param, create new cluster:
            if np.min(distances) > concentration_param:

                # increment number of clusters by 1:
                num_clusters += 1

                # centroid of new cluster = new sample
                means[num_clusters - 1, :] = observations[obs_idx, :]
                table_assignments[obs_idx, num_clusters - 1] = 1.

            else:

                # If the smallest distance is less than the cutoff concentration_param, assign point
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


def likelihood_continuous_bernoulli(torch_observation,
                                    obs_idx,
                                    cluster_parameters):
    cont_bern = torch.distributions.ContinuousBernoulli(
        logits=cluster_parameters['logits'][:obs_idx + 1])
    log_likelihoods_per_latent_per_obs_dim = cont_bern.log_prob(value=torch_observation)
    likelihoods_per_latent_per_obs_dim = torch.exp(log_likelihoods_per_latent_per_obs_dim)

    # compute Continuous Bernoulli likelihoods
    # normalizing_const = torch.divide(
    #     2. * torch.arctanh(1. - 2. * logits[:obs_idx+1]),
    #     1. - 2. * logits[:obs_idx+1])
    # normalizing_const[torch.isnan(normalizing_const)] = 2.
    # likelihoods_per_latent_per_obs_dim = torch.multiply(
    #     normalizing_const,
    #     torch.multiply(
    #         torch.pow(logits[:obs_idx + 1], torch_observation),
    #         torch.pow(1. - logits[:obs_idx + 1], 1. - torch_observation)))

    likelihoods_per_latent = torch.prod(likelihoods_per_latent_per_obs_dim, dim=1)
    log_likelihoods_per_latent = torch.sum(log_likelihoods_per_latent_per_obs_dim, dim=1)

    return likelihoods_per_latent, log_likelihoods_per_latent


def likelihood_dirichlet_multinomial(torch_observation,
                                     obs_idx,
                                     cluster_parameters):
    # Approach 1: Multinomial-Dirichlet
    words_in_doc = torch.sum(torch_observation)
    # previous version, copied from numpy
    # log_numerator = torch.log(words_in_doc) + torch.log(scipy.special.beta(
    #     torch.sum(cluster_parameters['topics_concentrations'], axis=1),
    #     words_in_doc))
    total_concentrations_per_latent = torch.sum(
        cluster_parameters['topics_concentrations'][:obs_idx + 1],
        dim=1)
    # compute the log Beta using defn of Beta(a,b)=Gamma(a)*Gamma(b)/Gamma(a+b)
    log_numerator = torch.log(words_in_doc) \
                    + log_beta(a=total_concentrations_per_latent, b=words_in_doc)

    # shape (doc idx, vocab size)
    log_beta_terms = log_beta(
        a=cluster_parameters['topics_concentrations'][:obs_idx + 1],
        b=torch_observation,
    )
    log_x_times_beta_terms = torch.add(
        log_beta_terms,
        torch.log(torch_observation))
    # beta numerically can create infs if x is 0, even though 0*Beta(., 0) should be 0
    # consequently, filter these out by setting equal to 0
    log_x_times_beta_terms[torch.isnan(log_x_times_beta_terms)] = 0.
    # shape (max num latents, )
    log_denominator = torch.sum(log_x_times_beta_terms, dim=1)
    assert_torch_no_nan_no_inf(log_denominator)
    log_likelihoods_per_latent = log_numerator - log_denominator
    assert_torch_no_nan_no_inf(log_likelihoods_per_latent)
    likelihoods_per_latent = torch.exp(log_likelihoods_per_latent)

    return likelihoods_per_latent, log_likelihoods_per_latent


def likelihood_discrete_bernoulli(torch_observation,
                                  obs_idx,
                                  cluster_parameters):
    raise NotImplementedError
    # return likelihoods_per_latent_per_obs_dim, log_likelihoods_per_latent_per_obs_dim


def likelihood_multivariate_normal(torch_observation,
                                   obs_idx,
                                   cluster_parameters):
    # TODO: figure out how to do gradient descent using the post-grad step means
    # covariances = torch.stack([
    #     torch.matmul(stddev, stddev.T) for stddev in cluster_parameters['stddevs']])
    #
    obs_dim = torch_observation.shape[0]
    covariances = torch.stack([torch.matmul(torch.eye(obs_dim), torch.eye(obs_dim).T)
                               for stddev in cluster_parameters['stddevs']]).double()

    mv_normal = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=cluster_parameters['means'][:obs_idx + 1],
        covariance_matrix=covariances[:obs_idx + 1],
        # scale_tril=cluster_parameters['stddevs'][:obs_idx + 1],
    )
    log_likelihoods_per_latent = mv_normal.log_prob(value=torch_observation)
    likelihoods_per_latent = torch.exp(log_likelihoods_per_latent)
    return likelihoods_per_latent, log_likelihoods_per_latent


def log_beta(a, b):
    log_gamma_a = torch.lgamma(a)
    log_gamma_b = torch.lgamma(b)
    log_gamma_a_plus_b = torch.lgamma(a + b)
    result = log_gamma_a + log_gamma_b - log_gamma_a_plus_b
    return result


def mix_weights(beta):
    # stick-breaking construction of DP
    beta1m_cumprod = jnp.cumprod(1 - beta, axis=-1)
    term1 = jnp.pad(beta, (0, 1), mode='constant', constant_values=1.)
    term2 = jnp.pad(beta1m_cumprod, (1, 0), mode='constant', constant_values=1.)
    return jnp.multiply(term1, term2)


def online_crp(observations,
               concentration_param: float,
               likelihood_model: str,
               learning_rate):
    # Online Chinese Restaurant Process
    # Liu, Tsai, Lee (2014)

    assert concentration_param > 0
    assert likelihood_model in {'multivariate_normal', 'dirichlet_multinomial',
                                'bernoulli', 'continuous_bernoulli'}
    num_obs, obs_dim = observations.shape

    # The recursion does not require recording the full history of priors/posteriors
    # but we record the full history for subsequent analysis
    max_num_latents = num_obs
    table_assignment_priors = torch.zeros(
        (num_obs, max_num_latents),
        dtype=torch.float64,
        requires_grad=False)
    table_assignment_priors[0, 0] = 1.

    table_assignment_posteriors = torch.zeros(
        (num_obs, max_num_latents),
        dtype=torch.float64,
        requires_grad=False)

    table_assignment_posteriors_running_sum = torch.zeros(
        (num_obs, max_num_latents),
        dtype=torch.float64,
        requires_grad=False)

    num_table_posteriors = torch.zeros(
        (num_obs, max_num_latents),
        dtype=torch.float64,
        requires_grad=False)

    if likelihood_model == 'multivariate_normal':
        cluster_parameters = dict(
            means=torch.full(
                size=(max_num_latents, obs_dim),
                fill_value=np.nan,
                dtype=torch.float64,
                requires_grad=True),
            stddevs=torch.full(
                size=(max_num_latents, obs_dim, obs_dim),
                fill_value=np.nan,
                dtype=torch.float64,
                requires_grad=True),
        )
        create_new_cluster_params_fn = create_new_cluster_params_multivariate_normal
        likelihood_fn = likelihood_multivariate_normal
    elif likelihood_model == 'dirichlet_multinomial':
        cluster_parameters = dict(
            topics_concentrations=torch.full(
                size=(max_num_latents, obs_dim),
                fill_value=np.nan,
                dtype=torch.float64,
                requires_grad=True),
        )

        create_new_cluster_params_fn = create_new_cluster_params_dirichlet_multinomial
        likelihood_fn = likelihood_dirichlet_multinomial
    elif likelihood_model == 'continuous_bernoulli':
        # need to use logits, otherwise gradient descent will carry parameters outside
        # valid interval
        cluster_parameters = dict(
            logits=torch.full(
                size=(max_num_latents, obs_dim),
                fill_value=np.nan,
                dtype=torch.float64,
                requires_grad=True)
        )
        create_new_cluster_params_fn = create_new_cluster_params_continuous_bernoulli
        likelihood_fn = likelihood_continuous_bernoulli

        # make sure no observation is 0 or 1 by adding epsilon
        epsilon = 1e-2
        observations[observations == 1.] -= epsilon
        observations[observations == 0.] += epsilon
    else:
        raise NotImplementedError

    # optimizer = torch.optim.SGD(params=[logits], lr=1.)
    optimizer = torch.optim.SGD(params=cluster_parameters.values(), lr=1.)

    # needed later for error checking
    one_tensor = torch.Tensor([1.]).double()

    torch_observations = torch.from_numpy(observations)
    for obs_idx, torch_observation in enumerate(torch_observations):

        # create new params for possible cluster, centered at that point
        create_new_cluster_params_fn(
            torch_observation=torch_observation,
            obs_idx=obs_idx,
            cluster_parameters=cluster_parameters)

        if obs_idx == 0:
            # first customer has to go at first table
            table_assignment_priors[obs_idx, 0] = 1.
            table_assignment_posteriors[obs_idx, 0] = 1.
            num_table_posteriors[obs_idx, 0] = 1.

            # update running sum of posteriors
            table_assignment_posteriors_running_sum[obs_idx, :] = torch.add(
                table_assignment_posteriors_running_sum[obs_idx - 1, :],
                table_assignment_posteriors[obs_idx, :])
            assert torch.allclose(torch.sum(table_assignment_posteriors_running_sum[obs_idx, :]),
                                  torch.Tensor([obs_idx + 1]).double())
        else:
            # construct prior
            table_assignment_prior = torch.clone(
                table_assignment_posteriors_running_sum[obs_idx - 1, :obs_idx + 1])
            # we don't subtract 1 because Python uses 0-based indexing
            assert torch.allclose(torch.sum(table_assignment_prior), torch.Tensor([obs_idx]).double())
            # add new table probability
            table_assignment_prior[1:] += concentration_param * torch.clone(
                num_table_posteriors[obs_idx - 1, :obs_idx])
            # renormalize
            table_assignment_prior /= (concentration_param + obs_idx)
            assert torch.allclose(torch.sum(table_assignment_prior), one_tensor)

            # record latent prior
            table_assignment_priors[obs_idx, :len(table_assignment_prior)] = table_assignment_prior

            optimizer.zero_grad()

            # infer posteriors using parameters
            likelihoods_per_latent, log_likelihoods_per_latent = likelihood_fn(
                torch_observation=torch_observation,
                obs_idx=obs_idx,
                cluster_parameters=cluster_parameters)
            assert torch.all(~torch.isnan(likelihoods_per_latent[:obs_idx + 1]))
            assert torch.all(~torch.isnan(log_likelihoods_per_latent[:obs_idx + 1]))

            unnormalized_table_assignment_posterior = torch.multiply(
                likelihoods_per_latent.detach(),
                table_assignment_prior)
            table_assignment_posterior = unnormalized_table_assignment_posterior / torch.sum(
                unnormalized_table_assignment_posterior)

            # sample from table assignment posterior
            sampled_table_assignment = torch.distributions.categorical.Categorical(
                probs=table_assignment_posterior).sample()
            table_assignment_posterior = torch.nn.functional.one_hot(
                sampled_table_assignment,
                num_classes=table_assignment_posterior.shape[0]).double()
            assert torch.allclose(torch.sum(table_assignment_posterior), one_tensor)

            # record latent posterior
            table_assignment_posteriors[obs_idx, :obs_idx + 1] = table_assignment_posterior

            # update running sum of posteriors
            table_assignment_posteriors_running_sum[obs_idx, :] = torch.add(
                table_assignment_posteriors_running_sum[obs_idx - 1, :],
                table_assignment_posteriors[obs_idx, :])
            assert torch.allclose(torch.sum(table_assignment_posteriors_running_sum[obs_idx, :]),
                                  torch.Tensor([obs_idx + 1]).double())

            # next, update parameters
            # Note: log likelihood is all we need for optimization because
            # log p(x, z; params) = log p(x|z; params) + log p(z)
            # and the second is constant w.r.t. params gradient
            loss = torch.mean(log_likelihoods_per_latent)
            loss.backward()

            # instead of typical dynamics:
            #       p_k <- p_k + (obs - p_k) / number of obs assigned to kth cluster
            # we use the new dynamics
            #       p_k <- p_k + posterior(obs belongs to kth cluster) * (obs - p_k) / total mass on kth cluster
            # that effectively means the learning rate should be this scaled_prefactor
            scaled_learning_rate = learning_rate * torch.divide(
                table_assignment_posteriors[obs_idx, :],
                table_assignment_posteriors_running_sum[obs_idx, :])
            scaled_learning_rate[torch.isnan(scaled_learning_rate)] = 0.
            scaled_learning_rate[torch.isinf(scaled_learning_rate)] = 0.

            # don't update the newest cluster
            scaled_learning_rate[obs_idx] = 0.

            for param_descr, param_tensor in cluster_parameters.items():
                # the scaled learning rate has shape (num latents,) aka (num obs,)
                # we need to create extra dimensions of size 1 for broadcasting to work
                # because param_tensor can have variable number of dimensions e.g. (num obs, obs dim)
                # for mean vs (num obs, obs dim, obs dim) for covariance, we need to dynamically
                # add the corect number of dimensions
                reshaped_scaled_learning_rate = scaled_learning_rate.view(
                    [scaled_learning_rate.shape[0]] + [1 for _ in range(len(param_tensor.shape[1:]))])
                if param_tensor.grad is None:
                    continue
                else:
                    scaled_param_tensor_grad = torch.multiply(
                        reshaped_scaled_learning_rate,
                        param_tensor.grad)
                    param_tensor.data += scaled_param_tensor_grad
                    assert_torch_no_nan_no_inf(param_tensor.data[:obs_idx + 1])

            # update posterior over number of tables using posterior over customer seat
            num_table_posteriors[obs_idx, torch.sum(table_assignment_posteriors_running_sum[obs_idx] != 0) - 1] = 1.

    online_crp_results = dict(
        table_assignment_priors=table_assignment_priors.numpy(),
        table_assignment_posteriors=table_assignment_posteriors.numpy(),
        table_assignment_posteriors_running_sum=table_assignment_posteriors_running_sum.numpy(),
        num_table_posteriors=num_table_posteriors.numpy(),
        parameters={k: v.detach().numpy() for k, v in cluster_parameters.items()},
        # TODO: why do I have to detach here?
    )

    return online_crp_results


def recursive_crp(observations,
                  concentration_param: float,
                  likelihood_model: str,
                  learning_rate,
                  num_em_steps: int = 3):
    assert concentration_param > 0
    assert likelihood_model in {'multivariate_normal', 'dirichlet_multinomial',
                                'bernoulli', 'continuous_bernoulli'}
    num_obs, obs_dim = observations.shape

    # The recursion does not require recording the full history of priors/posteriors
    # but we record the full history for subsequent analysis
    max_num_latents = num_obs
    table_assignment_priors = torch.zeros(
        (num_obs, max_num_latents),
        dtype=torch.float64,
        requires_grad=False)
    table_assignment_priors[0, 0] = 1.

    table_assignment_posteriors = torch.zeros(
        (num_obs, max_num_latents),
        dtype=torch.float64,
        requires_grad=False)

    table_assignment_posteriors_running_sum = torch.zeros(
        (num_obs, max_num_latents),
        dtype=torch.float64,
        requires_grad=False)

    num_table_posteriors = torch.zeros(
        (num_obs, max_num_latents),
        dtype=torch.float64,
        requires_grad=False)

    if likelihood_model == 'continuous_bernoulli':
        # need to use logits, otherwise gradient descent will carry parameters outside
        # valid interval
        cluster_parameters = dict(
            logits=torch.full(
                size=(max_num_latents, obs_dim),
                fill_value=np.nan,
                dtype=torch.float64,
                requires_grad=True)
        )
        create_new_cluster_params_fn = create_new_cluster_params_continuous_bernoulli
        likelihood_fn = likelihood_continuous_bernoulli

        # make sure no observation is 0 or 1 by adding epsilon
        epsilon = 1e-2
        observations[observations == 1.] -= epsilon
        observations[observations == 0.] += epsilon
    elif likelihood_model == 'dirichlet_multinomial':
        cluster_parameters = dict(
            topics_concentrations=torch.full(
                size=(max_num_latents, obs_dim),
                fill_value=np.nan,
                dtype=torch.float64,
                requires_grad=True),
        )
        create_new_cluster_params_fn = create_new_cluster_params_dirichlet_multinomial
        likelihood_fn = likelihood_dirichlet_multinomial
    elif likelihood_model == 'multivariate_normal':
        cluster_parameters = dict(
            means=torch.full(
                size=(max_num_latents, obs_dim),
                fill_value=np.nan,
                dtype=torch.float64,
                requires_grad=True),
            stddevs=torch.full(
                size=(max_num_latents, obs_dim, obs_dim),
                fill_value=np.nan,
                dtype=torch.float64,
                requires_grad=True),
        )
        create_new_cluster_params_fn = create_new_cluster_params_multivariate_normal
        likelihood_fn = likelihood_multivariate_normal
    else:
        raise NotImplementedError

    # optimizer = torch.optim.SGD(params=[logits], lr=1.)
    optimizer = torch.optim.SGD(params=cluster_parameters.values(), lr=1.)

    # needed later for error checking
    one_tensor = torch.Tensor([1.]).double()

    torch_observations = torch.from_numpy(observations)
    for obs_idx, torch_observation in enumerate(torch_observations):

        # create new params for possible cluster, centered at that point
        create_new_cluster_params_fn(
            torch_observation=torch_observation,
            obs_idx=obs_idx,
            cluster_parameters=cluster_parameters)

        if obs_idx == 0:
            # first customer has to go at first table
            table_assignment_priors[obs_idx, 0] = 1.
            table_assignment_posteriors[obs_idx, 0] = 1.
            num_table_posteriors[obs_idx, 0] = 1.

            # update running sum of posteriors
            table_assignment_posteriors_running_sum[obs_idx, :] = torch.add(
                table_assignment_posteriors_running_sum[obs_idx - 1, :],
                table_assignment_posteriors[obs_idx, :])
            assert torch.allclose(torch.sum(table_assignment_posteriors_running_sum[obs_idx, :]),
                                  torch.Tensor([obs_idx + 1]).double())
        else:
            # construct prior
            table_assignment_prior = torch.clone(
                table_assignment_posteriors_running_sum[obs_idx - 1, :obs_idx + 1])
            # we don't subtract 1 because Python uses 0-based indexing
            assert torch.allclose(torch.sum(table_assignment_prior), torch.Tensor([obs_idx]).double())
            # add new table probability
            table_assignment_prior[1:] += concentration_param * torch.clone(
                num_table_posteriors[obs_idx - 1, :obs_idx])
            # renormalize
            table_assignment_prior /= (concentration_param + obs_idx)
            assert torch.allclose(torch.sum(table_assignment_prior), one_tensor)

            # record latent prior
            table_assignment_priors[obs_idx, :len(table_assignment_prior)] = table_assignment_prior

            for em_idx in range(num_em_steps):

                optimizer.zero_grad()

                # E step: infer posteriors using parameters
                likelihoods_per_latent, log_likelihoods_per_latent = likelihood_fn(
                    torch_observation=torch_observation,
                    obs_idx=obs_idx,
                    cluster_parameters=cluster_parameters)
                assert torch.all(~torch.isnan(likelihoods_per_latent[:obs_idx + 1]))
                assert torch.all(~torch.isnan(log_likelihoods_per_latent[:obs_idx + 1]))

                unnormalized_table_assignment_posterior = torch.multiply(
                    likelihoods_per_latent.detach(),
                    table_assignment_prior)
                table_assignment_posterior = unnormalized_table_assignment_posterior / torch.sum(
                    unnormalized_table_assignment_posterior)
                assert torch.allclose(torch.sum(table_assignment_posterior), one_tensor)

                # record latent posterior
                table_assignment_posteriors[obs_idx, :len(table_assignment_posterior)] = table_assignment_posterior

                # update running sum of posteriors
                table_assignment_posteriors_running_sum[obs_idx, :] = torch.add(
                    table_assignment_posteriors_running_sum[obs_idx - 1, :],
                    table_assignment_posteriors[obs_idx, :])
                assert torch.allclose(torch.sum(table_assignment_posteriors_running_sum[obs_idx, :]),
                                      torch.Tensor([obs_idx + 1]).double())

                # M step: update parameters
                # Note: log likelihood is all we need for optimization because
                # log p(x, z; params) = log p(x|z; params) + log p(z)
                # and the second is constant w.r.t. params gradient
                loss = torch.mean(log_likelihoods_per_latent)
                loss.backward()

                # instead of typical dynamics:
                #       p_k <- p_k + (obs - p_k) / number of obs assigned to kth cluster
                # we use the new dynamics
                #       p_k <- p_k + posterior(obs belongs to kth cluster) * (obs - p_k) / total mass on kth cluster
                # that effectively means the learning rate should be this scaled_prefactor
                scaled_learning_rate = learning_rate * torch.divide(
                    table_assignment_posteriors[obs_idx, :],
                    table_assignment_posteriors_running_sum[obs_idx, :]) / num_em_steps
                scaled_learning_rate[torch.isnan(scaled_learning_rate)] = 0.
                scaled_learning_rate[torch.isinf(scaled_learning_rate)] = 0.

                # don't update the newest cluster
                scaled_learning_rate[obs_idx] = 0.

                for param_descr, param_tensor in cluster_parameters.items():
                    # the scaled learning rate has shape (num latents,) aka (num obs,)
                    # we need to create extra dimensions of size 1 for broadcasting to work
                    # because param_tensor can have variable number of dimensions e.g. (num obs, obs dim)
                    # for mean vs (num obs, obs dim, obs dim) for covariance, we need to dynamically
                    # add the corect number of dimensions
                    reshaped_scaled_learning_rate = scaled_learning_rate.view(
                        [scaled_learning_rate.shape[0]] + [1 for _ in range(len(param_tensor.shape[1:]))])
                    if param_tensor.grad is None:
                        continue
                    else:
                        scaled_param_tensor_grad = torch.multiply(
                            reshaped_scaled_learning_rate,
                            param_tensor.grad)
                        param_tensor.data += scaled_param_tensor_grad
                        assert_torch_no_nan_no_inf(param_tensor.data[:obs_idx + 1])

            # # previous approach with time complexity O(t^2)
            # # update posterior over number of tables using posterior over customer seat
            # for k1, p_z_t_equals_k1 in enumerate(table_assignment_posteriors[obs_idx, :obs_idx + 1]):
            #     for k2, p_prev_num_tables_equals_k2 in enumerate(num_table_posteriors[obs_idx - 1, :obs_idx + 1]):
            #         # advance number of tables by 1 if customer seating > number of current tables
            #         if k1 > k2 + 1:
            #             num_table_posteriors.data[obs_idx, k2 + 1] += p_z_t_equals_k1 * p_prev_num_tables_equals_k2
            #         # customer allocated to previous table
            #         elif k1 <= k2:
            #             num_table_posteriors.data[obs_idx, k2] += p_z_t_equals_k1 * p_prev_num_tables_equals_k2
            #         # create new table
            #         elif k1 == k2 + 1:
            #             num_table_posteriors.data[obs_idx, k1] += p_z_t_equals_k1 * p_prev_num_tables_equals_k2
            #         else:
            #             raise ValueError
            # assert torch.allclose(torch.sum(num_table_posteriors[obs_idx, :]), one_tensor)

            # new approach with time complexity O(t)
            # update posterior over number of tables using posterior over customer seat
            cum_table_assignment_posterior = torch.cumsum(
                table_assignment_posteriors[obs_idx, :obs_idx + 1],
                dim=0)
            one_minus_cum_table_assignment_posterior = 1. - cum_table_assignment_posterior
            prev_table_posterior = num_table_posteriors[obs_idx - 1, :obs_idx]
            num_table_posteriors[obs_idx, :obs_idx] += torch.multiply(
                cum_table_assignment_posterior[:-1],
                prev_table_posterior)
            num_table_posteriors[obs_idx, 1:obs_idx + 1] += torch.multiply(
                one_minus_cum_table_assignment_posterior[:-1],
                prev_table_posterior)
            assert torch.allclose(torch.sum(num_table_posteriors[obs_idx, :]), one_tensor)

    bayesian_recursion_results = dict(
        table_assignment_priors=table_assignment_priors.numpy(),
        table_assignment_posteriors=table_assignment_posteriors.numpy(),
        table_assignment_posteriors_running_sum=table_assignment_posteriors_running_sum.numpy(),
        num_table_posteriors=num_table_posteriors.numpy(),
        parameters={k: v.detach().numpy() for k, v in cluster_parameters.items()},
    )

    return bayesian_recursion_results


def run_inference_alg(inference_alg_str,
                      observations,
                      concentration_param,
                      likelihood_model,
                      learning_rate):
    # allow algorithm-specific arguments to inference alg function
    inference_alg_kwargs = dict()

    # select inference alg and add kwargs as necessary
    if inference_alg_str == 'R-CRP':
        inference_alg_fn = recursive_crp
    elif inference_alg_str == 'Online CRP':
        inference_alg_fn = online_crp
    elif inference_alg_str == 'SUSG':
        inference_alg_fn = sequential_updating_and_greedy_search
    elif inference_alg_str == 'VSUSG':
        inference_alg_fn = variational_sequential_updating_and_greedy_search
    elif inference_alg_str.startswith('DP-Means'):
        inference_alg_fn = dp_means
        if inference_alg_str.endswith('(offline)'):
            inference_alg_kwargs['num_passes'] = 8  # same as Kulis and Jordan
        elif inference_alg_str.endswith('(online)'):
            inference_alg_kwargs['num_passes'] = 1
        else:
            raise ValueError('Invalid DP Means')
    elif inference_alg_str.startswith('HMC-Gibbs'):
        #     gaussian_cov_scaling=gaussian_cov_scaling,
        #     gaussian_mean_prior_cov_scaling=gaussian_mean_prior_cov_scaling,
        inference_alg_fn = sampling_hmc_gibbs

        # Suppose inference_alg_str is 'HMC-Gibbs (5000 Samples)'. We want to extract
        # the number of samples. To do this, we use the following
        num_samples = int(inference_alg_str.split(' ')[1][1:])
        inference_alg_kwargs['num_samples'] = num_samples
        inference_alg_kwargs['truncation_num_clusters'] = 12

        if likelihood_model == 'dirichlet_multinomial':
            inference_alg_kwargs['model_params'] = dict(
                dirichlet_concentration_param=10.)  # same as R-CRP
        elif likelihood_model == 'multivariate_normal':
            # Note: these are the ground truth parameters
            inference_alg_kwargs['model_params'] = dict(
                gaussian_mean_prior_cov_scaling=6,
                gaussian_cov_scaling=0.3)
        else:
            raise ValueError(f'Unknown likelihood model: {likelihood_model}')
    elif inference_alg_str.startswith('SVI'):
        inference_alg_fn = stochastic_variational_inference
        learning_rate = 5e-4
        # suppose the inference_alg_str is 'SVI (5k Steps)'
        substrs = inference_alg_str.split(' ')
        num_steps = 1000 * int(substrs[1][1:-1])
        inference_alg_kwargs['num_steps'] = num_steps
        # Note: these are the ground truth parameters
        if likelihood_model == 'dirichlet_multinomial':
            inference_alg_kwargs['model_params'] = dict(
                dirichlet_concentration_param=10.)  # same as R-CRP
        elif likelihood_model == 'multivariate_normal':
            inference_alg_kwargs['model_params'] = dict(
                gaussian_mean_prior_cov_scaling=6.,
                gaussian_cov_scaling=0.3)
        else:
            raise ValueError
    elif inference_alg_str.startswith('Variational Bayes'):
        inference_alg_fn = variational_bayes
        # Suppose we have an algorithm string 'Variational Bayes (10 Init, 10 Iterations)',
        substrs = inference_alg_str.split(' ')
        num_initializations = int(substrs[2][1:])
        max_iters = int(substrs[4])
        inference_alg_kwargs['num_initializations'] = num_initializations
        inference_alg_kwargs['max_iter'] = max_iters
    else:
        raise ValueError(f'Unknown inference algorithm: {inference_alg_str}')

    # run inference algorithm
    inference_alg_results = inference_alg_fn(
        observations=observations,
        concentration_param=concentration_param,
        likelihood_model=likelihood_model,
        learning_rate=learning_rate,
        **inference_alg_kwargs)

    return inference_alg_results


def sampling_hmc_gibbs(observations,
                       concentration_param: float,
                       likelihood_model: str,
                       learning_rate: float,
                       num_samples: int,
                       model_params: dict,
                       truncation_num_clusters: int = None):
    # why sampling is so hard:
    # https://mc-stan.org/users/documentation/case-studies/identifying_mixture_models.html

    # learning rate is not used

    num_obs, obs_dim = observations.shape
    if truncation_num_clusters is None:
        # multiply by 2 for safety
        truncation_num_clusters = 2 * int(np.ceil(concentration_param * np.log(1 + num_obs / concentration_param)))

    if likelihood_model == 'dirichlet_multinomial':
        # TODO: move into separate function
        def model(obs):
            with numpyro.plate('beta_plate', truncation_num_clusters - 1):
                beta = numpyro.sample(
                    'beta',
                    numpyro.distributions.Beta(1, concentration_param))

            with numpyro.plate('topic_concentration_plate', truncation_num_clusters):
                topics_concentrations = numpyro.sample(
                    'topics_concentrations',
                    numpyro.distributions.Dirichlet(
                        concentration=jnp.full(obs_dim,
                                               fill_value=model_params['dirichlet_concentration_param'])))  # TODO: is mask necessary?

            with numpyro.plate('data', num_obs):
                z = numpyro.sample(
                    'z',
                    numpyro.distributions.Categorical(mix_weights(beta=beta)).mask(False))
                numpyro.sample(
                    'obs',
                    numpyro.distributions.DirichletMultinomial(
                        concentration=topics_concentrations[z]),
                    obs=obs)
    elif likelihood_model == 'multivariate_normal':
        # TODO: move into own function
        def model(obs):
            with numpyro.plate('beta_plate', truncation_num_clusters - 1):
                beta = numpyro.sample(
                    'beta',
                    numpyro.distributions.Beta(1, concentration_param))

            with numpyro.plate('mean_plate', truncation_num_clusters):
                mean = numpyro.sample(
                    'mean',
                    numpyro.distributions.MultivariateNormal(
                        jnp.zeros(obs_dim),
                        model_params['gaussian_mean_prior_cov_scaling'] * jnp.eye(obs_dim)))

            with numpyro.plate('data', num_obs):
                z = numpyro.sample(
                    'z',
                    numpyro.distributions.Categorical(mix_weights(beta=beta)).mask(False))
                numpyro.sample(
                    'obs',
                    numpyro.distributions.MultivariateNormal(
                        mean[z],
                        model_params['gaussian_cov_scaling'] * jnp.eye(obs_dim)),
                    obs=obs)
    else:
        raise ValueError(f'Likelihood model ({likelihood_model} not yet implemented)')

    hmc_kernel = numpyro.infer.NUTS(model)
    kernel = numpyro.infer.DiscreteHMCGibbs(inner_kernel=hmc_kernel)
    mcmc = numpyro.infer.MCMC(kernel, num_warmup=100, num_samples=num_samples, progress_bar=True)
    mcmc.run(jax.random.PRNGKey(0), obs=observations)
    # mcmc.print_summary()
    samples = mcmc.get_samples()

    if likelihood_model == 'dirichlet_multinomial':
        parameters = dict(
            beta=np.mean(np.array(samples['beta'][-1000:, :]), axis=0),
            topics_concentrations=np.mean(np.array(samples['topics_concentrations'][-1000:, :, :]), axis=0))
    elif likelihood_model == 'multivariate_normal':
        # shape (num samples, num centroids, obs dim)
        parameters = dict(
            beta=np.mean(np.array(samples['beta'][-1000:, :]), axis=0),
            means=np.mean(np.array(samples['mean'][-1000:, :, :]), axis=0))
    else:
        raise ValueError

    # shape (num samples, num obs)
    sampled_table_assignments = np.array(samples['z'])
    # convert sampled cluster assignments from (num samples, num obs) to (num obs, num clusters)
    bins = np.arange(0, 2 + np.max(sampled_table_assignments))
    table_assignment_posteriors = np.stack([
        np.histogram(sampled_table_assignments[-1000:, obs_idx], bins=bins, density=True)[0]
        for obs_idx in range(num_obs)])
    table_assignment_posteriors_running_sum = np.cumsum(table_assignment_posteriors,
                                                        axis=0)

    # returns classes assigned and centroids of corresponding classes
    hmc_gibbs_results = dict(
        table_assignment_posteriors=table_assignment_posteriors,
        table_assignment_posteriors_running_sum=table_assignment_posteriors_running_sum,
        parameters=parameters,
    )

    return hmc_gibbs_results


def sequential_updating_and_greedy_search(observations,
                                          concentration_param: float,
                                          likelihood_model: str,
                                          learning_rate):
    # Fast Bayesian Inference in Dirichlet Process Mixture Models
    # Wang and Dunson (2011)
    # The authors use the acronym SUGS, but others call it the "local MAP approximation"

    assert concentration_param > 0
    assert likelihood_model in {'multivariate_normal', 'dirichlet_multinomial',
                                'bernoulli', 'continuous_bernoulli'}
    num_obs, obs_dim = observations.shape

    # The recursion does not require recording the full history of priors/posteriors
    # but we record the full history for subsequent analysis
    max_num_latents = num_obs
    table_assignment_priors = torch.zeros(
        (num_obs, max_num_latents),
        dtype=torch.float64,
        requires_grad=False)
    table_assignment_priors[0, 0] = 1.

    table_assignment_posteriors = torch.zeros(
        (num_obs, max_num_latents),
        dtype=torch.float64,
        requires_grad=False)

    table_assignment_posteriors_running_sum = torch.zeros(
        (num_obs, max_num_latents),
        dtype=torch.float64,
        requires_grad=False)

    num_table_posteriors = torch.zeros(
        (num_obs, max_num_latents),
        dtype=torch.float64,
        requires_grad=False)

    if likelihood_model == 'multivariate_normal':
        cluster_parameters = dict(
            means=torch.full(
                size=(max_num_latents, obs_dim),
                fill_value=np.nan,
                dtype=torch.float64,
                requires_grad=True),
            stddevs=torch.full(
                size=(max_num_latents, obs_dim, obs_dim),
                fill_value=np.nan,
                dtype=torch.float64,
                requires_grad=True),
        )
        create_new_cluster_params_fn = create_new_cluster_params_multivariate_normal
        likelihood_fn = likelihood_multivariate_normal
    elif likelihood_model == 'dirichlet_multinomial':
        cluster_parameters = dict(
            topics_concentrations=torch.full(
                size=(max_num_latents, obs_dim),
                fill_value=np.nan,
                dtype=torch.float64,
                requires_grad=True),
        )

        create_new_cluster_params_fn = create_new_cluster_params_dirichlet_multinomial
        likelihood_fn = likelihood_dirichlet_multinomial
    elif likelihood_model == 'continuous_bernoulli':
        # need to use logits, otherwise gradient descent will carry parameters outside
        # valid interval
        cluster_parameters = dict(
            logits=torch.full(
                size=(max_num_latents, obs_dim),
                fill_value=np.nan,
                dtype=torch.float64,
                requires_grad=True)
        )
        create_new_cluster_params_fn = create_new_cluster_params_continuous_bernoulli
        likelihood_fn = likelihood_continuous_bernoulli

        # make sure no observation is 0 or 1 by adding epsilon
        epsilon = 1e-2
        observations[observations == 1.] -= epsilon
        observations[observations == 0.] += epsilon
    else:
        raise NotImplementedError

    # optimizer = torch.optim.SGD(params=[logits], lr=1.)
    optimizer = torch.optim.SGD(params=cluster_parameters.values(), lr=1.)

    # needed later for error checking
    one_tensor = torch.Tensor([1.]).double()

    torch_observations = torch.from_numpy(observations)
    for obs_idx, torch_observation in enumerate(torch_observations):

        # create new params for possible cluster, centered at that point
        create_new_cluster_params_fn(
            torch_observation=torch_observation,
            obs_idx=obs_idx,
            cluster_parameters=cluster_parameters)

        if obs_idx == 0:
            # first customer has to go at first table
            table_assignment_priors[obs_idx, 0] = 1.
            table_assignment_posteriors[obs_idx, 0] = 1.
            num_table_posteriors[obs_idx, 0] = 1.

            # update running sum of posteriors
            table_assignment_posteriors_running_sum[obs_idx, :] = torch.add(
                table_assignment_posteriors_running_sum[obs_idx - 1, :],
                table_assignment_posteriors[obs_idx, :])
            assert torch.allclose(torch.sum(table_assignment_posteriors_running_sum[obs_idx, :]),
                                  torch.Tensor([obs_idx + 1]).double())
        else:
            # construct prior
            table_assignment_prior = torch.clone(
                table_assignment_posteriors_running_sum[obs_idx - 1, :obs_idx + 1])
            # we don't subtract 1 because Python uses 0-based indexing
            assert torch.allclose(torch.sum(table_assignment_prior), torch.Tensor([obs_idx]).double())
            # add new table probability
            table_assignment_prior[1:] += concentration_param * torch.clone(
                num_table_posteriors[obs_idx - 1, :obs_idx])
            # renormalize
            table_assignment_prior /= (concentration_param + obs_idx)
            assert torch.allclose(torch.sum(table_assignment_prior), one_tensor)

            # record latent prior
            table_assignment_priors[obs_idx, :len(table_assignment_prior)] = table_assignment_prior

            optimizer.zero_grad()

            # infer posteriors using parameters
            likelihoods_per_latent, log_likelihoods_per_latent = likelihood_fn(
                torch_observation=torch_observation,
                obs_idx=obs_idx,
                cluster_parameters=cluster_parameters)
            assert torch.all(~torch.isnan(likelihoods_per_latent[:obs_idx + 1]))
            assert torch.all(~torch.isnan(log_likelihoods_per_latent[:obs_idx + 1]))

            unnormalized_table_assignment_posterior = torch.multiply(
                likelihoods_per_latent.detach(),
                table_assignment_prior)
            table_assignment_posterior = unnormalized_table_assignment_posterior / torch.sum(
                unnormalized_table_assignment_posterior)

            # apply local MAP approximation
            max_idx = torch.argmax(table_assignment_posterior)
            table_assignment_posterior = torch.nn.functional.one_hot(
                max_idx,
                num_classes=table_assignment_posterior.shape[0]).double()
            assert torch.allclose(torch.sum(table_assignment_posterior), one_tensor)

            # record latent posterior
            table_assignment_posteriors[obs_idx, :obs_idx + 1] = table_assignment_posterior

            # update running sum of posteriors
            table_assignment_posteriors_running_sum[obs_idx, :] = torch.add(
                table_assignment_posteriors_running_sum[obs_idx - 1, :],
                table_assignment_posteriors[obs_idx, :])
            assert torch.allclose(torch.sum(table_assignment_posteriors_running_sum[obs_idx, :]),
                                  torch.Tensor([obs_idx + 1]).double())

            # next, update parameters
            # Note: log likelihood is all we need for optimization because
            # log p(x, z; params) = log p(x|z; params) + log p(z)
            # and the second is constant w.r.t. params gradient
            loss = torch.mean(log_likelihoods_per_latent)
            loss.backward()

            # instead of typical dynamics:
            #       p_k <- p_k + (obs - p_k) / number of obs assigned to kth cluster
            # we use the new dynamics
            #       p_k <- p_k + posterior(obs belongs to kth cluster) * (obs - p_k) / total mass on kth cluster
            # that effectively means the learning rate should be this scaled_prefactor
            scaled_learning_rate = learning_rate * torch.divide(
                table_assignment_posteriors[obs_idx, :],
                table_assignment_posteriors_running_sum[obs_idx, :])
            scaled_learning_rate[torch.isnan(scaled_learning_rate)] = 0.
            scaled_learning_rate[torch.isinf(scaled_learning_rate)] = 0.

            # don't update the newest cluster
            scaled_learning_rate[obs_idx] = 0.

            for param_descr, param_tensor in cluster_parameters.items():
                # the scaled learning rate has shape (num latents,) aka (num obs,)
                # we need to create extra dimensions of size 1 for broadcasting to work
                # because param_tensor can have variable number of dimensions e.g. (num obs, obs dim)
                # for mean vs (num obs, obs dim, obs dim) for covariance, we need to dynamically
                # add the corect number of dimensions
                reshaped_scaled_learning_rate = scaled_learning_rate.view(
                    [scaled_learning_rate.shape[0]] + [1 for _ in range(len(param_tensor.shape[1:]))])
                if param_tensor.grad is None:
                    continue
                else:
                    scaled_param_tensor_grad = torch.multiply(
                        reshaped_scaled_learning_rate,
                        param_tensor.grad)
                    param_tensor.data += scaled_param_tensor_grad
                    assert_torch_no_nan_no_inf(param_tensor.data[:obs_idx + 1])

            # update posterior over number of tables using posterior over customer seat
            num_table_posteriors[obs_idx, torch.sum(table_assignment_posteriors_running_sum[obs_idx] != 0) - 1] = 1.

    local_map_results = dict(
        table_assignment_priors=table_assignment_priors.numpy(),
        table_assignment_posteriors=table_assignment_posteriors.numpy(),
        table_assignment_posteriors_running_sum=table_assignment_posteriors_running_sum.numpy(),
        num_table_posteriors=num_table_posteriors.numpy(),
        parameters={k: v.detach().numpy() for k, v in cluster_parameters.items()},
        # TODO: why do I have to detach here?
    )

    return local_map_results


def stochastic_variational_inference(observations,
                                     concentration_param: float,
                                     likelihood_model: str,
                                     learning_rate: float,
                                     num_steps: int,
                                     model_params: dict,
                                     truncation_num_clusters=None, ):
    num_obs, obs_dim = observations.shape
    if truncation_num_clusters is None:
        # multiply by 2 to be safe
        truncation_num_clusters = 2 * int(np.ceil(concentration_param * np.log(1 + num_obs / concentration_param)))
        truncation_num_clusters += 1

    if likelihood_model == 'dirichlet_multinomial':
        # TODO: move into separate function
        def model(obs):
            with numpyro.plate('beta_plate', truncation_num_clusters - 1):
                beta = numpyro.sample(
                    'beta',
                    numpyro.distributions.Beta(1, concentration_param))

            with numpyro.plate('topic_concentrations_plate', truncation_num_clusters):
                topics_concentrations = numpyro.sample(
                    'topics_concentrations',
                    numpyro.distributions.Dirichlet(
                        concentration=jnp.full(shape=obs_dim,
                                               fill_value=model_params['dirichlet_concentration_param'])))

            with numpyro.plate('data', num_obs):
                z = numpyro.sample(
                    'z',
                    numpyro.distributions.Categorical(mix_weights(beta=beta)).mask(False))  # TODO: is mask necessary?
                numpyro.sample(
                    'obs',
                    numpyro.distributions.DirichletMultinomial(
                        concentration=topics_concentrations[z]),
                    obs=obs)

        def guide(obs):
            q_beta_params = numpyro.param(
                'q_beta_params',
                init_value=jax.random.uniform(key=jax.random.PRNGKey(0),
                                              minval=0,
                                              maxval=2,
                                              shape=(truncation_num_clusters - 1,)),
                constraint=numpyro.distributions.constraints.positive)

            with numpyro.plate('beta_plate', truncation_num_clusters - 1):
                q_beta = numpyro.sample(
                    'beta',
                    numpyro.distributions.Beta(
                        concentration0=jnp.ones(truncation_num_clusters - 1),
                        concentration1=q_beta_params))

            # TODO: why is shape = (truncation, obs dim)?
            q_topics_concentrations_params = numpyro.param(
                f'q_topics_concentrations_params',
                init_value=jax.random.exponential(key=jax.random.PRNGKey(0),
                                                  shape=(truncation_num_clusters, obs_dim)),
                constraint=numpyro.distributions.constraints.positive)

            with numpyro.plate('topic_concentrations_plate', truncation_num_clusters):
                q_topics_concentrations = numpyro.sample(
                    'topics_concentrations',
                    numpyro.distributions.Dirichlet(
                        concentration=q_topics_concentrations_params))

            q_z_assignment_params = numpyro.param(
                'q_z_assignment_params',
                init_value=jax.random.dirichlet(key=jax.random.PRNGKey(0),
                                                alpha=jnp.ones(
                                                    truncation_num_clusters) / truncation_num_clusters,
                                                shape=(num_obs,)),
                constraint=numpyro.distributions.constraints.simplex)

            with numpyro.plate('data', num_obs):
                q_z = numpyro.sample(
                    'z',
                    numpyro.distributions.Categorical(
                        probs=q_z_assignment_params))

    elif likelihood_model == 'multivariate_normal':
        # TODO: move into own function
        def model(obs):
            with numpyro.plate('beta_plate', truncation_num_clusters - 1):
                beta = numpyro.sample(
                    'beta',
                    numpyro.distributions.Beta(1, concentration_param))

            with numpyro.plate('mean_plate', truncation_num_clusters):
                mean = numpyro.sample(
                    'mean',
                    numpyro.distributions.MultivariateNormal(
                        jnp.zeros(obs_dim),
                        model_params['gaussian_mean_prior_cov_scaling'] * jnp.eye(obs_dim)))

            with numpyro.plate('data', num_obs):
                z = numpyro.sample(
                    'z',
                    numpyro.distributions.Categorical(mix_weights(beta=beta)))
                numpyro.sample(
                    'obs',
                    numpyro.distributions.MultivariateNormal(
                        mean[z],
                        model_params['gaussian_cov_scaling'] * jnp.eye(obs_dim)),
                    obs=obs)

        def guide(obs):
            q_beta_params = numpyro.param(
                'q_beta_params',
                init_value=jax.random.uniform(
                    key=jax.random.PRNGKey(0),
                    minval=0,
                    maxval=2,
                    shape=(truncation_num_clusters - 1,)),
                constraint=numpyro.distributions.constraints.positive)

            with numpyro.plate('beta_plate', truncation_num_clusters - 1):
                q_beta = numpyro.sample(
                    'beta',
                    numpyro.distributions.Beta(
                        concentration0=jnp.ones(truncation_num_clusters - 1),
                        concentration1=q_beta_params))

            q_means_params = numpyro.param(
                'q_means_params',
                init_value=jax.random.multivariate_normal(
                    key=jax.random.PRNGKey(0),
                    mean=jnp.zeros(obs_dim),
                    cov=model_params['gaussian_mean_prior_cov_scaling'] * jnp.eye(obs_dim),
                    shape=(truncation_num_clusters, )))

            with numpyro.plate('mean_plate', truncation_num_clusters):
                q_mean = numpyro.sample(
                    'mean',
                    numpyro.distributions.MultivariateNormal(
                        q_means_params,
                        model_params['gaussian_cov_scaling'] * jnp.eye(obs_dim)))

            q_z_assignment_params = numpyro.param(
                'q_z_assignment_params',
                init_value=jax.random.dirichlet(key=jax.random.PRNGKey(0),
                                                alpha=jnp.ones(
                                                    truncation_num_clusters) / truncation_num_clusters,
                                                shape=(num_obs,)),
                constraint=numpyro.distributions.constraints.simplex)

            with numpyro.plate('data', num_obs):
                q_z = numpyro.sample(
                    'z',
                    numpyro.distributions.Categorical(probs=q_z_assignment_params))
    else:
        raise ValueError(f'Likelihood model ({likelihood_model} not yet implemented)')

    optimizer = numpyro.optim.Adam(step_size=learning_rate)
    svi = numpyro.infer.SVI(model,
                            guide,
                            optimizer,
                            loss=numpyro.infer.Trace_ELBO())
    svi_result = svi.run(jax.random.PRNGKey(0),
                         num_steps=num_steps,
                         obs=observations,
                         progress_bar=True)

    if likelihood_model == 'dirichlet_multinomial':
        parameters = dict(
            q_beta_params=np.array(svi_result.params['q_beta_params']),
            topics_concentrations=np.array(svi_result.params['q_topics_concentrations_params']))
    elif likelihood_model == 'multivariate_normal':
        parameters = dict(
            q_beta_params=np.array(svi_result.params['q_beta_params']),
            means=np.array(svi_result.params['q_means_params']))
    else:
        raise ValueError

    table_assignment_posteriors = np.array(svi_result.params['q_z_assignment_params'])
    table_assignment_posteriors_running_sum = np.cumsum(table_assignment_posteriors,
                                                        axis=0)

    # import seaborn as sns
    # fig, axes = plt.subplots(nrows=1, ncols=2)
    # sns.heatmap(docs, cmap='jet', ax=axes[0])
    # sns.heatmap(table_assignment_posteriors, cmap='jet', ax=axes[1])
    # fig.suptitle(f'Num Steps = {num_steps}')
    # plt.savefig(f'exp_02_mixture_of_unigrams/plots/hmc_gibbs_alpha={alpha}_num_steps={num_steps}.png')
    # plt.show()
    # plt.close()

    # returns classes assigned and centroids of corresponding classes
    stochastic_variational_inference_results = dict(
        table_assignment_posteriors=table_assignment_posteriors,
        table_assignment_posteriors_running_sum=table_assignment_posteriors_running_sum,
        parameters=parameters,
    )

    return stochastic_variational_inference_results


def variational_sequential_updating_and_greedy_search(observations,
                                                      concentration_param: float,
                                                      likelihood_model: str,
                                                      learning_rate):
    # A sequential algorithm for fast fitting of Dirichlet process mixture models
    # Nott, Zhang, Yau and Jasra (2013)
    raise NotImplementedError


def variational_bayes(observations,
                      likelihood_model: str,
                      learning_rate: float,
                      concentration_param: float,
                      max_iter: int = 8,  # same as DP-Means
                      num_initializations: int = 1):
    # Variational Inference for Dirichlet Process Mixtures
    # Blei and Jordan (2006)
    # likelihood_model not used
    # learning rate not used

    assert concentration_param > 0

    num_obs, obs_dim = observations.shape
    var_dp_gmm = sklearn.mixture.BayesianGaussianMixture(
        n_components=num_obs,
        max_iter=max_iter,
        n_init=num_initializations,
        init_params='random',
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=concentration_param)
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

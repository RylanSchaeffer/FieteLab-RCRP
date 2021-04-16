import numpy as np
import scipy
import scipy.special
import torch
import torch.distributions
import torch.nn
import torch.nn.functional
import torch.optim

from utils.helpers import assert_torch_no_nan_no_inf, torch_logits_to_probs, torch_probs_to_logits


def bayesian_recursion(observations,
                       concentration_param: float,
                       likelihood_model: str,
                       em_learning_rate=1e2,
                       num_em_steps=3):
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
                    likelihoods_per_latent.type(torch.float64),
                    table_assignment_prior)
                table_assignment_posterior = unnormalized_table_assignment_posterior / torch.sum(
                    unnormalized_table_assignment_posterior)
                assert torch.allclose(torch.sum(table_assignment_posterior), one_tensor)

                # record latent posterior
                table_assignment_posteriors[obs_idx, :len(table_assignment_posterior)] = table_assignment_posterior

                # print(f'Obs Index: {obs_idx}, EM Index: {em_idx}')
                # print('Table Assignment Prior: ', table_assignment_prior.detach().numpy())
                # print('Table Assignment Likelihood: ', likelihoods_per_latent.detach().numpy())
                # print('Table Assignment Posterior: ', table_assignment_posterior.detach().numpy())

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
                scaled_learning_rate = em_learning_rate * torch.divide(
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

            # update posterior over number of tables using posterior over customer seat
            for k1, p_z_t_equals_k1 in enumerate(table_assignment_posteriors[obs_idx, :obs_idx + 1]):
                for k2, p_prev_num_tables_equals_k2 in enumerate(num_table_posteriors[obs_idx - 1, :obs_idx + 1]):
                    # advance number of tables by 1 if customer seating > number of current tables
                    if k1 > k2 + 1:
                        num_table_posteriors.data[obs_idx, k2 + 1] += p_z_t_equals_k1 * p_prev_num_tables_equals_k2
                    # customer allocated to previous table
                    elif k1 <= k2:
                        num_table_posteriors.data[obs_idx, k2] += p_z_t_equals_k1 * p_prev_num_tables_equals_k2
                    # create new table
                    elif k1 == k2 + 1:
                        num_table_posteriors.data[obs_idx, k1] += p_z_t_equals_k1 * p_prev_num_tables_equals_k2
                    else:
                        raise ValueError
            assert torch.allclose(torch.sum(num_table_posteriors[obs_idx, :]), one_tensor)

    bayesian_recursion_results = dict(
        table_assignment_priors=table_assignment_priors.detach().numpy(),
        table_assignment_posteriors=table_assignment_posteriors.detach().numpy(),
        table_assignment_posteriors_running_sum=table_assignment_posteriors_running_sum.detach().numpy(),
        num_table_posteriors=num_table_posteriors.detach().numpy(),
        parameters={k: v.detach().numpy() for k, v in cluster_parameters.items()},
        # TODO: why do I have to detach here?
    )

    return bayesian_recursion_results


def local_map(observations,
              concentration_param: float,
              likelihood_model: str,
              em_learning_rate=1e2):

    # TODO: merge with Bayesian recursion

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

            # E step: infer posteriors using parameters
            likelihoods_per_latent, log_likelihoods_per_latent = likelihood_fn(
                torch_observation=torch_observation,
                obs_idx=obs_idx,
                cluster_parameters=cluster_parameters)
            assert torch.all(~torch.isnan(likelihoods_per_latent[:obs_idx + 1]))
            assert torch.all(~torch.isnan(log_likelihoods_per_latent[:obs_idx + 1]))

            unnormalized_table_assignment_posterior = torch.multiply(
                likelihoods_per_latent.type(torch.float64),
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
            scaled_learning_rate = em_learning_rate * torch.divide(
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
            num_table_posteriors[obs_idx, torch.sum(table_assignment_posteriors_running_sum[obs_idx] != 0)] = 1.

    local_map_results = dict(
        table_assignment_priors=table_assignment_priors.detach().numpy(),
        table_assignment_posteriors=table_assignment_posteriors.detach().numpy(),
        table_assignment_posteriors_running_sum=table_assignment_posteriors_running_sum.detach().numpy(),
        num_table_posteriors=num_table_posteriors.detach().numpy(),
        parameters={k: v.detach().numpy() for k, v in cluster_parameters.items()},
        # TODO: why do I have to detach here?
    )

    return local_map_results


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


def beta(a, b):
    result = torch.exp(log_beta(a, b))
    return result

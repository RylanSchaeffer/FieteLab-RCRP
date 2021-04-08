import numpy as np
import torch
import torch.distributions
import torch.distributions.utils
import torch.nn
import torch.optim

from utils.helpers import assert_torch_no_nan_no_inf


def bayesian_recursion(observations,
                       alpha: float,
                       num_em_steps=1):
    assert alpha > 0
    num_obs, obs_dim = observations.shape

    # The inference algorithm does not require recording the full history of priors/posteriors
    # We record the full history for subsequent analysis
    max_num_latents = int(1.3 * num_obs)
    table_assignment_priors = torch.zeros(
        (num_obs, max_num_latents),
        dtype=torch.float32,
        requires_grad=False)
    table_assignment_priors[0, 0] = 1.

    table_assignment_posteriors = torch.zeros(
        (num_obs, max_num_latents),
        dtype=torch.float32,
        requires_grad=False)

    table_assignment_posteriors_running_sum = torch.zeros(
        (num_obs, max_num_latents),
        dtype=torch.float32,
        requires_grad=False)

    num_table_posteriors = torch.zeros(
        (num_obs, max_num_latents),
        dtype=torch.float32,
        requires_grad=False)

    # need to use logits, otherwise gradient descent will carry parameters outside
    # valid interval
    cluster_logits = torch.full(
        size=(max_num_latents, obs_dim),
        fill_value=np.nan,
        dtype=torch.float32,
        requires_grad=True)

    epsilon = 1e-2
    optimizer = torch.optim.SGD(params=[cluster_logits], lr=1.)

    # make sure no observation is 0 or 1 by adding epsilon
    observations[observations == 1.] -= epsilon
    observations[observations == 0.] += epsilon
    torch_observations = torch.from_numpy(observations)
    em_learning_rate = 1e0
    one_tensor = torch.Tensor([1.])

    for obs_idx, torch_observation in enumerate(torch_observations):

        # create new params for possible cluster, centered at that point
        # += is necessary for the operation to be performed in place
        # data is necessary to not break backprop
        # see https://stackoverflow.com/questions/53819383/how-to-assign-a-new-value-to-a-pytorch-variable-without-breaking-backpropagation
        cluster_logits.data[obs_idx, :] = torch.distributions.utils.probs_to_logits(
            torch_observation)

        for em_idx in range(num_em_steps):

            optimizer.zero_grad()

            # E step: infer posteriors using parameters

            # compute Continuous Bernoulli likelihoods
            cont_bern = torch.distributions.ContinuousBernoulli(
                logits=cluster_logits[:obs_idx+1])
            log_likelihoods_per_obs_dim = cont_bern.log_prob(value=torch_observation)

            likelihoods_per_obs_dim = torch.exp(log_likelihoods_per_obs_dim)

            # compute Continuous Bernoulli likelihoods
            # normalizing_const = torch.divide(
            #     2. * torch.arctanh(1. - 2. * cluster_logits[:obs_idx+1]),
            #     1. - 2. * cluster_logits[:obs_idx+1])
            # normalizing_const[torch.isnan(normalizing_const)] = 2.
            # likelihoods_per_obs_dim = torch.multiply(
            #     normalizing_const,
            #     torch.multiply(
            #         torch.pow(cluster_logits[:obs_idx + 1], torch_observation),
            #         torch.pow(1. - cluster_logits[:obs_idx + 1], 1. - torch_observation)))
            assert torch.all(~torch.isnan(likelihoods_per_obs_dim[:obs_idx + 1]))
            likelihoods = torch.prod(likelihoods_per_obs_dim, dim=1)

            if obs_idx == 0:
                # first customer has to go at first table
                table_assignment_posterior = torch.Tensor([1.])
                table_assignment_posteriors[obs_idx, 0] = table_assignment_posterior
                num_table_posteriors[0, 0] = 1.
            else:
                table_assignment_prior = torch.clone(
                    table_assignment_posteriors_running_sum[obs_idx - 1, :obs_idx + 1])
                # we don't subtract 1 because Python uses 0-based indexing
                assert torch.allclose(torch.sum(table_assignment_prior), torch.Tensor([obs_idx]))
                # right shift by 1
                # table_assignment_prior[1:] += alpha * torch.clone(
                #     num_table_posteriors[obs_idx - 1, :len(likelihoods) - 1])
                table_assignment_prior[1:] += alpha * torch.clone(
                    num_table_posteriors[obs_idx - 1, :obs_idx])
                table_assignment_prior /= (alpha + obs_idx)
                assert torch.allclose(torch.sum(table_assignment_prior), one_tensor)

                # record latent prior
                table_assignment_priors[obs_idx, :len(table_assignment_prior)] = table_assignment_prior

                unnormalized_table_assignment_posterior = torch.multiply(
                    likelihoods.type(torch.float32),
                    table_assignment_prior)
                table_assignment_posterior = unnormalized_table_assignment_posterior / torch.sum(
                    unnormalized_table_assignment_posterior)
                assert torch.allclose(torch.sum(table_assignment_posterior), one_tensor)

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
                            num_table_posteriors.data[obs_idx, k2] += p_z_t_equals_k1 * p_prev_num_tables_equals_k2
                        # create new table
                        elif k1 == k2 + 1:
                            num_table_posteriors.data[obs_idx, k1] += p_z_t_equals_k1 * p_prev_num_tables_equals_k2
                        else:
                            raise ValueError
                num_table_posteriors[obs_idx, :] /= torch.sum(num_table_posteriors[obs_idx, :])
                assert torch.allclose(torch.sum(num_table_posteriors[obs_idx, :]), one_tensor)

            # update running sum of posteriors
            table_assignment_posteriors_running_sum[obs_idx, :] = torch.add(
                table_assignment_posteriors_running_sum[obs_idx - 1, :],
                table_assignment_posteriors[obs_idx, :])
            assert torch.allclose(torch.sum(table_assignment_posteriors_running_sum[obs_idx, :]),
                                  torch.Tensor([obs_idx + 1]))

            # M step: update parameters
            loss = torch.mean(likelihoods)
            loss.backward()

            # instead of typical dynamics:
            #       p_k <- p_k + (obs - p_k) / number of obs assigned to kth cluster
            # we use the new dynamics
            #       p_k <- p_k + posterior(obs belongs to kth cluster) * (obs - p_k) / total mass on kth cluster
            # that effectively means the learning rate should be this prefactor
            prefactor = torch.divide(table_assignment_posterior[obs_idx],
                                     table_assignment_posteriors_running_sum[obs_idx])
            prefactor[torch.isnan(prefactor)] = 0.
            prefactor[torch.isinf(prefactor)] = 0.

            # don't update the newest cluster
            prefactor[obs_idx] = 0.

            scaled_cluster_logits_grad = torch.multiply(
                prefactor[:, None],
                cluster_logits.grad)
            cluster_logits.data += em_learning_rate * scaled_cluster_logits_grad

            # assert torch.all(cluster_logits[:obs_idx+1] >= epsilon)
            # assert torch.all(cluster_logits[:obs_idx+1] <= 1. - epsilon)

    # TODO: why do I have to detach here?
    bayesian_recursion_results = dict(
        table_assignment_priors=table_assignment_priors.detach().numpy(),
        table_assignment_posteriors=table_assignment_posteriors.detach().numpy(),
        table_assignment_posteriors_running_sum=table_assignment_posteriors_running_sum.detach().numpy(),
        num_table_posteriors=num_table_posteriors.detach().numpy(),
        parameters=dict(
            cluster_logits=cluster_logits.detach().numpy(),
            cluster_probs=torch.distributions.utils.logits_to_probs(cluster_logits).detach().numpy()),
    )

    return bayesian_recursion_results

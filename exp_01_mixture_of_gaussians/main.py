import numpy as np
import os

from exp_01_mixture_of_gaussians.plot import plot_inference_results

from utils.data import sample_sequence_from_mixture_of_gaussians
from utils.helpers import assert_no_nan_no_inf
from utils.inference import bayesian_recursion, dp_means_online, dp_means_offline
from utils.metrics import normalized_mutual_information


def main():
    plot_dir = 'exp_01_mixture_of_gaussians/plots'
    os.makedirs(plot_dir, exist_ok=True)

    np.random.seed(1)

    # sample data
    alpha = 1.5
    sampled_mog_results = sample_sequence_from_mixture_of_gaussians(
        seq_len=100,
        class_sampling='CRP',
        alpha=alpha,
        gaussian_params=dict(gaussian_cov_scaling=0.3,
                             gaussian_mean_prior_cov_scaling=6.)
    )

    lambd = 1.4
    dp_means_online_results = dp_means_online(
        observations=sampled_mog_results['gaussian_samples_seq'],
        lambd=lambd)

    plot_inference_results(
        sampled_mog_results=sampled_mog_results,
        inference_results=dp_means_online_results,
        inference_alg=f'dp_means_online_lambda={lambd}',
        plot_dir=plot_dir)

    # normalized_mutual_information(assigned_table_seq_one_hot=sampled_mog_results['assigned_table_seq_one_hot'],
    #                               table_posteriors_one_hot=dp_means_online_results['table_assignment_posteriors'])

    def likelihood_fn(observation, parameters):
        # create new mean for new table, centered at that point
        parameters['means'] = np.vstack([parameters['means'], observation[np.newaxis, :]])
        obs_dim = parameters['means'].shape[1]
        parameters['covs'] = np.vstack([parameters['covs'], np.eye(obs_dim)[np.newaxis, :, :]])

        # calculate likelihood under each cluster mean
        covariance_determinants = np.linalg.det(parameters['covs'])
        normalizing_constant = np.sqrt(np.power(2 * np.pi, obs_dim) * covariance_determinants)
        # shape (num gaussians, dim of gaussians)
        diff = (observation - parameters['means'])
        quadratic = np.einsum(
            'bi, bij, bj->b',
            diff,
            np.linalg.inv(parameters['covs']),
            diff
        )
        likelihoods = np.exp(-0.5 * quadratic) / normalizing_constant
        assert np.all(~np.isnan(likelihoods))

        return likelihoods, parameters

    def update_parameters_fn(observation,
                             table_assignment_posteriors_running_sum,
                             table_assignment_posterior,
                             parameters):
        # the strategy here is to update parameters as a moving average, but instead of dividing
        # by the number of points assigned to each cluster, we divide by the total probability
        # mass assigned to each cluster

        # create a copy of observation for each possible cluster
        stacked_observation = np.repeat(observation[np.newaxis, :],
                                        repeats=len(table_assignment_posteriors_running_sum),
                                        axis=0)

        # compute online average of clusters' means
        # instead of typical dynamics:
        #       m_k <- m_k + (obs - m_k) / number of obs assigned to kth cluster
        # we use the new dynamics
        #       m_k <- m_k + posterior(obs belongs to kth cluster) * (obs - m_k) / total mass on kth cluster
        # floating point errors are common here!
        prefactor = np.divide(table_assignment_posterior,
                              table_assignment_posteriors_running_sum)
        prefactor[np.isnan(prefactor)] = 0.
        assert_no_nan_no_inf(prefactor)

        diff = stacked_observation - parameters['means']
        assert_no_nan_no_inf(diff)
        means_updates = np.multiply(
            prefactor[:, np.newaxis],
            diff)
        parameters['means'] += means_updates
        assert_no_nan_no_inf(parameters['means'])

        return parameters

    bayesian_recursion_results = bayesian_recursion(
        observations=sampled_mog_results['gaussian_samples_seq'],
        alpha=alpha,
        likelihood_fn=likelihood_fn,
        update_parameters_fn=update_parameters_fn)

    plot_inference_results(
        sampled_mog_results=sampled_mog_results,
        inference_results=bayesian_recursion_results,
        inference_alg=f'bayesian_recursion_alpha={alpha}',
        plot_dir=plot_dir)


if __name__ == '__main__':
    main()

import numpy as np
import os

from exp_01_mixture_of_gaussians.plot import *

from utils.data import sample_sequence_from_mixture_of_gaussians
from utils.helpers import assert_no_nan_no_inf
from utils.inference import bayesian_recursion, dp_means_online, dp_means_offline, gibbs_sampling
from utils.metrics import normalized_mutual_information


def main():
    plot_dir = 'exp_01_mixture_of_gaussians/plots'
    os.makedirs(plot_dir, exist_ok=True)

    np.random.seed(1)

    # sample data
    num_gaussians = 3
    gaussian_cov_scaling = 0.3
    gaussian_mean_prior_cov_scaling = 6.
    sampled_mog_results = sample_sequence_from_mixture_of_gaussians(
        seq_len=100,
        class_sampling='Uniform',
        alpha=None,
        num_gaussians=num_gaussians,
        gaussian_params=dict(gaussian_cov_scaling=gaussian_cov_scaling,
                             gaussian_mean_prior_cov_scaling=gaussian_mean_prior_cov_scaling)
    )

    gibbs_sampling_results = run_and_plot_gibbs_sampling(sampled_mog_results=sampled_mog_results,
                                                         plot_dir=plot_dir,
                                                         gaussian_cov_scaling=gaussian_cov_scaling,
                                                         gaussian_mean_prior_cov_scaling=gaussian_mean_prior_cov_scaling)

    # plot_sample_from_mixture_of_gaussians(assigned_table_seq=sampled_mog_results['assigned_table_seq'],
    #                                       gaussian_samples_seq=sampled_mog_results['gaussian_samples_seq'],
    #                                       plot_dir=plot_dir)

    dp_means_offline_results = run_and_plot_dp_means_offline(sampled_mog_results=sampled_mog_results,
                                                             plot_dir=plot_dir)

    dp_means_online_results = run_and_plot_dp_means_online(sampled_mog_results=sampled_mog_results,
                                                           plot_dir=plot_dir)

    bayesian_recursion_results = run_and_plot_bayesian_recursion(sampled_mog_results=sampled_mog_results,
                                                                 plot_dir=plot_dir)

    inference_algs_results = {
        'Bayesian Recursion': bayesian_recursion_results,
        'DP-Means (Online)': dp_means_online_results,
        'DP-Means (Offline)': dp_means_offline_results,
    }

    plot_num_clusters_by_param(
        inference_algs_results=inference_algs_results,
        plot_dir=plot_dir,
        num_clusters=num_gaussians)


def run_and_plot_gibbs_sampling(sampled_mog_results,
                                plot_dir,
                                gaussian_cov_scaling,
                                gaussian_mean_prior_cov_scaling):

    gibbs_sampling_plot_dir = os.path.join(plot_dir, 'gibbs_sampling')
    os.makedirs(gibbs_sampling_plot_dir, exist_ok=True)
    num_clusters_by_iteration = {}

    gibbs_sampling_results = gibbs_sampling(
        observations=sampled_mog_results['gaussian_samples_seq'],
        num_iterations=1000,
        gaussian_cov_scaling=gaussian_cov_scaling,
        gaussian_mean_prior_cov_scaling=gaussian_mean_prior_cov_scaling)

    num_clusters_by_lambda = np.sum(
        np.sum(gibbs_sampling_results['table_assignment_posteriors'], axis=0) != 0)

    plot_inference_results(
        sampled_mog_results=sampled_mog_results,
        inference_results=gibbs_sampling_results,
        inference_alg='dp_means_online_lambda={:.2f}'.format(lambd),
        plot_dir=gibbs_sampling_plot_dir)

    gibbs_sampling_results = dict(
        num_clusters_by_param=num_clusters_by_lambda
    )

    return gibbs_sampling_results

    # normalized_mutual_information(assigned_table_seq_one_hot=sampled_mog_results['assigned_table_seq_one_hot'],
    #                               table_posteriors_one_hot=gibbs_sampling_results['table_assignment_posteriors'])


def run_and_plot_bayesian_recursion(sampled_mog_results,
                                    plot_dir):
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

    alphas = 0.01 + np.arange(0., 2.01, 0.05)
    bayesian_recursion_plot_dir = os.path.join(plot_dir, 'bayesian_recursion')
    os.makedirs(bayesian_recursion_plot_dir, exist_ok=True)
    num_clusters_by_alpha = {}
    for alpha in alphas:
        bayesian_recursion_results = bayesian_recursion(
            observations=sampled_mog_results['gaussian_samples_seq'],
            alpha=alpha,
            likelihood_fn=likelihood_fn,
            update_parameters_fn=update_parameters_fn)

        # order the tables by mass
        total_mass = sampled_mog_results['gaussian_samples_seq'].shape[0]
        final_mass_at_tables = bayesian_recursion_results['table_assignment_posteriors_running_sum'][-1, :]
        sorted_fraction_final_mass_at_tables = np.sort(final_mass_at_tables / total_mass)[::-1]
        # find number of tables to reach 95% mass
        cumulative_final_mass_at_tables = np.cumsum(sorted_fraction_final_mass_at_tables)
        num_tables = 1 + np.argmax(cumulative_final_mass_at_tables > 0.95)
        num_clusters_by_alpha[alpha] = num_tables

        plot_inference_results(
            sampled_mog_results=sampled_mog_results,
            inference_results=bayesian_recursion_results,
            inference_alg='bayesian_recursion_alpha={:.2f}'.format(alpha),
            plot_dir=bayesian_recursion_plot_dir)

    bayesian_recursion_results = dict(
        num_clusters_by_param=num_clusters_by_alpha
    )

    return bayesian_recursion_results


def run_and_plot_dp_means_offline(sampled_mog_results,
                                  plot_dir):
    lambdas = 0.01 + np.arange(0., 2.01, 0.05)
    dp_means_plot_dir = os.path.join(plot_dir, 'dp_means_offline')
    os.makedirs(dp_means_plot_dir, exist_ok=True)
    num_clusters_by_lambda = {}
    for lambd in lambdas:
        dp_means_offline_results = dp_means_offline(
            observations=sampled_mog_results['gaussian_samples_seq'],
            num_passes=8,
            lambd=lambd)

        num_clusters_by_lambda[lambd] = np.sum(
            np.sum(dp_means_offline_results['table_assignment_posteriors'], axis=0) != 0)

        plot_inference_results(
            sampled_mog_results=sampled_mog_results,
            inference_results=dp_means_offline_results,
            inference_alg='dp_means_online_lambda={:.2f}'.format(lambd),
            plot_dir=dp_means_plot_dir)

    dp_means_offline_results = dict(
        num_clusters_by_param=num_clusters_by_lambda
    )

    return dp_means_offline_results


def run_and_plot_dp_means_online(sampled_mog_results,
                                 plot_dir):
    lambdas = 0.01 + np.arange(0., 2.01, 0.05)
    dp_means_plot_dir = os.path.join(plot_dir, 'dp_means_online')
    os.makedirs(dp_means_plot_dir, exist_ok=True)
    num_clusters_by_lambda = {}
    for lambd in lambdas:
        dp_means_online_results = dp_means_online(
            observations=sampled_mog_results['gaussian_samples_seq'],
            lambd=lambd)

        num_clusters_by_lambda[lambd] = np.sum(
            np.sum(dp_means_online_results['table_assignment_posteriors'], axis=0) != 0)

        plot_inference_results(
            sampled_mog_results=sampled_mog_results,
            inference_results=dp_means_online_results,
            inference_alg='dp_means_online_lambda={:.2f}'.format(lambd),
            plot_dir=dp_means_plot_dir)

    dp_means_online_results = dict(
        num_clusters_by_param=num_clusters_by_lambda
    )

    return dp_means_online_results

    # normalized_mutual_information(assigned_table_seq_one_hot=sampled_mog_results['assigned_table_seq_one_hot'],
    #                               table_posteriors_one_hot=dp_means_online_results['table_assignment_posteriors'])


if __name__ == '__main__':
    main()

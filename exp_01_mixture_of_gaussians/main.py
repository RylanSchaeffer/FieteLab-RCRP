import numpy as np
import os
import pandas as pd

from exp_01_mixture_of_gaussians.plot import *

from utils.data import sample_sequence_from_mixture_of_gaussians
from utils.helpers import assert_no_nan_no_inf
from utils.inference import bayesian_recursion, dp_means_online, dp_means_offline, nuts_sampling, variational_bayes
from utils.metrics import score_predicted_clusters


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
                             gaussian_mean_prior_cov_scaling=gaussian_mean_prior_cov_scaling))

    plot_sample_from_mixture_of_gaussians(assigned_table_seq=sampled_mog_results['assigned_table_seq'],
                                          gaussian_samples_seq=sampled_mog_results['gaussian_samples_seq'],
                                          plot_dir=plot_dir)

    # gibbs_sampling_results = run_and_plot_nuts_sampling(sampled_mog_results=sampled_mog_results,
    #                                                     plot_dir=plot_dir,
    #                                                     gaussian_cov_scaling=gaussian_cov_scaling,
    #                                                     gaussian_mean_prior_cov_scaling=gaussian_mean_prior_cov_scaling)

    bayesian_recursion_results = run_and_plot_bayesian_recursion(
        sampled_mog_results=sampled_mog_results,
        plot_dir=plot_dir)

    dp_means_offline_results = run_and_plot_dp_means_offline(
        sampled_mog_results=sampled_mog_results,
        plot_dir=plot_dir)

    dp_means_online_results = run_and_plot_dp_means_online(
        sampled_mog_results=sampled_mog_results,
        plot_dir=plot_dir)

    variational_bayes_results = run_and_plot_variational_bayes(
        sampled_mog_results=sampled_mog_results,
        plot_dir=plot_dir,
        gaussian_mean_prior_cov_scaling=gaussian_mean_prior_cov_scaling)

    inference_algs_results = {
        'Bayesian Recursion': bayesian_recursion_results,
        # 'NUTS Sampling': nuts_sampling_results,
        'DP-Means (Online)': dp_means_online_results,
        'DP-Means (Offline)': dp_means_offline_results,
        'Variational Bayes': variational_bayes_results,
    }

    plot_inference_algs_comparison(
        inference_algs_results=inference_algs_results,
        plot_dir=plot_dir,
        sampled_mog_results=sampled_mog_results)


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

    alphas = 0.01 + np.arange(0., 3.01, 0.25)
    bayesian_recursion_plot_dir = os.path.join(plot_dir, 'bayesian_recursion')
    os.makedirs(bayesian_recursion_plot_dir, exist_ok=True)
    num_clusters_by_alpha = {}
    scores_by_alpha = {}
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

        # record scores
        scores = score_predicted_clusters(
            true_cluster_labels=sampled_mog_results['assigned_table_seq'],
            table_assignment_posteriors=bayesian_recursion_results['table_assignment_posteriors'])
        scores_by_alpha[alpha] = scores

        plot_inference_results(
            sampled_mog_results=sampled_mog_results,
            inference_results=bayesian_recursion_results,
            inference_alg='bayesian_recursion_alpha={:.2f}'.format(alpha),
            plot_dir=bayesian_recursion_plot_dir)

        print('Finished Bayesian recursion alpha={:.2f}'.format(alpha))

    bayesian_recursion_results = dict(
        num_clusters_by_param=num_clusters_by_alpha,
        scores_by_param=pd.DataFrame(scores_by_alpha).T,
    )

    return bayesian_recursion_results


def run_and_plot_dp_means_offline(sampled_mog_results,
                                  plot_dir):
    lambdas = 0.01 + np.arange(0., 3.01, 0.25)
    dp_means_plot_dir = os.path.join(plot_dir, 'dp_means_offline')
    os.makedirs(dp_means_plot_dir, exist_ok=True)
    num_clusters_by_lambda = {}
    scores_by_lambda = {}
    for lambd in lambdas:
        dp_means_offline_results = dp_means_offline(
            observations=sampled_mog_results['gaussian_samples_seq'],
            num_passes=8,
            lambd=lambd)

        # count number of clusters
        num_clusters_by_lambda[lambd] = np.sum(
            np.sum(dp_means_offline_results['table_assignment_posteriors'], axis=0) != 0)

        # score clusters
        scores = score_predicted_clusters(
            true_cluster_labels=sampled_mog_results['assigned_table_seq'],
            table_assignment_posteriors=dp_means_offline_results['table_assignment_posteriors'])
        scores_by_lambda[lambd] = scores

        plot_inference_results(
            sampled_mog_results=sampled_mog_results,
            inference_results=dp_means_offline_results,
            inference_alg='dp_means_online_lambda={:.2f}'.format(lambd),
            plot_dir=dp_means_plot_dir)

        print('Finished DP-Means Offline lambda={:.2f}'.format(lambd))

    dp_means_offline_results = dict(
        num_clusters_by_param=num_clusters_by_lambda,
        scores_by_param=pd.DataFrame(scores_by_lambda).T,
    )
    return dp_means_offline_results


def run_and_plot_dp_means_online(sampled_mog_results,
                                 plot_dir):
    lambdas = 0.01 + np.arange(0., 3.01, 0.25)
    dp_means_plot_dir = os.path.join(plot_dir, 'dp_means_online')
    os.makedirs(dp_means_plot_dir, exist_ok=True)
    num_clusters_by_lambda = {}
    scores_by_lambda = {}
    for lambd in lambdas:
        dp_means_online_results = dp_means_online(
            observations=sampled_mog_results['gaussian_samples_seq'],
            lambd=lambd)

        # score clusters
        scores = score_predicted_clusters(
            true_cluster_labels=sampled_mog_results['assigned_table_seq'],
            table_assignment_posteriors=dp_means_online_results['table_assignment_posteriors'])
        scores_by_lambda[lambd] = scores

        num_clusters_by_lambda[lambd] = np.sum(
            np.sum(dp_means_online_results['table_assignment_posteriors'], axis=0) != 0)

        plot_inference_results(
            sampled_mog_results=sampled_mog_results,
            inference_results=dp_means_online_results,
            inference_alg='dp_means_online_lambda={:.2f}'.format(lambd),
            plot_dir=dp_means_plot_dir)

        print('Finished DP-Means Online lambda={:.2f}'.format(lambd))

    dp_means_online_results = dict(
        num_clusters_by_param=num_clusters_by_lambda,
        scores_by_param=pd.DataFrame(scores_by_lambda).T,
    )

    return dp_means_online_results


def run_and_plot_nuts_sampling(sampled_mog_results,
                               plot_dir,
                               gaussian_cov_scaling,
                               gaussian_mean_prior_cov_scaling):
    nuts_sampling_plot_dir = os.path.join(plot_dir, 'gibbs_sampling')
    os.makedirs(nuts_sampling_plot_dir, exist_ok=True)
    num_clusters_by_num_samples = {}

    alpha = 1.5
    possible_num_samples = np.arange(1000, 5001, 1000)
    for num_samples in possible_num_samples:
        nuts_sampling_results = nuts_sampling(
            observations=sampled_mog_results['gaussian_samples_seq'],
            num_samples=1000,
            alpha=alpha,
            gaussian_cov_scaling=gaussian_cov_scaling,
            gaussian_mean_prior_cov_scaling=gaussian_mean_prior_cov_scaling)

        num_clusters_by_num_samples[num_samples] = 10

        plot_inference_results(
            sampled_mog_results=sampled_mog_results,
            inference_results=nuts_sampling_results,
            inference_alg='gibbs_sampling={}'.format(num_samples),
            plot_dir=nuts_sampling_plot_dir)

    nuts_sampling_results = dict(
        num_clusters_by_param=num_clusters_by_num_samples
    )

    return nuts_sampling_results


def run_and_plot_variational_bayes(sampled_mog_results,
                                   gaussian_mean_prior_cov_scaling,
                                   plot_dir, ):
    alphas = 0.01 + np.arange(0., 3.01, 0.25)
    variational_plot_dir = os.path.join(plot_dir, 'variational')
    os.makedirs(variational_plot_dir, exist_ok=True)
    num_clusters_by_alpha = {}
    scores_by_alpha = {}
    for alpha in alphas:
        variational_bayes_results = variational_bayes(
            observations=sampled_mog_results['gaussian_samples_seq'],
            alpha=alpha,
            gaussian_mean_prior_cov_scaling=gaussian_mean_prior_cov_scaling)

        # order the tables by mass
        total_mass = sampled_mog_results['gaussian_samples_seq'].shape[0]
        final_mass_at_tables = variational_bayes_results['table_assignment_posteriors_running_sum'][-1, :]
        sorted_fraction_final_mass_at_tables = np.sort(final_mass_at_tables / total_mass)[::-1]
        # find number of tables to reach 95% mass
        cumulative_final_mass_at_tables = np.cumsum(sorted_fraction_final_mass_at_tables)
        num_tables = 1 + np.argmax(cumulative_final_mass_at_tables > 0.95)
        num_clusters_by_alpha[alpha] = num_tables

        # score clusters
        scores = score_predicted_clusters(
            true_cluster_labels=sampled_mog_results['assigned_table_seq'],
            table_assignment_posteriors=variational_bayes_results['table_assignment_posteriors'])
        scores_by_alpha[alpha] = scores

        plot_inference_results(
            sampled_mog_results=sampled_mog_results,
            inference_results=variational_bayes_results,
            inference_alg='variational_bayes={:.2f}'.format(alpha),
            plot_dir=variational_plot_dir)

        print('Finished Variational Bayes alpha={:.2f}'.format(alpha))

    variational_bayes_results = dict(
        num_clusters_by_param=num_clusters_by_alpha,
        scores_by_param=pd.DataFrame(scores_by_alpha).T,
    )

    return variational_bayes_results


if __name__ == '__main__':
    main()

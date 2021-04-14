import joblib
import numpy as np
import pandas as pd

from exp_01_mixture_of_gaussians.plot import *

from utils.data import sample_sequence_from_mixture_of_gaussians
from utils.helpers import assert_numpy_no_nan_no_inf
from utils.inference import bayesian_recursion
from utils.inference_mix_of_gauss import dp_means_online, dp_means_offline, \
    sampling_hmc_gibbs, variational_bayes
from utils.metrics import score_predicted_clusters


def main():
    plot_dir = 'exp_01_mixture_of_gaussians/plots'
    os.makedirs(plot_dir, exist_ok=True)
    np.random.seed(1)

    num_datasets = 5
    inference_algs_results_by_dataset = {}
    sampled_mog_results_by_dataset = {}
    # generate lots of datasets and record performance
    for dataset_idx in range(num_datasets):
        dataset_dir = os.path.join(plot_dir, f'dataset={dataset_idx}')
        dataset_results_path = os.path.join(dataset_dir, 'dataset_results.joblib')

        if os.path.isfile(dataset_results_path):
            # load from disk if exists
            dataset_results = joblib.load(dataset_results_path)
        else:
            # otherwise, generate anew
            dataset_inference_algs_results, dataset_sampled_mog_results = \
                run_one_dataset(plot_dir=dataset_dir)
            dataset_results = dict(
                dataset_inference_algs_results=dataset_inference_algs_results,
                dataset_sampled_mog_results=dataset_sampled_mog_results,
            )
            joblib.dump(dataset_results, dataset_results_path)

            # delete variables from memory and perform fresh read from disk
            del dataset_inference_algs_results, dataset_sampled_mog_results
            del dataset_results
            dataset_results = joblib.load(dataset_results_path)

        inference_algs_results_by_dataset[dataset_idx] = dataset_results['dataset_inference_algs_results']
        sampled_mog_results_by_dataset[dataset_idx] = dataset_results['dataset_sampled_mog_results']

    plot_inference_algs_comparison(
        plot_dir=plot_dir,
        inference_algs_results_by_dataset=inference_algs_results_by_dataset,
        sampled_mog_results_by_dataset=sampled_mog_results_by_dataset)


def run_one_dataset(plot_dir,
                    num_gaussians: int = 3,
                    gaussian_cov_scaling: float = 0.3,
                    gaussian_mean_prior_cov_scaling: float = 6.):
    # sample data
    sampled_mog_results = sample_sequence_from_mixture_of_gaussians(
        seq_len=100,
        class_sampling='Uniform',
        alpha=None,
        num_gaussians=num_gaussians,
        gaussian_params=dict(gaussian_cov_scaling=gaussian_cov_scaling,
                             gaussian_mean_prior_cov_scaling=gaussian_mean_prior_cov_scaling))

    bayesian_recursion_results = run_and_plot_bayesian_recursion(
        sampled_mog_results=sampled_mog_results,
        plot_dir=plot_dir)

    dp_means_offline_results = run_and_plot_dp_means_offline(
        sampled_mog_results=sampled_mog_results,
        plot_dir=plot_dir)

    dp_means_online_results = run_and_plot_dp_means_online(
        sampled_mog_results=sampled_mog_results,
        plot_dir=plot_dir)

    hmc_gibbs_5000_samples_results = run_and_plot_hmc_gibbs_sampling(
        sampled_mog_results=sampled_mog_results,
        plot_dir=plot_dir,
        gaussian_cov_scaling=gaussian_cov_scaling,
        gaussian_mean_prior_cov_scaling=gaussian_mean_prior_cov_scaling,
        num_samples=5000)

    hmc_gibbs_20000_samples_results = run_and_plot_hmc_gibbs_sampling(
        sampled_mog_results=sampled_mog_results,
        plot_dir=plot_dir,
        gaussian_cov_scaling=gaussian_cov_scaling,
        gaussian_mean_prior_cov_scaling=gaussian_mean_prior_cov_scaling,
        num_samples=20000)

    variational_bayes_results = run_and_plot_variational_bayes(
        sampled_mog_results=sampled_mog_results,
        plot_dir=plot_dir)

    inference_algs_results = {
        'Bayesian Recursion': bayesian_recursion_results,
        'HMC-Gibbs (5k Samples)': hmc_gibbs_5000_samples_results,
        'HMC-Gibbs (20k Samples)': hmc_gibbs_20000_samples_results,
        'DP-Means (Online)': dp_means_online_results,
        'DP-Means (Offline)': dp_means_offline_results,
        'Variational Bayes': variational_bayes_results,
    }

    return inference_algs_results, sampled_mog_results


def run_and_plot_bayesian_recursion(sampled_mog_results,
                                    plot_dir):

    alphas = 0.01 + np.arange(0., 5.01, 0.25)
    bayesian_recursion_plot_dir = os.path.join(plot_dir, 'bayesian_recursion')
    os.makedirs(bayesian_recursion_plot_dir, exist_ok=True)
    num_clusters_by_alpha = {}
    scores_by_alpha = {}
    for alpha in alphas:
        bayesian_recursion_results = bayesian_recursion(
            observations=sampled_mog_results['gaussian_samples_seq'],
            alpha=alpha,
            likelihood_model='multivariate_normal',
            em_learning_rate=1e0,
            # likelihood_fn=likelihood_fn,
            # update_parameters_fn=update_parameters_fn,
        )

        # record scores
        scores, pred_cluster_labels = score_predicted_clusters(
            true_cluster_labels=sampled_mog_results['assigned_table_seq'],
            table_assignment_posteriors=bayesian_recursion_results['table_assignment_posteriors'])
        scores_by_alpha[alpha] = scores

        # count number of clusters
        num_clusters_by_alpha[alpha] = len(np.unique(pred_cluster_labels))

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
    lambdas = 0.01 + np.arange(0., 5.01, 0.25)
    dp_means_plot_dir = os.path.join(plot_dir, 'dp_means_offline')
    os.makedirs(dp_means_plot_dir, exist_ok=True)
    num_clusters_by_lambda = {}
    scores_by_lambda = {}
    for lambd in lambdas:
        dp_means_offline_results = dp_means_offline(
            observations=sampled_mog_results['gaussian_samples_seq'],
            num_passes=8,
            lambd=lambd)

        # score clusters
        scores, pred_cluster_labels = score_predicted_clusters(
            true_cluster_labels=sampled_mog_results['assigned_table_seq'],
            table_assignment_posteriors=dp_means_offline_results['table_assignment_posteriors'])
        scores_by_lambda[lambd] = scores

        # count number of clusters
        num_clusters_by_lambda[lambd] = len(np.unique(pred_cluster_labels))

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
    lambdas = 0.01 + np.arange(0., 5.01, 0.25)
    dp_means_plot_dir = os.path.join(plot_dir, 'dp_means_online')
    os.makedirs(dp_means_plot_dir, exist_ok=True)
    num_clusters_by_lambda = {}
    scores_by_lambda = {}
    for lambd in lambdas:
        dp_means_online_results = dp_means_online(
            observations=sampled_mog_results['gaussian_samples_seq'],
            lambd=lambd)

        # score clusters
        scores, pred_cluster_labels = score_predicted_clusters(
            true_cluster_labels=sampled_mog_results['assigned_table_seq'],
            table_assignment_posteriors=dp_means_online_results['table_assignment_posteriors'])
        scores_by_lambda[lambd] = scores

        # count number of clusters
        num_clusters_by_lambda[lambd] = len(np.unique(pred_cluster_labels))

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


def run_and_plot_hmc_gibbs_sampling(sampled_mog_results,
                                    plot_dir,
                                    gaussian_cov_scaling,
                                    gaussian_mean_prior_cov_scaling,
                                    num_samples: int = 5000):
    hmc_gibbs_sampling_plot_dir = os.path.join(plot_dir,
                                               f'hmc_gibbs_sampling_nsamples={num_samples}')
    os.makedirs(hmc_gibbs_sampling_plot_dir, exist_ok=True)
    num_clusters_by_alpha = {}
    scores_by_alpha = {}
    alphas = np.arange(0.01, 5.01, 0.25)
    for alpha in alphas:
        sampling_hmc_gibbs_results = sampling_hmc_gibbs(
            observations=sampled_mog_results['gaussian_samples_seq'],
            num_samples=num_samples,
            alpha=alpha,
            gaussian_cov_scaling=gaussian_cov_scaling,
            gaussian_mean_prior_cov_scaling=gaussian_mean_prior_cov_scaling)

        # # score clusters
        scores, pred_cluster_labels = score_predicted_clusters(
            true_cluster_labels=sampled_mog_results['assigned_table_seq'],
            table_assignment_posteriors=sampling_hmc_gibbs_results['table_assignment_posteriors'])
        scores_by_alpha[alpha] = scores

        # count number of clusters
        num_clusters_by_alpha[alpha] = len(np.unique(pred_cluster_labels))

        plot_inference_results(
            sampled_mog_results=sampled_mog_results,
            inference_results=sampling_hmc_gibbs_results,
            inference_alg='hmc_gibbs={:.2f}'.format(alpha),
            plot_dir=hmc_gibbs_sampling_plot_dir)

    sampling_hmc_gibbs_results = dict(
        num_clusters_by_param=num_clusters_by_alpha,
        scores_by_param=pd.DataFrame(scores_by_alpha).T
    )

    return sampling_hmc_gibbs_results


def run_and_plot_variational_bayes(sampled_mog_results,
                                   plot_dir):
    alphas = 0.01 + np.arange(0., 5.01, 0.25)
    variational_plot_dir = os.path.join(plot_dir, 'variational')
    os.makedirs(variational_plot_dir, exist_ok=True)
    num_clusters_by_alpha = {}
    scores_by_alpha = {}
    for alpha in alphas:
        variational_bayes_results = variational_bayes(
            observations=sampled_mog_results['gaussian_samples_seq'],
            alpha=alpha)

        # score clusters
        scores, pred_cluster_labels = score_predicted_clusters(
            true_cluster_labels=sampled_mog_results['assigned_table_seq'],
            table_assignment_posteriors=variational_bayes_results['table_assignment_posteriors'])
        scores_by_alpha[alpha] = scores

        # count number of clusters
        num_clusters_by_alpha[alpha] = len(np.unique(pred_cluster_labels))

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

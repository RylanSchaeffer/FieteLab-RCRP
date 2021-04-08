import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from utils.helpers import assert_no_nan_no_inf
from utils.inference_mix_of_bernoullis import bayesian_recursion
from utils.metrics import score_predicted_clusters


def main():
    plot_dir = 'exp_07_olfactory/plots'
    os.makedirs(plot_dir, exist_ok=True)
    np.random.seed(1)

    load_olfaction_results = load_olfaction_data()
    odors_df = load_olfaction_results['odors_df']
    features_df = load_olfaction_results['features_df']
    num_obs = odors_df.shape[0]

    num_permutations = 5
    inference_algs_results_by_dataset = {}
    sampled_mog_results_by_dataset = {}

    for permutation_idx in range(num_permutations):

        dataset_dir = os.path.join(plot_dir, f'dataset={permutation_idx}')
        dataset_results_path = os.path.join(dataset_dir, 'dataset_results.joblib')

        # generate permutation and reorder data
        index_permutation = np.random.permutation(np.arange(num_obs, dtype=np.int))
        odors_df = odors_df[index_permutation]
        features_df = features_df[index_permutation]

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

        inference_algs_results_by_dataset[permutation_idx] = dataset_results['dataset_inference_algs_results']
        sampled_mog_results_by_dataset[permutation_idx] = dataset_results['dataset_sampled_mog_results']

    plot_inference_algs_comparison(
        plot_dir=plot_dir,
        inference_algs_results_by_dataset=inference_algs_results_by_dataset,
        sampled_mog_results_by_dataset=sampled_mog_results_by_dataset)


def load_olfaction_data():

    # TODO: fetch dataset if not in directory
    # 55k rows, 38 columns
    olfaction_df = pd.read_csv('exp_07_olfactory/12868_2016_287_MOESM1_ESM_headers_removed.csv')

    odors_df = olfaction_df['C.A.S.'].copy()
    olfaction_cols = olfaction_df.columns.values.tolist()

    features_df = olfaction_df[olfaction_cols[15:]].copy()
    features_df['CAN OR CAN\'T SMELL'] = olfaction_df['CAN OR CAN\'T SMELL'].replace({
        'I smell something': 100.,
        'I can\'t smell anything': 0.,
        np.nan: 50.
    })
    features_df['KNOW OR DON\'T KNOW THE SMELL'] = olfaction_df['KNOW OR DON\'T KNOW THE SMELL'].replace({
        'I don\'t know what the odor is': 0.,
        'I know what the odor is': 100.,
        np.nan: 50.
    })
    features_df[olfaction_cols[15:]] = features_df[olfaction_cols[15:]].fillna(value=50)
    features_df /= 100.

    # features_df.plot.hist(subplots=True, legend=True)
    # for col in features_df:
    #     plt.hist(features_df[col])
    #     plt.title(f'{col}')
    #     plt.show()

    results = dict(
        odors_df=odors_df,
        features_df=features_df)

    return results


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

    alphas = 0.01 + np.arange(0., 50.01, 1.)
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


if __name__ == '__main__':
    main()

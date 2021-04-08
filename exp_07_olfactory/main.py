import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

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
        odors_df = odors_df.iloc[index_permutation]
        features_df = features_df.iloc[index_permutation]

        if os.path.isfile(dataset_results_path):
            # load from disk if exists
            dataset_results = joblib.load(dataset_results_path)
        else:
            # otherwise, generate anew
            dataset_inference_algs_results = run_one_dataset(
                odors_df=odors_df,
                features_df=features_df,
                plot_dir=dataset_dir)
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
    # https://bmcneurosci.biomedcentral.com/articles/10.1186/s12868-016-0287-2
    # https://www.nature.com/articles/s41598-020-73978-1.pdf
    # 55k rows, 38 columns
    olfaction_df = pd.read_csv('exp_07_olfactory/12868_2016_287_MOESM1_ESM_headers_removed.csv')

    olfaction_df = olfaction_df.iloc[:100]

    olfaction_cols = olfaction_df.columns.values.tolist()

    # create target series
    odors_df = olfaction_df[['C.A.S.']].copy()
    odors_df['C.A.S.'] = pd.factorize(odors_df['C.A.S.'])[0]

    # maybe add odor dilution back in? ['Odor dilution']
    features_df = olfaction_df[olfaction_cols[15:]].copy()
    features_df['CAN OR CAN\'T SMELL'] = olfaction_df['CAN OR CAN\'T SMELL'].replace({
        'I smell something': 100.,
        'I can\'t smell anything': 0.,
    })
    features_df['KNOW OR DON\'T KNOW THE SMELL'] = olfaction_df['KNOW OR DON\'T KNOW THE SMELL'].replace({
        'I don\'t know what the odor is': 0.0,
        'I know what the odor is': 100.,
    })
    features_df /= 100.
    # replace nans with random normals centered at 0
    for col in features_df:
        nan_idx = pd.isna(features_df[col])
        replacement_vals = np.random.normal(
            loc=0.5, scale=np.sqrt(0.005), size=nan_idx.sum())
        replacement_vals[replacement_vals < 0.01] = 0.01
        replacement_vals[replacement_vals > 0.99] = 0.99
        features_df.loc[nan_idx, col] = replacement_vals

    # features_df.plot.hist(subplots=True, legend=True)
    # for col in features_df:
    #     plt.hist(features_df[col])
    #     plt.title(f'{col}')
    #     plt.show()

    results = dict(
        odors_df=odors_df,
        features_df=features_df)

    return results


def run_one_dataset(features_df,
                    odors_df,
                    plot_dir):

    bayesian_recursion_results = run_and_plot_bayesian_recursion(
        features_df=features_df,
        odors_df=odors_df,
        plot_dir=plot_dir)

    # dp_means_offline_results = run_and_plot_dp_means_offline(
    #     sampled_mog_results=sampled_mog_results,
    #     plot_dir=plot_dir)
    #
    # dp_means_online_results = run_and_plot_dp_means_online(
    #     sampled_mog_results=sampled_mog_results,
    #     plot_dir=plot_dir)
    #
    # hmc_gibbs_5000_samples_results = run_and_plot_hmc_gibbs_sampling(
    #     sampled_mog_results=sampled_mog_results,
    #     plot_dir=plot_dir,
    #     gaussian_cov_scaling=gaussian_cov_scaling,
    #     gaussian_mean_prior_cov_scaling=gaussian_mean_prior_cov_scaling,
    #     num_samples=5000)
    #
    # hmc_gibbs_20000_samples_results = run_and_plot_hmc_gibbs_sampling(
    #     sampled_mog_results=sampled_mog_results,
    #     plot_dir=plot_dir,
    #     gaussian_cov_scaling=gaussian_cov_scaling,
    #     gaussian_mean_prior_cov_scaling=gaussian_mean_prior_cov_scaling,
    #     num_samples=20000)
    #
    # variational_bayes_results = run_and_plot_variational_bayes(
    #     sampled_mog_results=sampled_mog_results,
    #     plot_dir=plot_dir)

    inference_algs_results = {
        'Bayesian Recursion': bayesian_recursion_results,
        # 'HMC-Gibbs (5k Samples)': hmc_gibbs_5000_samples_results,
        # 'HMC-Gibbs (20k Samples)': hmc_gibbs_20000_samples_results,
        # 'DP-Means (Online)': dp_means_online_results,
        # 'DP-Means (Offline)': dp_means_offline_results,
        # 'Variational Bayes': variational_bayes_results,
    }

    return inference_algs_results, sampled_mog_results


def run_and_plot_bayesian_recursion(features_df,
                                    odors_df,
                                    plot_dir):

    alphas = 0.01 + np.arange(0., 50.01, 1.)
    bayesian_recursion_plot_dir = os.path.join(plot_dir, 'bayesian_recursion')
    os.makedirs(bayesian_recursion_plot_dir, exist_ok=True)
    num_clusters_by_alpha = {}
    scores_by_alpha = {}
    for alpha in alphas:
        bayesian_recursion_results = bayesian_recursion(
            observations=features_df.values,
            alpha=alpha)

        # record scores
        scores, pred_cluster_labels = score_predicted_clusters(
            true_cluster_labels=odors_df['C.A.S.'],
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

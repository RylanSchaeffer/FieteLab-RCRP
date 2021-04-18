import joblib
import numpy as np
import os
import pandas as pd
import torch

from exp_08_omniglot.plot import plot_inference_results, plot_inference_algs_comparison

import utils.data
import utils.helpers
import utils.inference_mix_of_cont_bernoullis
import utils.inference
from utils.metrics import score_predicted_clusters


def main():
    exp_dir = 'exp_08_omniglot'

    plot_dir = os.path.join(exp_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    np.random.seed(1)
    torch.manual_seed(0)

    omniglot_dataset_results = utils.data.load_omniglot_dataset(
        data_dir='data')

    num_obs = omniglot_dataset_results['labels'].shape[0]
    num_permutations = 5
    inference_algs_results_by_dataset = {}
    sampled_permutation_indices_by_dataset = {}
    for dataset_idx in range(num_permutations):

        dataset_dir = os.path.join(plot_dir, f'dataset={dataset_idx}')

        # generate permutation and reorder data
        index_permutation = np.random.permutation(np.arange(num_obs, dtype=np.int))
        sampled_permutation_indices_by_dataset[dataset_idx] = index_permutation
        omniglot_dataset_results['images'] = omniglot_dataset_results['images'][index_permutation]
        omniglot_dataset_results['labels'] = omniglot_dataset_results['labels'][index_permutation]

        dataset_results_path = os.path.join(dataset_dir, 'dataset_results.joblib')
        if os.path.isfile(dataset_results_path):
            # load from disk if exists
            dataset_results = joblib.load(dataset_results_path)
        else:
            # otherwise, generate anew
            dataset_inference_algs_results = run_one_dataset(
                omniglot_dataset_results=omniglot_dataset_results,
                plot_dir=dataset_dir)
            dataset_results = dict(
                dataset_inference_algs_results=dataset_inference_algs_results,
            )
            joblib.dump(dataset_results, dataset_results_path)

            # delete variables from memory and perform fresh read from disk
            del dataset_inference_algs_results
            del dataset_results
            dataset_results = joblib.load(dataset_results_path)

        inference_algs_results_by_dataset[dataset_idx] = dataset_results['dataset_inference_algs_results']

    plot_inference_algs_comparison(
        images=omniglot_dataset_results['images'],
        labels=omniglot_dataset_results['labels'],
        plot_dir=plot_dir,
        inference_algs_results_by_dataset=inference_algs_results_by_dataset,
        sampled_permutation_indices_by_dataset=sampled_permutation_indices_by_dataset)

    print('Successfully completed Exp 08 Omniglot')


def run_one_dataset(omniglot_dataset_results,
                    plot_dir):

    bayesian_recursion_results = run_and_plot_bayesian_recursion(
        omniglot_dataset_results=omniglot_dataset_results,
        plot_dir=plot_dir)

    inference_algs_results = {
        'Bayesian Recursion': bayesian_recursion_results,
        # 'HMC-Gibbs (5k Samples)': hmc_gibbs_5000_samples_results,
        # 'HMC-Gibbs (20k Samples)': hmc_gibbs_20000_samples_results,
        # 'DP-Means (Online)': dp_means_online_results,
        # 'DP-Means (Offline)': dp_means_offline_results,
        # 'Variational Bayes': variational_bayes_results,
    }

    return inference_algs_results


def run_and_plot_bayesian_recursion(omniglot_dataset_results,
                                    plot_dir):

    alphas = np.arange(1., 2.01, 1.)
    bayesian_recursion_plot_dir = os.path.join(plot_dir, 'bayesian_recursion')
    os.makedirs(bayesian_recursion_plot_dir, exist_ok=True)
    num_clusters_by_alpha = {}
    scores_by_alpha = {}
    for alpha in alphas:

        # bayesian_recursion_results = utils.inference_mix_of_cont_bernoullis.bayesian_recursion(
        #     observations=images.reshape(images.shape[0], -1),
        #     alpha=alpha)

        bayesian_recursion_results = utils.inference.recursive_crp(
            observations=omniglot_dataset_results['pca_images'],
            # likelihood_model='continuous_bernoulli',
            likelihood_model='multivariate_normal',
            learning_rate=1e1,
            concentration_param=alpha)
        
        # bayesian_recursion_results = utils.inference.bayesian_recursion(
        #     observations=omniglot_dataset_results['images'].reshape(omniglot_dataset_results['images'].shape[0], -1),
        #     likelihood_model='continuous_bernoulli',
        #     em_learning_rate=1e1,
        #     alpha=alpha)

        # add cluster probs from cluster logits
        # bayesian_recursion_results['parameters']['probs'] = utils.helpers.numpy_logits_to_probs(
        #     bayesian_recursion_results['parameters']['logits'])

        # record scores
        scores, pred_cluster_labels = score_predicted_clusters(
            true_cluster_labels=omniglot_dataset_results['labels'],
            table_assignment_posteriors=bayesian_recursion_results['table_assignment_posteriors'])
        scores_by_alpha[alpha] = scores

        # count number of clusters
        num_clusters_by_alpha[alpha] = len(np.unique(pred_cluster_labels))

        plot_inference_results(
            omniglot_dataset_results=omniglot_dataset_results,
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

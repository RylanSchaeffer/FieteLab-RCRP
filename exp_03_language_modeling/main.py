import joblib
import numpy as np
import os
import pandas as pd
from timeit import default_timer as timer
import torch

# from exp_03_language_modeling.plot import plot_inference_algs_comparison

import utils.data
import utils.inference
import utils.metrics
import utils.plot


def main():
    plot_dir = 'exp_03_language_modeling/plots'
    os.makedirs(plot_dir, exist_ok=True)
    np.random.seed(1)
    torch.manual_seed(0)

    reddit_dataset_results = utils.data.load_reddit_dataset(
        data_dir='data')

    # plot number of topics versus number of posts
    utils.plot.plot_num_clusters_by_num_obs(
        true_cluster_labels=reddit_dataset_results['assigned_table_seq'],
        plot_dir=plot_dir)

    num_obs = reddit_dataset_results['assigned_table_seq'].shape[0]
    num_permutations = 5
    inference_algs_results_by_dataset_idx = {}
    sampled_permutation_indices_by_dataset_idx = {}

    # generate lots of datasets and record performance for each
    for dataset_idx in range(num_permutations):
        print(f'Dataset Index: {dataset_idx}')
        dataset_dir = os.path.join(plot_dir, f'dataset={dataset_idx}')
        os.makedirs(dataset_dir, exist_ok=True)

        # generate permutation and reorder data
        index_permutation = np.random.permutation(np.arange(num_obs, dtype=np.int))
        sampled_permutation_indices_by_dataset_idx[dataset_idx] = index_permutation
        reddit_dataset_results['assigned_table_seq'] = reddit_dataset_results['assigned_table_seq'][index_permutation]
        reddit_dataset_results['observations_tfidf'] = reddit_dataset_results['observations_tfidf'][index_permutation]

        dataset_inference_algs_results = run_one_dataset(
            dataset_dir=dataset_dir,
            reddit_dataset_results=reddit_dataset_results)
        inference_algs_results_by_dataset_idx[dataset_idx] = dataset_inference_algs_results

    utils.plot.plot_inference_algs_comparison(
        plot_dir=plot_dir,
        inference_algs_results_by_dataset_idx=inference_algs_results_by_dataset_idx,
        dataset_by_dataset_idx=sampled_permutation_indices_by_dataset_idx)

    print('Successfully completed Exp 03 Language Modeling')


def run_one_dataset(reddit_dataset_results,
                    dataset_dir):

    concentration_params = np.linspace(0.1*np.log(reddit_dataset_results['assigned_table_seq'].shape[0]),
                                       10*np.log(reddit_dataset_results['assigned_table_seq'].shape[0]),
                                       11)

    inference_alg_strs = [
        # online algorithms
        'R-CRP',
        'SUSG',  # deterministically select highest table assignment posterior
        'Online CRP',  # sample from table assignment posterior; potentially correct
        # offline algorithms
        'HMC-Gibbs (5000 Samples)',
        'HMC-Gibbs (20000 Samples)',
        'SVI (5k Steps)',
        'SVI (20k Steps)',
    ]

    inference_algs_results = {}
    for inference_alg_str in inference_alg_strs:

        inference_alg_dir = os.path.join(dataset_dir, inference_alg_str)
        os.makedirs(inference_alg_dir, exist_ok=True)
        inference_alg_results_path = os.path.join(inference_alg_dir, 'results.joblib')

        # if results do not exist, generate
        if not os.path.isfile(inference_alg_results_path):
            inference_alg_results = run_and_plot_inference_alg(
                reddit_dataset_results=reddit_dataset_results,
                inference_alg_str=inference_alg_str,
                concentration_params=concentration_params,
                plot_dir=dataset_dir)

            # write to disk and delete results from memory
            joblib.dump(inference_alg_results, filename=inference_alg_results_path)
            del inference_alg_results

        # read results from disk
        inference_alg_results = joblib.load(inference_alg_results_path)
        inference_algs_results[inference_alg_str] = inference_alg_results

    return inference_algs_results


def run_and_plot_inference_alg(reddit_dataset_results,
                               inference_alg_str,
                               concentration_params,
                               plot_dir):

    inference_alg_plot_dir = os.path.join(plot_dir, inference_alg_str)
    os.makedirs(inference_alg_plot_dir, exist_ok=True)
    num_clusters_by_concentration_param = {}
    scores_by_concentration_param = {}
    runtimes_by_concentration_param = {}

    for concentration_param in concentration_params:

        # run inference algorithm
        # time using timer because https://stackoverflow.com/a/25823885/4570472
        start_time = timer()
        inference_alg_results = utils.inference.run_inference_alg(
            inference_alg_str=inference_alg_str,
            observations=reddit_dataset_results['observations_tfidf'],
            concentration_param=concentration_param,
            likelihood_model='dirichlet_multinomial',
            learning_rate=1e0)

        # record elapsed time
        stop_time = timer()
        runtimes_by_concentration_param[concentration_param] = stop_time - start_time

        # record scores
        scores, pred_cluster_labels = utils.metrics.score_predicted_clusters(
            true_cluster_labels=reddit_dataset_results['true_cluster_labels'],
            table_assignment_posteriors=inference_alg_results['table_assignment_posteriors'])
        scores_by_concentration_param[concentration_param] = scores

        # count number of clusters
        num_clusters_by_concentration_param[concentration_param] = len(np.unique(pred_cluster_labels))

        # plot_inference_results(
        #     sampled_mog_results=sampled_mog_results,
        #     inference_results=inference_alg_results,
        #     inference_alg_str=inference_alg_str,
        #     concentration_param=concentration_param,
        #     plot_dir=inference_alg_plot_dir)

        print('Finished {} concentration_param={:.2f}'.format(inference_alg_str, concentration_param))

    inference_alg_results = dict(
        num_clusters_by_param=num_clusters_by_concentration_param,
        scores_by_param=pd.DataFrame(scores_by_concentration_param).T)

    return inference_alg_results


if __name__ == '__main__':
    main()

import joblib
import numpy as np
import os
import pandas as pd
from timeit import default_timer as timer
import torch

from exp_04_newsgroup.plot import plot_inference_algs_comparison

import utils.data
import utils.inference
import utils.metrics
import utils.plot


def main():
    num_data = 20000
    num_features = 500
    tf_or_tfidf_or_counts = 'counts'
    plot_dir = f'exp_04_newsgroup/plots_obs={tf_or_tfidf_or_counts}_ndata={num_data}_nfeat={num_features}'
    os.makedirs(plot_dir, exist_ok=True)
    np.random.seed(1)
    torch.manual_seed(0)

    newsgroup_dataset_results = utils.data.load_newsgroup_dataset(
        data_dir='data',
        num_data=num_data,
        num_features=num_features,
        tf_or_tfidf_or_counts=tf_or_tfidf_or_counts)

    # plot number of topics versus number of posts
    utils.plot.plot_num_clusters_by_num_obs(
        true_cluster_labels=newsgroup_dataset_results['assigned_table_seq'],
        plot_dir=plot_dir)

    num_obs = newsgroup_dataset_results['assigned_table_seq'].shape[0]
    num_permutations = 1
    inference_algs_results_by_dataset_idx = {}
    dataset_by_dataset_idx = {}

    # generate lots of datasets and record performance for each
    for dataset_idx in range(num_permutations):
        print(f'Dataset Index: {dataset_idx}')
        dataset_dir = os.path.join(plot_dir, f'dataset={dataset_idx}')
        os.makedirs(dataset_dir, exist_ok=True)

        # generate permutation and reorder data
        index_permutation = np.random.permutation(np.arange(num_obs, dtype=np.int))
        newsgroup_dataset_results['assigned_table_seq'] = newsgroup_dataset_results['assigned_table_seq'][index_permutation]
        newsgroup_dataset_results['observations_transformed'] = newsgroup_dataset_results['observations_transformed'][index_permutation]
        dataset_by_dataset_idx[dataset_idx] = dict(
            assigned_table_seq=np.copy(newsgroup_dataset_results['assigned_table_seq']),
            observations=np.copy(newsgroup_dataset_results['observations_transformed']))

        dataset_inference_algs_results = run_one_dataset(
            dataset_dir=dataset_dir,
            dataset_idx=dataset_idx,
            newsgroup_dataset_results=newsgroup_dataset_results)
        inference_algs_results_by_dataset_idx[dataset_idx] = dataset_inference_algs_results

    utils.plot.plot_inference_algs_comparison(
        plot_dir=plot_dir,
        inference_algs_results_by_dataset_idx=inference_algs_results_by_dataset_idx,
        dataset_by_dataset_idx=dataset_by_dataset_idx)

    print('Successfully completed Exp 03 Language Modeling')


def run_one_dataset(newsgroup_dataset_results,
                    dataset_idx,
                    dataset_dir):

    concentration_params = np.linspace(0.1 * np.log(newsgroup_dataset_results['assigned_table_seq'].shape[0]),
                                       5 * np.log(newsgroup_dataset_results['assigned_table_seq'].shape[0]),
                                       7)

    inference_alg_strs = [
        # online algorithms
        'R-CRP',
        'SUGS',  # deterministically select highest table assignment posterior
        'Online CRP',  # sample from table assignment posterior; potentially correct
        # offline algorithms
        # 'HMC-Gibbs (5000 Samples)',
        # 'HMC-Gibbs (20000 Samples)',
        # 'SVI (20k Steps)',
        # 'SVI (50k Steps)',
    ]

    inference_algs_results = {}
    for inference_alg_str in inference_alg_strs:
        inference_alg_results = run_and_plot_inference_alg(
            newsgroup_dataset_results=newsgroup_dataset_results,
            inference_alg_str=inference_alg_str,
            concentration_params=concentration_params,
            plot_dir=dataset_dir)
        inference_algs_results[inference_alg_str] = inference_alg_results
    return inference_algs_results


def run_and_plot_inference_alg(newsgroup_dataset_results,
                               inference_alg_str,
                               concentration_params,
                               plot_dir):

    inference_alg_plot_dir = os.path.join(plot_dir, inference_alg_str)
    os.makedirs(inference_alg_plot_dir, exist_ok=True)
    num_clusters_by_concentration_param = {}
    scores_by_concentration_param = {}
    runtimes_by_concentration_param = {}

    for concentration_param in concentration_params:

        inference_alg_results_concentration_param_path = os.path.join(
            inference_alg_plot_dir,
            f'results_{np.round(concentration_param, 2)}.joblib')

        # if results do not exist, generate
        if not os.path.isfile(inference_alg_results_concentration_param_path):

            # run inference algorithm
            # time using timer because https://stackoverflow.com/a/25823885/4570472
            start_time = timer()
            inference_alg_concentration_param_results = utils.inference.run_inference_alg(
                inference_alg_str=inference_alg_str,
                observations=newsgroup_dataset_results['observations_transformed'],
                concentration_param=concentration_param,
                likelihood_model='dirichlet_multinomial',
                learning_rate=1e0)

            # record elapsed time
            stop_time = timer()
            runtime = stop_time - start_time

            # record scores
            scores, pred_cluster_labels = utils.metrics.score_predicted_clusters(
                true_cluster_labels=newsgroup_dataset_results['assigned_table_seq'],
                table_assignment_posteriors=inference_alg_concentration_param_results['table_assignment_posteriors'])

            # count number of clusters
            num_clusters = len(np.unique(pred_cluster_labels))

            # write to disk and delete
            data_to_store = dict(
                inference_alg_concentration_param_results=inference_alg_concentration_param_results,
                num_clusters=num_clusters,
                scores=scores,
                runtime=runtime,
            )

            joblib.dump(data_to_store,
                        filename=inference_alg_results_concentration_param_path)
            del inference_alg_concentration_param_results
            del data_to_store

        # read results from disk
        stored_data = joblib.load(
            inference_alg_results_concentration_param_path)

        # plot_inference_results(
        #     sampled_mog_results=sampled_mog_results,
        #     inference_results=inference_alg_results,
        #     inference_alg_str=inference_alg_str,
        #     concentration_param=concentration_param,
        #     plot_dir=inference_alg_plot_dir)

        num_clusters_by_concentration_param[concentration_param] = stored_data[
            'num_clusters']
        scores_by_concentration_param[concentration_param] = stored_data[
            'scores']
        runtimes_by_concentration_param[concentration_param] = stored_data[
            'runtime']

        print('Finished {} concentration_param={:.2f}'.format(inference_alg_str, concentration_param))

    inference_alg_results = dict(
        num_clusters_by_param=num_clusters_by_concentration_param,
        scores_by_param=pd.DataFrame(scores_by_concentration_param).T,
        runtimes_by_param=runtimes_by_concentration_param)

    return inference_alg_results


if __name__ == '__main__':
    main()

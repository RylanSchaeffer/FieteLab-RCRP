import joblib
import numpy as np
import os
import pandas as pd
from timeit import default_timer as timer
import torch

from exp_08_omniglot.plot import plot_inference_results

import utils.data
import utils.helpers
import utils.inference
import utils.metrics
import utils.plot


def main():
    plot_dir = 'exp_08_omniglot/plots'
    os.makedirs(plot_dir, exist_ok=True)
    np.random.seed(1)
    torch.manual_seed(0)

    omniglot_dataset_results = utils.data.load_omniglot_dataset(
        data_dir='data',
        num_data=1200,
        center_crop=True,
        avg_pool=True)

    # plot number of topics versus number of posts
    utils.plot.plot_num_clusters_by_num_obs(
        true_cluster_labels=omniglot_dataset_results['assigned_table_seq'],
        plot_dir=plot_dir)

    num_obs = omniglot_dataset_results['assigned_table_seq'].shape[0]
    num_permutations = 3
    inference_algs_results_by_dataset_idx = {}
    dataset_by_dataset_idx = {}

    # generate lots of datasets and record performance for each
    for dataset_idx in range(num_permutations):
        print(f'Dataset Index: {dataset_idx}')
        dataset_dir = os.path.join(plot_dir, f'dataset={dataset_idx}')
        os.makedirs(dataset_dir, exist_ok=True)

        # generate permutation and reorder data
        index_permutation = np.random.permutation(np.arange(num_obs, dtype=np.int))
        omniglot_dataset_results['pca_latents'] = omniglot_dataset_results['pca_latents'][index_permutation]
        omniglot_dataset_results['assigned_table_seq'] = omniglot_dataset_results['assigned_table_seq'][index_permutation]
        dataset_by_dataset_idx[dataset_idx] = dict(
            assigned_table_seq=np.copy(omniglot_dataset_results['assigned_table_seq']),
            observations=np.copy(omniglot_dataset_results['pca_latents']))

        dataset_inference_algs_results = run_one_dataset(
            dataset_dir=dataset_dir,
            omniglot_dataset_results=omniglot_dataset_results)
        inference_algs_results_by_dataset_idx[dataset_idx] = dataset_inference_algs_results

    utils.plot.plot_inference_algs_comparison(
        inference_algs_results_by_dataset_idx=inference_algs_results_by_dataset_idx,
        dataset_by_dataset_idx=dataset_by_dataset_idx,
        plot_dir=plot_dir)

    print('Successfully completed Exp 08 Omniglot')


def run_one_dataset(omniglot_dataset_results,
                    dataset_dir):

    concentration_params = np.linspace(0.1*np.log(omniglot_dataset_results['assigned_table_seq'].shape[0]),
                                       3*np.log(omniglot_dataset_results['assigned_table_seq'].shape[0]),
                                       11)

    inference_alg_strs = [
        # online algorithms
        'R-CRP',
        'SUSG',  # deterministically select highest table assignment posterior
        'Online CRP',  # sample from table assignment posterior; potentially correct
        'DP-Means (online)',  # deterministically select highest assignment posterior
        # offline algorithms
        # 'DP-Means (offline)',
        # 'HMC-Gibbs (5000 Samples)',
        # 'HMC-Gibbs (20000 Samples)',
        # 'SVI (5k Steps)',
        # 'SVI (20k Steps)',
        # 'Variational Bayes (15 Init, 30 Iter)',
        # 'Variational Bayes (5 Init, 30 Iter)',
        # 'Variational Bayes (1 Init, 8 Iter)',
    ]

    inference_algs_results = {}
    for inference_alg_str in inference_alg_strs:
        inference_alg_results = run_and_plot_inference_alg(
            omniglot_dataset_results=omniglot_dataset_results,
            inference_alg_str=inference_alg_str,
            concentration_params=concentration_params,
            plot_dir=dataset_dir)
        inference_algs_results[inference_alg_str] = inference_alg_results
    return inference_algs_results


def run_and_plot_inference_alg(omniglot_dataset_results,
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
            print(f'Generating {inference_alg_results_concentration_param_path}')

            # run inference algorithm
            # time using timer because https://stackoverflow.com/a/25823885/4570472
            start_time = timer()
            inference_alg_concentration_param_results = utils.inference.run_inference_alg(
                inference_alg_str=inference_alg_str,
                observations=omniglot_dataset_results['pca_latents'],
                concentration_param=concentration_param,
                likelihood_model='multivariate_normal',
                learning_rate=1e0)

            # record elapsed time
            stop_time = timer()
            runtime = stop_time - start_time

            # record scores
            scores, pred_cluster_labels = utils.metrics.score_predicted_clusters(
                true_cluster_labels=omniglot_dataset_results['assigned_table_seq'],
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
        else:
            print(f'Loading {inference_alg_results_concentration_param_path} from disk...')

        # read results from disk
        stored_data = joblib.load(
            inference_alg_results_concentration_param_path)

        plot_inference_results(
            omniglot_dataset_results=omniglot_dataset_results,
            inference_results=stored_data['inference_alg_concentration_param_results'],
            inference_alg_str=inference_alg_str,
            concentration_param=concentration_param,
            plot_dir=inference_alg_plot_dir)

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
        runtimes_by_param=runtimes_by_concentration_param,
    )

    return inference_alg_results


if __name__ == '__main__':
    main()

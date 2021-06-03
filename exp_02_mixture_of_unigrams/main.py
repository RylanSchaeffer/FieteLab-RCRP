import joblib
import os
from timeit import default_timer as timer

from exp_02_mixture_of_unigrams.plot import *
import utils.data
import utils.inference
import utils.metrics


def main():
    plot_dir = 'exp_02_mixture_of_unigrams/plots'
    os.makedirs(plot_dir, exist_ok=True)
    np.random.seed(1)

    num_datasets = 10
    inference_algs_results_by_dataset = {}
    sampled_mou_results_by_dataset_idx = {}

    # generate lots of datasets and record performance for each
    for dataset_idx in range(num_datasets):
        print(f'Dataset Index: {dataset_idx}')
        dataset_dir = os.path.join(plot_dir, f'dataset={dataset_idx}')
        os.makedirs(dataset_dir, exist_ok=True)
        dataset_inference_algs_results, dataset_sampled_mix_of_unigrams_results = run_one_dataset(
            dataset_dir=dataset_dir)

        inference_algs_results_by_dataset[dataset_idx] = dataset_inference_algs_results
        sampled_mou_results_by_dataset_idx[dataset_idx] = dataset_sampled_mix_of_unigrams_results

    plot_inference_algs_comparison(
        plot_dir=plot_dir,
        inference_algs_results_by_dataset_idx=inference_algs_results_by_dataset,
        dataset_by_dataset_idx=sampled_mou_results_by_dataset_idx)

    print('Successfully completed Exp 02 Mixture of Unigrams')


def run_one_dataset(dataset_dir,
                    dp_concentration_param: float = 4.6,
                    prior_over_topic_parameters: float = 0.3):
    # sample data
    sampled_mou_results = utils.data.sample_sequence_from_mixture_of_unigrams(
        seq_len=150,
        num_topics=7,
        unigram_params=dict(dp_concentration_param=dp_concentration_param,
                            prior_over_topic_parameters=prior_over_topic_parameters))

    concentration_params = 0.01 + np.arange(0., 10.01,
                                            1.,
                                            # 0.25
                                            )

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
        inference_alg_results = run_and_plot_inference_alg(
            sampled_mou_results=sampled_mou_results,
            inference_alg_str=inference_alg_str,
            concentration_params=concentration_params,
            plot_dir=dataset_dir)
        inference_algs_results[inference_alg_str] = inference_alg_results

    return inference_algs_results, sampled_mou_results


def run_and_plot_inference_alg(sampled_mou_results,
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
            start_time = timer()
            inference_alg_concentration_param_results = utils.inference.run_inference_alg(
                inference_alg_str=inference_alg_str,
                observations=sampled_mou_results['doc_samples_seq'],
                concentration_param=concentration_param,
                likelihood_model='dirichlet_multinomial',
                learning_rate=1e0)
            # record elapsed time
            stop_time = timer()
            runtime = stop_time - start_time

            # record scores
            scores, pred_cluster_labels = utils.metrics.score_predicted_clusters(
                true_cluster_labels=sampled_mou_results['assigned_table_seq'],
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

        # TODO: finish adapting this to Mixture of Unigrams
        # plot_inference_results(
        #     sampled_mou_results=sampled_mou_results,
        #     inference_results=stored_data['inference_alg_concentration_param_results'],
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

    inference_alg_concentration_param_results = dict(
        num_clusters_by_param=num_clusters_by_concentration_param,
        scores_by_param=pd.DataFrame(scores_by_concentration_param).T,
        runtimes_by_param=runtimes_by_concentration_param,
    )

    return inference_alg_concentration_param_results


if __name__ == '__main__':
    main()

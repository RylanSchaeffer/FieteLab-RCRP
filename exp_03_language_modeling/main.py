import joblib
import sklearn.datasets

from exp_03_language_modeling.plot import *

import utils.data
import utils.inference
import utils.metrics
import utils.plot


def main():
    exp_dir = 'exp_03_language_modeling'
    plot_dir = os.path.join(exp_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    reddit_dataset_results = utils.data.load_reddit_dataset()

    # plot number of topics versus number of posts
    utils.plot.plot_num_clusters_by_num_obs(
        true_cluster_labels=reddit_dataset_results['true_cluster_labels'],
        plot_dir=plot_dir)

    num_obs = reddit_dataset_results['true_cluster_labels'].shape[0]
    num_permutations = 5
    inference_algs_results_by_dataset = {}
    sampled_permutation_indices_by_dataset = {}
    for dataset_idx in range(num_permutations):

        dataset_dir = os.path.join(plot_dir, f'dataset={dataset_idx}')
        os.makedirs(dataset_dir, exist_ok=True)

        # generate permutation and reorder data
        index_permutation = np.random.permutation(np.arange(num_obs, dtype=np.int))
        sampled_permutation_indices_by_dataset[dataset_idx] = index_permutation
        reddit_dataset_results['true_cluster_labels'] = reddit_dataset_results['true_cluster_labels'][index_permutation]
        reddit_dataset_results['observations_tfidf'] = reddit_dataset_results['observations_tfidf'][index_permutation]

        dataset_inference_algs_results, dataset_sampled_mog_results = run_one_dataset(
            reddit_dataset_results=reddit_dataset_results,
            dataset_dir=dataset_dir)

        inference_algs_results_by_dataset[dataset_idx] = dataset_inference_algs_results

    plot_inference_algs_comparison(
        plot_dir=plot_dir,
        inference_algs_results_by_dataset_idx=inference_algs_results_by_dataset,
        dataset_by_dataset_idx=sampled_permutation_indices_by_dataset)

    print('Successfully completed Exp 03 Language Modeling')


def run_one_dataset(dataset_dir,
                    reddit_dataset_results):

    concentration_params = np.arange(500., 2001, 100.,)

    inference_alg_strs = [
        # online algorithms
        'R-CRP',
        'SUSG',
        # offline algorithms
        # 'DP-Means (offline)',
        # 'HMC-Gibbs',
        # 'SVI',
        # 'Variational Bayes',
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

    for concentration_param in concentration_params:

        # run inference algorithm
        inference_alg_results = utils.inference.run_inference_alg(
            inference_alg_str=inference_alg_str,
            observations=reddit_dataset_results['observations_tfidf'],
            concentration_param=concentration_param,
            likelihood_model='dirichlet_multinomial',
            learning_rate=1e0)

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

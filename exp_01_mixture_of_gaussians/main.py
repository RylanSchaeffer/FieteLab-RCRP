import joblib
import os

from exp_01_mixture_of_gaussians.plot import *
import utils.data
import utils.inference
import utils.metrics


def main():
    plot_dir = 'exp_01_mixture_of_gaussians/plots'
    os.makedirs(plot_dir, exist_ok=True)
    np.random.seed(1)

    num_datasets = 5
    inference_algs_results_by_dataset = {}
    sampled_mog_results_by_dataset = {}

    # generate lots of datasets and record performance for each
    for dataset_idx in range(num_datasets):
        print(f'Dataset Index: {dataset_idx}')
        dataset_dir = os.path.join(plot_dir, f'dataset={dataset_idx}')
        os.makedirs(dataset_dir, exist_ok=True)
        dataset_inference_algs_results, dataset_sampled_mog_results = run_one_dataset(
            dataset_dir=dataset_dir)

        inference_algs_results_by_dataset[dataset_idx] = dataset_inference_algs_results
        sampled_mog_results_by_dataset[dataset_idx] = dataset_sampled_mog_results

    plot_inference_algs_comparison(
        plot_dir=plot_dir,
        inference_algs_results_by_dataset=inference_algs_results_by_dataset,
        sampled_mog_results_by_dataset=sampled_mog_results_by_dataset)


def run_one_dataset(dataset_dir,
                    num_gaussians: int = 3,
                    gaussian_cov_scaling: float = 0.3,
                    gaussian_mean_prior_cov_scaling: float = 6.):

    # sample data
    sampled_mog_results = utils.data.sample_sequence_from_mixture_of_gaussians(
        seq_len=100,
        class_sampling='Uniform',
        alpha=None,
        num_gaussians=num_gaussians,
        gaussian_params=dict(gaussian_cov_scaling=gaussian_cov_scaling,
                             gaussian_mean_prior_cov_scaling=gaussian_mean_prior_cov_scaling))

    concentration_params = 0.01 + np.arange(0., 5.01,
                                            1.,
                                            # 0.25
                                            )

    inference_alg_strs = [
        # online algorithms
        'R-CRP',
        'SUSG',
        'DP-Means (online)',
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
                sampled_mog_results=sampled_mog_results,
                inference_alg_str=inference_alg_str,
                concentration_params=concentration_params,
                plot_dir=dataset_dir)

            # write to disk and delete
            joblib.dump(inference_alg_results, filename=inference_alg_results_path)
            del inference_alg_results

        # read results from disk
        inference_alg_results = joblib.load(inference_alg_results_path)
        inference_algs_results[inference_alg_str] = inference_alg_results

    return inference_algs_results, sampled_mog_results


def run_and_plot_inference_alg(sampled_mog_results,
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
            observations=sampled_mog_results['gaussian_samples_seq'],
            concentration_param=concentration_param,
            likelihood_model='multivariate_normal',
            learning_rate=1e0)

        # record scores
        scores, pred_cluster_labels = utils.metrics.score_predicted_clusters(
            true_cluster_labels=sampled_mog_results['assigned_table_seq'],
            table_assignment_posteriors=inference_alg_results['table_assignment_posteriors'])
        scores_by_concentration_param[concentration_param] = scores

        # count number of clusters
        num_clusters_by_concentration_param[concentration_param] = len(np.unique(pred_cluster_labels))

        plot_inference_results(
            sampled_mog_results=sampled_mog_results,
            inference_results=inference_alg_results,
            inference_alg_str=inference_alg_str,
            concentration_param=concentration_param,
            plot_dir=inference_alg_plot_dir)

        print('Finished {} concentration_param={:.2f}'.format(inference_alg_str, concentration_param))

    inference_alg_results = dict(
        num_clusters_by_param=num_clusters_by_concentration_param,
        scores_by_param=pd.DataFrame(scores_by_concentration_param).T,
    )

    return inference_alg_results


if __name__ == '__main__':
    main()

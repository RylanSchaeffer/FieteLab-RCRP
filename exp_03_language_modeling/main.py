import joblib
import numpy as np
import os
import pandas as pd
import sklearn.datasets
import sklearn.feature_extraction.text
import tensorflow as tf
import tensorflow_datasets as tfds

from exp_03_language_modeling.plot import *
import utils.data
import utils.inference
import utils.metrics


def main():
    exp_dir = 'exp_03_language_modeling'
    data_dir = os.path.join(exp_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    plot_dir = os.path.join(exp_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    reddit_dataset_results = load_reddit_dataset(data_dir=data_dir,
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
            dataset_dir=dataset_dir)

        inference_algs_results_by_dataset[dataset_idx] = dataset_inference_algs_results

    plot_inference_algs_comparison(
        plot_dir=plot_dir,
        inference_algs_results_by_dataset=inference_algs_results_by_dataset,
        sampled_mog_results_by_dataset=sampled_permutation_indices_by_dataset)

    print('Successfully completed Exp 03 Language Modeling')


def load_reddit_dataset(data_dir,
                        plot_dir):
    # possible other alternative datasets:
    #   https://www.tensorflow.org/datasets/catalog/cnn_dailymail
    #   https://www.tensorflow.org/datasets/catalog/newsroom (also in sklearn)

    # useful overview: https://www.tensorflow.org/datasets/overview
    # take only subset of data for speed: https://www.tensorflow.org/datasets/splits
    # specific dataset: https://www.tensorflow.org/datasets/catalog/reddit
    reddit_dataset, reddit_dataset_info = tfds.load(
        'reddit',
        split='train',  # [:1%]',
        shuffle_files=False,
        download=True,
        with_info=True,
        data_dir=data_dir)
    assert isinstance(reddit_dataset, tf.data.Dataset)
    # reddit_dataframe = pd.DataFrame(reddit_dataset.take(10))
    reddit_dataframe = tfds.as_dataframe(
        ds=reddit_dataset.take(20000),
        ds_info=reddit_dataset_info)
    reddit_dataframe = pd.DataFrame(reddit_dataframe)

    true_cluster_labels = reddit_dataframe[['subreddit']].copy()
    documents_text = reddit_dataframe['normalizedBody'].values

    # compute the empirical number of topics per number of posts
    unique_labels = set()
    empiric_num_unique_clusters_by_end_index = []
    for cluster_label in true_cluster_labels['subreddit']:
        unique_labels.add(cluster_label)
        empiric_num_unique_clusters_by_end_index.append(len(unique_labels))
    empiric_num_unique_clusters_by_end_index = np.array(empiric_num_unique_clusters_by_end_index)

    obs_indices = 1+np.arange(len(empiric_num_unique_clusters_by_end_index))

    # fit alpha to the empirical number of topics per number of posts
    def expected_num_tables(customer_idx, alpha):
        return np.multiply(alpha, np.log(1 + customer_idx / alpha))

    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(f=expected_num_tables,
                           xdata=obs_indices,
                           ydata=empiric_num_unique_clusters_by_end_index)
    fitted_alpha = popt[0]

    fit_num_unique_clusters_by_end_index = expected_num_tables(
        customer_idx=obs_indices,
        alpha=fitted_alpha)

    # plot number of topics versus number of posts
    plot_num_clusters_by_num_docs(
        obs_indices=obs_indices,
        empiric_num_unique_clusters_by_end_index=empiric_num_unique_clusters_by_end_index,
        fit_num_unique_clusters_by_end_index=fit_num_unique_clusters_by_end_index,
        fitted_alpha=fitted_alpha,
        plot_dir=plot_dir)

    # convert documents' word counts to tf-idf (Term Frequency times Inverse Document Frequency)
    # equivalent to CountVectorizer() + TfidfTransformer()
    # for more info, see
    # https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
    # https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#sphx-glr-auto-examples-text-plot-document-clustering-py
    tfidf_vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
        max_features=5000,
        sublinear_tf=False)
    observations_tfidf = tfidf_vectorizer.fit_transform(documents_text)

    # quoting from Lin NeurIPS 2013:
    # We pruned the vocabulary to 5000 words by removing stop words and
    # those with low TF-IDF scores, and obtained 150 topics by running LDA [3]
    # on a subset of 20K documents. We held out 10K documents for testing and use the
    # remaining to train the DPMM. We compared SVA,SVA-PM, and TVF.

    # possible likelihoods for TF-IDF data
    # https://stats.stackexchange.com/questions/271923/how-to-use-tfidf-vectors-with-multinomial-naive-bayes
    # https://stackoverflow.com/questions/43237286/how-can-we-use-tfidf-vectors-with-multinomial-naive-bayes
    reddit_dataset_results = dict(
        observations_tfidf=observations_tfidf.toarray(),  # remove .toarray() to keep CSR matrix
        true_cluster_labels=true_cluster_labels,
        tfidf_vectorizer=tfidf_vectorizer,
        feature_names=tfidf_vectorizer.get_feature_names(),
        fitted_alpha=fitted_alpha,
    )

    return reddit_dataset_results


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

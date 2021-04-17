import numpy as np
import os
import pandas as pd
import sklearn.datasets
import sklearn.feature_extraction.text
import tensorflow as tf
import tensorflow_datasets as tfds


def main():
    exp_dir = 'exp_03_language_modeling'
    data_dir = os.path.join(exp_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    plot_dir = os.path.join(exp_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    reddit_dataset_results = load_reddit_dataset(data_dir=data_dir)

    print(10)


def load_reddit_dataset(data_dir):
    # possible other alternative datasets:
    #   https://www.tensorflow.org/datasets/catalog/cnn_dailymail
    #   https://www.tensorflow.org/datasets/catalog/newsroom (also in sklearn)

    # useful tutorial: https://www.tensorflow.org/datasets/overview
    # take only subset of data for speed: https://www.tensorflow.org/datasets/splits
    reddit_dataset, reddit_dataset_info = tfds.load(
        'reddit',
        split='train',  #[:1%]',
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
    num_unique_clusters_by_end_index = []
    for cluster_label in true_cluster_labels['subreddit']:
        unique_labels.add(cluster_label)
        num_unique_clusters_by_end_index.append(len(unique_labels))
    num_unique_clusters_by_end_index = np.array(num_unique_clusters_by_end_index)

    customer_indices = 1+np.arange(len(num_unique_clusters_by_end_index))

    # fit alpha to the empirical number of topics per number of posts
    def expected_num_tables(customer_idx, alpha):
        return alpha * np.log(1 + customer_idx / alpha)

    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(f=expected_num_tables,
                           xdata=customer_indices,
                           ydata=num_unique_clusters_by_end_index)

    # plot number of topics versus number of posts
    import matplotlib.pyplot as plt
    plt.plot(customer_indices,
             num_unique_clusters_by_end_index,
             label='Empiric')
    plt.plot(customer_indices,
             expected_num_tables(customer_idx=customer_indices, alpha=popt[0]),
             label=f'Fit (alpha = {np.round(popt[0], 2)})')
    plt.legend()
    plt.show()

    # convert documents' word counts to tf-idf (Term Frequency times Inverse Document Frequency)
    # equivalent to CountVectorizer() + TfidfTransformer()
    # for more info, see
    # https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
    # https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#sphx-glr-auto-examples-text-plot-document-clustering-py
    tfidf_vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(max_features=5000)
    observations_tfidf = tfidf_vectorizer.fit_transform(documents_text)

    # quoting from Lin NeurIPS 2013:
    # We pruned the vocabulary to 5000 words by removing stop words and
    # those with low TF-IDF scores, and obtained 150 topics by running LDA [3]
    # on a subset of 20K documents
    # We held out 10K documents for testing and use the remaining to train the
    # DPMM. We compared SVA,SVA-PM, and TVF.

    reddit_dataset_results = dict(
        observations_tfidf=observations_tfidf,
        true_cluster_labels=true_cluster_labels,
    )

    return reddit_dataset_results


if __name__ == '__main__':
    main()

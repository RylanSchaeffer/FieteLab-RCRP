import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import sklearn.feature_extraction.text
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import torch.nn.functional
import torch.utils.data
import torchvision


def generate_mixture_of_gaussians(num_gaussians: int = 3,
                                  gaussian_dim: int = 2,
                                  gaussian_mean_prior_cov_scaling: float = 3.,
                                  gaussian_cov_scaling: float = 0.3):
    # sample Gaussians' means from prior = N(0, rho * I)
    means = np.random.multivariate_normal(
        mean=np.zeros(gaussian_dim),
        cov=gaussian_mean_prior_cov_scaling * np.eye(gaussian_dim),
        size=num_gaussians)

    # all Gaussians have same covariance
    cov = gaussian_cov_scaling * np.eye(gaussian_dim)
    covs = np.repeat(cov[np.newaxis, :, :],
                     repeats=num_gaussians,
                     axis=0)

    mixture_of_gaussians = dict(means=means, covs=covs)

    return mixture_of_gaussians


def generate_mixture_of_unigrams(num_topics: int,
                                 vocab_dim: int,
                                 dp_concentration_param: float,
                                 prior_over_topic_parameters: float):

    assert dp_concentration_param > 0
    assert prior_over_topic_parameters > 0

    beta_samples = np.random.beta(1, dp_concentration_param, size=num_topics)
    stick_weights = np.zeros(shape=num_topics, dtype=np.float64)
    remaining_stick = 1.
    for topic_index in range(num_topics):
        stick_weights[topic_index] = beta_samples[topic_index] * remaining_stick
        remaining_stick *= (1. - beta_samples[topic_index])
    # ordinarily, we'd set the last stick weight so that the total stick weights sum to 1.
    # However, floating-point errors can make this last value negative (yeah, I was surprised too)
    # so I only do this if the sum of the weights isn't sufficiently close to 1
    try:
        assert np.allclose(np.sum(stick_weights), 1.)
    except AssertionError:
        stick_weights[-1] = 1 - np.sum(stick_weights[:-1])
    assert np.alltrue(stick_weights >= 0.)
    assert np.allclose(np.sum(stick_weights), 1.)

    # sort stick weights to be descending in magnitude
    stick_weights = np.sort(stick_weights)[::-1]

    topics_parameters = np.zeros((num_topics, vocab_dim))
    for topic_index in range(num_topics):
        gam_t = np.random.gamma(prior_over_topic_parameters, size=vocab_dim)
        topics_parameters[topic_index, :] = gam_t / np.sum(gam_t)

    mixture_of_unigrams = dict(stick_weights=stick_weights,
                               topics_parameters=topics_parameters)

    return mixture_of_unigrams


def load_omniglot_dataset(data_dir='data'):

    # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    omniglot_dataset = torchvision.datasets.Omniglot(
        root=data_dir,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.CenterCrop((80, 80)),
            torchvision.transforms.Lambda(
                lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=9, stride=3)),
            # torchvision.transforms.GaussianBlur(kernel_size=11),
            # torchvision.transforms.Resize((10, 10)),
        ])
    )

    # truncate dataset for now
    # character_classes = [images_and_classes[1] for images_and_classes in
    #                      omniglot_dataset._flat_character_images]
    omniglot_dataset._flat_character_images = omniglot_dataset._flat_character_images[:120]
    dataset_size = len(omniglot_dataset._flat_character_images)

    omniglot_dataloader = torch.utils.data.DataLoader(
        dataset=omniglot_dataset,
        batch_size=1,
        shuffle=True,
    )

    images, labels = [], []
    for image, label in omniglot_dataloader:
        labels.append(label)
        images.append(image[0, 0, :, :])

        # uncomment to deterministically append the first image
        # images.append(omniglot_dataset[0][0][0, :, :])

    images = torch.stack(images).numpy()
    epsilon = 1e-2
    # ensure all values between [epsilon, 1 - epsilon]
    images[images > 1. - epsilon] = 1. - epsilon
    images[images < epsilon] = epsilon

    # these might be swapped but I think height = width for omniglot
    _, image_height, image_width = images.shape
    labels = np.array(labels)

    reshaped_images = np.reshape(images, newshape=(dataset_size, image_height * image_width))
    pca = PCA(n_components=20)
    pca_latents = pca.fit_transform(reshaped_images)
    pca_images = np.reshape(pca.inverse_transform(pca_latents),
                            newshape=(dataset_size, image_height, image_width))
    cum_frac_var_explained = np.sum(pca.explained_variance_ratio_)

    # # visualize images if curious
    # import matplotlib.pyplot as plt
    # for idx in range(10):
    #     plt.imshow(pca_images[idx], cmap='gray')
    #     plt.show()

    omniglot_dataset_results = dict(
        images=images,
        assigned_table_seq=labels,
        pca=pca,
        pca_latents=pca_latents,
        pca_images=pca_images,
        cum_frac_var_explained=cum_frac_var_explained,
    )

    return omniglot_dataset_results


def load_reddit_dataset(data_dir='data'):

    os.makedirs(data_dir, exist_ok=True)

    # possible other alternative datasets:
    #   https://www.tensorflow.org/datasets/catalog/cnn_dailymail
    #   https://www.tensorflow.org/datasets/catalog/newsroom (also in sklearn)

    # useful overview: https://www.tensorflow.org/datasets/overview
    # take only subset of data for speed: https://www.tensorflow.org/datasets/splits
    # specific dataset: https://www.tensorflow.org/datasets/catalog/reddit
    reddit_dataset, reddit_dataset_info = tfds.load(
        'reddit',
        split='train',  # [:1%]',
        shuffle_files=True,
        download=True,
        with_info=True,
        data_dir=data_dir)
    assert isinstance(reddit_dataset, tf.data.Dataset)
    # reddit_dataframe = pd.DataFrame(reddit_dataset.take(10))
    reddit_dataframe = tfds.as_dataframe(
        ds=reddit_dataset.take(200),
        ds_info=reddit_dataset_info)
    reddit_dataframe = pd.DataFrame(reddit_dataframe)

    true_cluster_label_strs = reddit_dataframe['subreddit'].values
    true_cluster_labels = reddit_dataframe['subreddit'].astype('category').cat.codes.values

    documents_text = reddit_dataframe['normalizedBody'].values

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
        true_cluster_label_strs=true_cluster_label_strs,
        true_cluster_labels=true_cluster_labels,
        tfidf_vectorizer=tfidf_vectorizer,
        feature_names=tfidf_vectorizer.get_feature_names(),
    )

    return reddit_dataset_results


def sample_sequence_from_crp(T: int,
                             alpha: float):
    assert alpha > 0
    table_occupancies = np.zeros(shape=T, dtype=np.int)
    customer_tables = np.zeros(shape=T, dtype=np.int)

    # the first customer always goes at the first table
    table_occupancies[0] = 1

    for t in range(1, T):
        max_k = np.argmin(table_occupancies)  # first zero index
        freq = table_occupancies.copy()
        freq[max_k] = alpha
        probs = freq / np.sum(freq)
        z_t = np.random.choice(np.arange(max_k + 1), p=probs[:max_k + 1])
        customer_tables[t] = z_t
        table_occupancies[z_t] += 1

    customer_tables_one_hot = np.zeros(shape=(customer_tables.shape[0], customer_tables.shape[0]))
    customer_tables_one_hot[np.arange(customer_tables.shape[0]), customer_tables] = 1

    return table_occupancies, customer_tables, customer_tables_one_hot


vectorized_sample_sequence_from_crp = np.vectorize(
    sample_sequence_from_crp,
    otypes=[np.ndarray, np.ndarray, np.ndarray])


def sample_sequence_from_ibp(T: int,
                             alpha: float):

    # shape: (number of customers, number of dishes)
    # heuristic: 10 * expected number
    max_dishes = int(10 * alpha * np.sum(1 / (1 + np.arange(T))))
    customers_dishes_draw = np.zeros(shape=(T, max_dishes), dtype=np.int)

    current_num_dishes = 0
    for t in range(T):

        # sample old dishes for new customer
        frac_prev_customers_sampling_dish = np.sum(customers_dishes_draw[:t, :], axis=0) / (t + 1)
        dishes_for_new_customer = np.random.binomial(n=1, p=frac_prev_customers_sampling_dish[np.newaxis, :])[0]
        customers_dishes_draw[t, :] = dishes_for_new_customer.astype(np.int)

        # sample number of new dishes for new customer
        # add +1 to t because of 0-based indexing
        num_new_dishes = np.random.poisson(alpha / (t + 1))
        customers_dishes_draw[t, current_num_dishes:current_num_dishes + num_new_dishes] = 1

        # increment current num dishes
        current_num_dishes += num_new_dishes

    return customers_dishes_draw


vectorized_sample_sequence_from_ibp = np.vectorize(sample_sequence_from_ibp,
                                                   otypes=[np.ndarray])


def sample_sequence_from_mixture_of_gaussians(seq_len: int = 100,
                                              class_sampling: str = 'Uniform',
                                              alpha: float = None,
                                              num_gaussians: int = None,
                                              gaussian_params: dict = {}):
    """
    Draw sample from mixture of Gaussians, using either uniform sampling or
    CRP sampling.

    Exactly one of alpha and num_gaussians must be specified.

    :param seq_len: desired sequence length
    :param class_sampling:
    :param alpha:
    :param num_gaussians:
    :param gaussian_params:
    :return:
        assigned_table_seq: NumPy array with shape (seq_len,) of (integer) sampled classes
        gaussian_samples_seq: NumPy array with shape (seq_len, dim of Gaussian) of
                                Gaussian samples
    """

    if class_sampling == 'Uniform':
        assert num_gaussians is not None
        assigned_table_seq = np.random.choice(np.arange(num_gaussians),
                                              replace=True,
                                              size=seq_len)

    elif class_sampling == 'DP':
        assert alpha is not None
        table_counts, assigned_table_seq = sample_sequence_from_crp(T=seq_len,
                                                                    alpha=alpha)
        num_gaussians = np.max(assigned_table_seq) + 1
    else:
        raise ValueError(f'Impermissible class sampling: {class_sampling}')

    mixture_of_gaussians = generate_mixture_of_gaussians(
        num_gaussians=num_gaussians,
        **gaussian_params)

    # create sequence of Gaussian samples from (mean_t, cov_t)
    gaussian_samples_seq = np.array([
        np.random.multivariate_normal(mean=mixture_of_gaussians['means'][assigned_table],
                                      cov=mixture_of_gaussians['covs'][assigned_table])
        for assigned_table in assigned_table_seq])

    # convert assigned table sequence to one-hot codes
    assigned_table_seq_one_hot = np.zeros((seq_len, seq_len))
    assigned_table_seq_one_hot[np.arange(seq_len), assigned_table_seq] = 1.

    result = dict(
        mixture_of_gaussians=mixture_of_gaussians,
        assigned_table_seq=assigned_table_seq,
        assigned_table_seq_one_hot=assigned_table_seq_one_hot,
        gaussian_samples_seq=gaussian_samples_seq
    )

    return result


def sample_sequence_from_mixture_of_unigrams(seq_len: int = 450,
                                             num_topics: int = 10,
                                             vocab_dim: int = 8,
                                             doc_len: int = 120,
                                             unigram_params: dict = {}):

    mixture_of_unigrams = generate_mixture_of_unigrams(
        num_topics=num_topics,
        vocab_dim=vocab_dim,
        **unigram_params)

    # create sequence of Multinomial samples from (num_topics, vocab_dim)
    # draw a Sample of num_docs documents each of size doc_len
    num_topics_in_corpus = 0
    # rejection sample till we get dataset with correct number of topics
    num_samples = 0
    while num_topics_in_corpus != num_topics:
        assigned_table_seq_one_hot = np.random.multinomial(
            n=1,
            pvals=mixture_of_unigrams['stick_weights'],
            size=seq_len)
        # convert assigned table sequence to one-hot codes
        assigned_table_seq = np.argmax(assigned_table_seq_one_hot, axis=1)
        num_topics_in_corpus = len(np.unique(assigned_table_seq))
        num_samples += 1
        # print(f'Num of corpus samples: {num_samples}')

    doc_samples_seq = np.zeros(shape=(seq_len, vocab_dim))
    for doc_idx in range(seq_len):
        topic_idx = assigned_table_seq[doc_idx]
        doc = np.random.multinomial(n=doc_len,
                                    pvals=mixture_of_unigrams['topics_parameters'][topic_idx, :],
                                    size=1)
        doc_samples_seq[doc_idx, :] = doc

    result = dict(
        mixture_of_unigrams=mixture_of_unigrams,
        assigned_table_seq=assigned_table_seq,
        assigned_table_seq_one_hot=assigned_table_seq_one_hot,
        doc_samples_seq=doc_samples_seq,
    )

    return result

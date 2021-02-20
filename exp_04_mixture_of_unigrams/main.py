import numpy as np
import os
import sklearn.datasets

from exp_04_mixture_of_unigrams.plot import *

from utils.data import sample_sequence_from_mixture_of_unigrams
from utils.helpers import assert_no_nan_no_inf
from utils.inference import bayesian_recursion, dp_means_online, dp_means_offline, nuts_sampling, variational_bayes
from utils.metrics import score_predicted_clusters


def main():
    plot_dir = 'exp_04_mixture_of_unigrams/plots'
    os.makedirs(plot_dir, exist_ok=True)
    np.random.seed(1)

    # sample data
    sampled_mou_results = sample_sequence_from_mixture_of_unigrams(
        seq_len=450,
        unigram_params=dict(dp_concentration_param=5.7,
                            prior_over_topic_parameters=0.3))

    plot_sample_from_mixture_of_unigrams(assigned_table_seq=sampled_mou_results['assigned_table_seq'],
                                         mixture_of_unigrams=sampled_mou_results['mixture_of_unigrams'],
                                         doc_samples_seq=sampled_mou_results['doc_samples_seq'],
                                         plot_dir=plot_dir)

    # plot data



if __name__ == '__main__':
    main()

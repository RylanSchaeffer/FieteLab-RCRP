import numpy as np
import os
import sklearn.datasets

from utils.helpers import assert_no_nan_no_inf
from utils.inference import bayesian_recursion, dp_means_online, dp_means_offline, nuts_sampling, variational_bayes
from utils.metrics import score_predicted_clusters


def main():
    plot_dir = 'exp_03_uci_datasets/plots'
    os.makedirs(plot_dir, exist_ok=True)
    np.random.seed(1)

    # sample data

    # plot data





if __name__ == '__main__':
    main()

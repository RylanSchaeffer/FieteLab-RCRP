import numpy as np
import os
import sklearn.datasets

from utils.helpers import assert_no_nan_no_inf
from utils.inference import bayesian_recursion, dp_means_online, dp_means_offline, nuts_sampling, variational_bayes
from utils.metrics import score_predicted_clusters



# https://scikit-learn.org/stable/datasets/toy_dataset.html
def main():
    plot_dir = 'exp_03_uci_datasets/plots'
    os.makedirs(plot_dir, exist_ok=True)
    np.random.seed(1)

    dataset_load_functions = {
        'Iris': sklearn.datasets.load_iris,
        'Diabetes': sklearn.datasets.load_diabetes,
        'Wine': sklearn.datasets.load_wine,
        'Breast Cancer': sklearn.datasets.load_breast_cancer,
#        'California Housing': sklearn.datasets.fetch_california_housing,
    }

    for dataset_str, dataset_load_function in dataset_load_functions.items():
        dataset_plot_dir = os.path.join(plot_dir, dataset_str)
        os.makedirs(dataset_plot_dir, exist_ok=True)
        dataset = dataset_load_function(as_frame=True)
        features, targets = dataset.data, dataset.target

        print(10)





if __name__ == '__main__':
    main()

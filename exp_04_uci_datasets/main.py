import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import sklearn.datasets
import torchvision
from torchvision.datasets import Omniglot

from utils.helpers import assert_numpy_no_nan_no_inf
from utils.inference_mix_of_gauss import bayesian_recursion, dp_means_online, dp_means_offline, sampling_hmc_gibbs, variational_bayes
from utils.metrics import score_predicted_clusters


# https://scikit-learn.org/stable/datasets/toy_dataset.html
def main():
    data_dir = 'exp_03_uci_datasets/data'

    dataset = torchvision.datasets.Omniglot(
        root=data_dir, download=True,
        transform=torchvision.transforms.ToTensor()
    )
    image, label = dataset[0]
    import matplotlib.pyplot as plt
    plt.imshow(image[0].numpy())
    plt.show()

    plot_dir = 'exp_03_uci_datasets/plots'
    os.makedirs(plot_dir, exist_ok=True)
    np.random.seed(1)

    dataset_load_functions = {
        # 'Iris': sklearn.datasets.load_iris,
        'Diabetes': sklearn.datasets.load_diabetes,
        'Wine': sklearn.datasets.load_wine,
        'Breast Cancer': sklearn.datasets.load_breast_cancer,
    }

    # real world datasets
    # https://scikit-learn.org/stable/datasets/real_world.html#olivetti-faces-dataset
    dataset_load_functions = {
        'Anomaly Detection': sklearn.datasets.fetch_kddcup99,  # 41 dimensions,
        'California Housing': sklearn.datasets.fetch_california_housing, # https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html
        'Olivetti Faces': sklearn.datasets.fetch_olivetti_faces,  # 10 images of 40 distinct subjects, 4096 features
        'Famous Faces': sklearn.datasets.fetch_lfw_people, # 5749 classes, 13233 samples, dim 5828
        'Forest Cover': sklearn.datasets.fetch_covtype,  # 54 dimensional features, mixed type
        'Reuters Dataset': sklearn.datasets.fetch_rcv1,
    }

    for dataset_str, dataset_load_function in dataset_load_functions.items():
        dataset_plot_dir = os.path.join(plot_dir, dataset_str)
        os.makedirs(dataset_plot_dir, exist_ok=True)
        dataset = dataset_load_function(as_frame=True, data_home=data_dir)
        features, targets = dataset.data, dataset.target
        # plt.hist(targets, density=True)
        g = sns.countplot(y=targets, logscale=True)
        g.set_yscale("log")
        plt.show()
        print(10)


if __name__ == '__main__':
    main()

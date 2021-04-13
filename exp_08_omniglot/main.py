import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional
import torch.utils.data
import torchvision

from exp_08_omniglot.plot import plot_inference_results, plot_inference_algs_comparison

from utils.inference_mix_of_cont_bernoullis import bayesian_recursion
from utils.metrics import score_predicted_clusters


def main():
    exp_dir = 'exp_08_omniglot'

    plot_dir = os.path.join(exp_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    np.random.seed(1)
    torch.manual_seed(0)

    images, labels = load_omniglot_dataset(exp_dir)

    num_obs = labels.shape[0]
    num_permutations = 5
    inference_algs_results_by_dataset = {}
    sampled_permutation_indices_by_dataset = {}
    for dataset_idx in range(num_permutations):

        dataset_dir = os.path.join(plot_dir, f'dataset={dataset_idx}')

        # generate permutation and reorder data
        index_permutation = np.random.permutation(np.arange(num_obs, dtype=np.int))
        sampled_permutation_indices_by_dataset[dataset_idx] = index_permutation
        images = images[index_permutation]
        labels = labels[index_permutation]

        dataset_results_path = os.path.join(dataset_dir, 'dataset_results.joblib')
        if os.path.isfile(dataset_results_path):
            # load from disk if exists
            dataset_results = joblib.load(dataset_results_path)
        else:
            # otherwise, generate anew
            dataset_inference_algs_results = run_one_dataset(
                images=images,
                labels=labels,
                plot_dir=dataset_dir)
            dataset_results = dict(
                dataset_inference_algs_results=dataset_inference_algs_results,
            )
            joblib.dump(dataset_results, dataset_results_path)

            # delete variables from memory and perform fresh read from disk
            del dataset_inference_algs_results
            del dataset_results
            dataset_results = joblib.load(dataset_results_path)

        inference_algs_results_by_dataset[dataset_idx] = dataset_results['dataset_inference_algs_results']

    plot_inference_algs_comparison(
        images=images,
        labels=labels,
        plot_dir=plot_dir,
        inference_algs_results_by_dataset=inference_algs_results_by_dataset,
        sampled_permutation_indices_by_dataset=sampled_permutation_indices_by_dataset)

    print('Successfully completed Exp 08 Omniglot')


def load_omniglot_dataset(exp_dir):

    data_dir = os.path.join(exp_dir, 'data')

    # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    omniglot_dataset = torchvision.datasets.Omniglot(
        root=data_dir,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.CenterCrop((70, 70)),
            torchvision.transforms.Lambda(
                lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=11, stride=7)),
            # torchvision.transforms.GaussianBlur(kernel_size=11),
            # torchvision.transforms.Resize((10, 10)),
        ])
    )

    # truncate dataset for now
    # character_classes = [images_and_classes[1] for images_and_classes in
    #                      omniglot_dataset._flat_character_images]
    omniglot_dataset._flat_character_images = omniglot_dataset._flat_character_images[:11]

    omniglot_dataloader = torch.utils.data.DataLoader(
        dataset=omniglot_dataset,
        batch_size=1,
        shuffle=False)

    images, labels = [], []
    for image, label in omniglot_dataloader:
        # visualize first image, if curious
        plt.imshow(image[0, 0, :, :].numpy(), cmap='gray')
        plt.show()
        # images.append(image[0, :, :])
        images.append(image[0, 0, :, :])
        labels.append(label)

    images = torch.stack(images).numpy()
    epsilon = 1e-2
    # ensure all values between [epsilon, 1 - epsilon]
    images[images > 1. - epsilon] = 1. - epsilon
    images[images < epsilon] = epsilon
    labels = np.array(labels)

    return images, labels


def run_one_dataset(images,
                    labels,
                    plot_dir):

    bayesian_recursion_results = run_and_plot_bayesian_recursion(
        images=images,
        labels=labels,
        plot_dir=plot_dir)

    inference_algs_results = {
        'Bayesian Recursion': bayesian_recursion_results,
        # 'HMC-Gibbs (5k Samples)': hmc_gibbs_5000_samples_results,
        # 'HMC-Gibbs (20k Samples)': hmc_gibbs_20000_samples_results,
        # 'DP-Means (Online)': dp_means_online_results,
        # 'DP-Means (Offline)': dp_means_offline_results,
        # 'Variational Bayes': variational_bayes_results,
    }

    return inference_algs_results


def run_and_plot_bayesian_recursion(images,
                                    labels,
                                    plot_dir):

    alphas = np.arange(0.1, 50.01, 5.)
    bayesian_recursion_plot_dir = os.path.join(plot_dir, 'bayesian_recursion')
    os.makedirs(bayesian_recursion_plot_dir, exist_ok=True)
    num_clusters_by_alpha = {}
    scores_by_alpha = {}
    for alpha in alphas:

        bayesian_recursion_results = bayesian_recursion(
            observations=images.reshape(images.shape[0], -1),
            alpha=alpha)

        # record scores
        scores, pred_cluster_labels = score_predicted_clusters(
            true_cluster_labels=labels,
            table_assignment_posteriors=bayesian_recursion_results['table_assignment_posteriors'])
        scores_by_alpha[alpha] = scores

        # count number of clusters
        num_clusters_by_alpha[alpha] = len(np.unique(pred_cluster_labels))

        plot_inference_results(
            images=images,
            labels=labels,
            inference_results=bayesian_recursion_results,
            inference_alg='bayesian_recursion_alpha={:.2f}'.format(alpha),
            plot_dir=bayesian_recursion_plot_dir)

        print('Finished Bayesian recursion alpha={:.2f}'.format(alpha))

    bayesian_recursion_results = dict(
        num_clusters_by_param=num_clusters_by_alpha,
        scores_by_param=pd.DataFrame(scores_by_alpha).T,
    )

    return bayesian_recursion_results



if __name__ == '__main__':
    main()

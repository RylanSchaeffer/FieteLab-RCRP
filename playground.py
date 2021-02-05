import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os.path

from utils.run import load_env_data


def run_envs():
    env_strs = [
        'hairpin',
        'linear_track',
        'organic_1',
        'organic_2',
        'spiral',
        'spiral_square',
        'square_room',
        'two_rooms_2',
    ]

    for env_str in env_strs:

        load_env_data_results = load_env_data(env_str=env_str)
        num_clusters = load_env_data_results['num_clusters']
        num_positions = load_env_data_results['num_positions']
        obs_vectors = load_env_data_results['obs_vectors']
        num_parameters = load_env_data_results['num_parameters']

        # infer theta_1 with single cluster
        mle = np.mean(obs_vectors, axis=1)

        # plot histogram of single cluster parameters
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.set_title(r'$\theta_1$')
        ax.hist(mle, bins=num_positions // 100)
        plt.show()

        # Infer theta_k with K clusters using expectation maximization
        # Let   k = 1, ..., K denote the number of clusters
        #       n = 1, ..., N denote the number of observations
        #       d = 1, ..., D denote the dimensionality of the observations
        #       theta_k \in R^D denote the parameters of the kth cluster
        #
        # The generative model is
        #       C_k ~ CRP(alpha)
        #       theta_k | C_k ~ Beta(a, b)^D
        #       obs_{n, d} | C_k ~ Bern(\theta_{kd})

        alpha = 0.5
        # prior over clusters: first dimension = data index, second dimension = cluster index
        C_prior = np.zeros(shape=(num_positions, num_positions))
        C_prior_nonnormalized = np.zeros_like(C_prior)
        for i in range(num_positions):
            if i > 0:
                C_prior_nonnormalized[i, :] = C_prior_nonnormalized[i-1, :] + C_prior[i-1, :]
            C_prior_nonnormalized[i, i] = alpha
            C_prior[i, :] = C_prior_nonnormalized[i, :] / np.sum(C_prior_nonnormalized[i, :])

        # check correctness of prior over cluster
        fig, ax = plt.subplots(nrows=1, ncols=1)
        mask = np.tri(C_prior.shape[0], k=-1).T
        masked_C_prior = np.ma.array(C_prior, mask=mask)  # mask out the upper triangle
        cax = ax.matshow(masked_C_prior, norm=LogNorm(vmin=np.min(masked_C_prior), vmax=np.max(masked_C_prior)))
        fig.colorbar(cax)
        plt.show()

        # prior over cluster parameters
        thetas = np.random.beta(
            a=1,
            b=5,
            size=(num_clusters, num_parameters)).astype(np.float32)

        print(10)


if __name__ == '__main__':
    run_envs()

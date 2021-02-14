import numpy as np
import os

from utils.data import sample_sequence_from_mixture_of_gaussians
from utils.inference import bayesian_recursion
from exp_01_mixture_of_gaussians.plot import plot_sample_from_mixture_of_gaussians


def main():
    plot_dir = 'exp_01_mixture_of_gaussians/plots'
    os.makedirs(plot_dir, exist_ok=True)

    np.random.seed(1)

    # sample data
    alpha = 1.5
    sampled_mog_results = sample_sequence_from_mixture_of_gaussians(
        seq_len=100,
        class_sampling='CRP',
        alpha=alpha,
    )

    # plot_sample_from_mixture_of_gaussians(
    #     assigned_table_seq=sampled_mog_results['assigned_table_seq'],
    #     gaussian_samples_seq=sampled_mog_results['gaussian_samples_seq'],
    #     plot_dir=plot_dir)

    def likelihood_fn(observation, parameters):
        # create new mean for new table, centered at that point
        parameters['means'] = np.vstack([parameters['means'], observation[np.newaxis, :]])
        obs_dim = parameters['means'].shape[1]
        parameters['covs'] = np.vstack([parameters['covs'], np.eye(obs_dim)[np.newaxis, :, :]])
        covariance_determinants = np.linalg.det(2*np.pi*parameters['covs'])
        normalizing_constant = np.sqrt(np.power(2*np.pi, obs_dim/2)) * np.sqrt(covariance_determinants)
        # shape (num gaussians, dim of gaussians)
        diff = (observation - parameters['means'])
        quadratic = np.einsum(
            'bi, bij, bj->b',
            diff,
            np.linalg.inv(parameters['covs']),
            diff
        )
        likelihoods = np.exp(-0.5 * quadratic) / normalizing_constant
        return likelihoods, parameters

    def update_parameters_fn(observation, latent_prior, latent_posterior, parameters):
        # create a copy of observation for each possible cluster
        stacked_observation = np.repeat(observation[np.newaxis, :],
                                        repeats=len(latent_posterior),
                                        axis=0)

        # weigh the observation by the filtered posterior p(z_t|o_{<=t})
        # then scale the
        weighted_stacked_observation = np.multiply(stacked_observation,
                                                   latent_posterior[:, np.newaxis])
        weighted_stacked_observation = np.divide(weighted_stacked_observation,
                                                 latent_prior[:, np.newaxis])

        # parameters['means'] = parameters['means'] + weighted_stacked_observation

        return parameters


    # inference
    bayesian_recursion(observations=sampled_mog_results['gaussian_samples_seq'],
                       alpha=alpha,
                       likelihood_fn=likelihood_fn,
                       update_parameters_fn=update_parameters_fn)


if __name__ == '__main__':
    main()

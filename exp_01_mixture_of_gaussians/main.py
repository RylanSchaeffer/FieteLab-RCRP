import numpy as np
import os

from utils.data import sample_sequence_from_mixture_of_gaussians
from utils.helpers import assert_no_nan_no_inf
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

        # calculate likelihood under each cluster mean
        covariance_determinants = np.linalg.det(parameters['covs'])
        normalizing_constant = np.sqrt(np.power(2 * np.pi, obs_dim) * covariance_determinants)
        # shape (num gaussians, dim of gaussians)
        diff = (observation - parameters['means'])
        quadratic = np.einsum(
            'bi, bij, bj->b',
            diff,
            np.linalg.inv(parameters['covs']),
            diff
        )
        likelihoods = np.exp(-0.5 * quadratic) / normalizing_constant
        assert np.all(~np.isnan(likelihoods))

        return likelihoods, parameters

    def update_parameters_fn(observation,
                             table_assignment_posteriors_running_sum,
                             table_assignment_posterior,
                             parameters):
        # the strategy here is to update parameters as a moving average, but instead of dividing
        # by the number of points assigned to each cluster, we divide by the total probability
        # mass assigned to each cluster

        # create a copy of observation for each possible cluster
        stacked_observation = np.repeat(observation[np.newaxis, :],
                                        repeats=len(table_assignment_posteriors_running_sum),
                                        axis=0)

        # compute online average of clusters' means
        # instead of typical dynamics:
        #       m_k <- m_k + (obs - m_k) / number of obs assigned to kth cluster
        # we use the new dynamics
        #       m_k <- m_k + posterior(obs belongs to kth cluster) * (obs - m_k) / total mass on kth cluster
        # floating point errors are common here!
        prefactor = np.divide(table_assignment_posterior,
                              table_assignment_posteriors_running_sum)
        prefactor[np.isnan(prefactor)] = 0.
        assert_no_nan_no_inf(prefactor)

        diff = stacked_observation - parameters['means']
        assert_no_nan_no_inf(diff)
        means_updates = np.multiply(
            prefactor[:, np.newaxis],
            diff)
        parameters['means'] += means_updates
        assert_no_nan_no_inf(parameters['means'])

        return parameters

    bayesian_recursion_results = bayesian_recursion(
        observations=sampled_mog_results['gaussian_samples_seq'],
        alpha=alpha,
        likelihood_fn=likelihood_fn,
        update_parameters_fn=update_parameters_fn)

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(nrows=1,
                             ncols=4,
                             figsize=(16, 4))

    ax_idx = 0
    ax = axes[ax_idx]
    ax.scatter(sampled_mog_results['gaussian_samples_seq'][:, 0],
               sampled_mog_results['gaussian_samples_seq'][:, 1],
               c=sampled_mog_results['assigned_table_seq'])
    ax.set_title('Ground Truth Data')

    ax_idx += 1
    ax = axes[ax_idx]
    sns.heatmap(bayesian_recursion_results['table_assignment_priors'][:, :10],
                ax=ax,
                cmap='Blues')
    ax.set_title(r'$P(z_t|o_{<t})$')
    ax.set_ylabel('Customer Number')
    ax.set_xlabel('Table Number')

    ax_idx += 1
    ax = axes[ax_idx]
    sns.heatmap(bayesian_recursion_results['table_assignment_posteriors'][:, :10],
                ax=ax,
                cmap='Blues')
    ax.set_title(r'$P(z_t|o_{\leq t})$')
    ax.set_ylabel('Customer Number')
    ax.set_xlabel('Table Number')

    ax_idx += 1
    ax = axes[ax_idx]
    ax.scatter(bayesian_recursion_results['parameters']['means'][:, 0],
               bayesian_recursion_results['parameters']['means'][:, 1],
               s=bayesian_recursion_results['table_assignment_posteriors_running_sum'][-1, :])
    ax.set_title(r'Cluster Centroids $\mu_z$ (Scaled by Total Mass)')
    plt.show()
    print(10)


if __name__ == '__main__':
    main()

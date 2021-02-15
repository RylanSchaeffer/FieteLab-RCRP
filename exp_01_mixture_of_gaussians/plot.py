import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns


def plot_sample_from_mixture_of_gaussians(assigned_table_seq,
                                          gaussian_samples_seq,
                                          plot_dir):
    plt.scatter(x=gaussian_samples_seq[:, 0],
                y=gaussian_samples_seq[:, 1],
                c=assigned_table_seq)
    plt.savefig(os.path.join(plot_dir, 'sample_from_mixture_of_gaussians.png'))
    plt.show()


def plot_inference_results(sampled_mog_results: dict,
                           inference_results: dict,
                           inference_alg: str,
                           plot_dir,
                           num_tables_to_plot: int = 10):

    assert isinstance(num_tables_to_plot, int)
    assert num_tables_to_plot > 0
    # num_obs = sampled_mog_results['gaussian_samples_seq'].shape[0]
    # yticklabels = np.arange(num_obs)
    # indices_to_keep = yticklabels % 10 == 0
    # yticklabels += 1
    # yticklabels = yticklabels.astype(np.str)
    # yticklabels[~indices_to_keep] = ''

    xmin = 1.1 * np.min(sampled_mog_results['gaussian_samples_seq'][:, 0])
    xmax = 1.1 * np.max(sampled_mog_results['gaussian_samples_seq'][:, 0])
    ymin = 1.1 * np.min(sampled_mog_results['gaussian_samples_seq'][:, 1])
    ymax = 1.1 * np.max(sampled_mog_results['gaussian_samples_seq'][:, 1])

    fig, axes = plt.subplots(nrows=1,
                             ncols=4,
                             figsize=(16, 4))

    ax_idx = 0
    ax = axes[ax_idx]
    ax.scatter(sampled_mog_results['gaussian_samples_seq'][:, 0],
               sampled_mog_results['gaussian_samples_seq'][:, 1],
               c=sampled_mog_results['assigned_table_seq'])
    ax.set_xlim(xmin=xmin, xmax=xmax)
    ax.set_ylim(ymin=ymin, ymax=ymax)
    ax.set_title('Ground Truth Data')

    ax_idx += 1
    ax = axes[ax_idx]
    if 'table_assignment_priors' in inference_results:
        sns.heatmap(inference_results['table_assignment_priors'][:, :num_tables_to_plot],
                    ax=ax,
                    cmap='Blues',
                    xticklabels=1+np.arange(num_tables_to_plot),
                    # yticklabels=yticklabels
                    mask=np.isnan(inference_results['table_assignment_priors'][:, :num_tables_to_plot]),
                    vmin=0.,
                    vmax=1.,
                    )
        ax.set_title(r'$P(z_t|o_{<t})$')
        ax.set_ylabel('Observation Index')
        ax.set_xlabel('Cluster Index')

    ax_idx += 1
    ax = axes[ax_idx]
    sns.heatmap(inference_results['table_assignment_posteriors'][:, :num_tables_to_plot],
                ax=ax,
                cmap='Blues',
                xticklabels=1+np.arange(num_tables_to_plot),
                # yticklabels=yticklabels
                vmin=0.,
                vmax=1.
                )
    ax.set_title(r'$P(z_t|o_{\leq t})$')
    ax.set_ylabel('Observation Index')
    ax.set_xlabel('Cluster Index')

    ax_idx += 1
    ax = axes[ax_idx]
    ax.scatter(inference_results['parameters']['means'][:, 0],
               inference_results['parameters']['means'][:, 1],
               s=inference_results['table_assignment_posteriors_running_sum'][-1, :])
    ax.set_xlim(xmin=xmin, xmax=xmax)
    ax.set_ylim(ymin=ymin, ymax=ymax)
    ax.set_title(r'Cluster Centroids $\mu_z$ (Scaled by Total Mass)')

    plt.savefig(os.path.join(plot_dir, f'{inference_alg}_results.png'))
    plt.show()

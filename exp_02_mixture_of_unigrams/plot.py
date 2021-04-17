import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

# common plotting functions
from utils.plot import *


def plot_inference_results(sampled_mou_results: dict,
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

    xmin = 1.1 * np.min(sampled_mou_results['gaussian_samples_seq'][:, 0])
    xmax = 1.1 * np.max(sampled_mou_results['gaussian_samples_seq'][:, 0])
    ymin = 1.1 * np.min(sampled_mou_results['gaussian_samples_seq'][:, 1])
    ymax = 1.1 * np.max(sampled_mou_results['gaussian_samples_seq'][:, 1])

    fig, axes = plt.subplots(nrows=1,
                             ncols=3,
                             figsize=(12, 4))

    ax_idx = 0
    # plot ground truth data
    ax = axes[ax_idx]
    ax.scatter(sampled_mou_results['gaussian_samples_seq'][:, 0],
               sampled_mou_results['gaussian_samples_seq'][:, 1],
               c=sampled_mou_results['assigned_table_seq'])
    ax.set_xlim(xmin=xmin, xmax=xmax)
    ax.set_ylim(ymin=ymin, ymax=ymax)
    ax.set_title('Ground Truth Data')

    # plot cluster centroids
    ax_idx += 1
    ax = axes[ax_idx]
    ax.scatter(inference_results['parameters']['means'][:, 0],
               inference_results['parameters']['means'][:, 1],
               s=2 * inference_results['table_assignment_posteriors_running_sum'][-1, :],
               facecolors='none',
               edgecolors='k')
    ax.set_xlim(xmin=xmin, xmax=xmax)
    ax.set_ylim(ymin=ymin, ymax=ymax)
    ax.set_title(r'Cluster Centroids $\mu_z$')

    # plot predicted cluster labels
    ax_idx += 1
    ax = axes[ax_idx]
    pred_cluster_labels = np.argmax(inference_results['table_assignment_posteriors'],
                                    axis=1)
    ax.scatter(sampled_mou_results['gaussian_samples_seq'][:, 0],
               sampled_mou_results['gaussian_samples_seq'][:, 1],
               c=pred_cluster_labels)
    ax.set_xlim(xmin=xmin, xmax=xmax)
    ax.set_ylim(ymin=ymin, ymax=ymax)
    ax.set_title(r'Predicted Cluster Labels')

    plt.savefig(os.path.join(plot_dir, f'{inference_alg}_pred_clusters.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()

    fig, axes = plt.subplots(nrows=1,
                             ncols=2,
                             figsize=(8, 4))

    ax_idx = 0
    # plot prior table assignments
    ax = axes[ax_idx]
    if 'table_assignment_priors' in inference_results:
        sns.heatmap(inference_results['table_assignment_priors'][:, :num_tables_to_plot],
                    ax=ax,
                    cmap='Blues',
                    xticklabels=1 + np.arange(num_tables_to_plot),
                    # yticklabels=yticklabels
                    mask=np.isnan(inference_results['table_assignment_priors'][:, :num_tables_to_plot]),
                    vmin=0.,
                    vmax=1.,
                    )
        ax.set_title(r'$P(z_t|o_{<t})$')
        ax.set_ylabel('Observation Index')
        ax.set_xlabel('Cluster Index')

    # plot posterior table assignments
    ax_idx += 1
    ax = axes[ax_idx]
    sns.heatmap(inference_results['table_assignment_posteriors'][:, :num_tables_to_plot],
                ax=ax,
                cmap='Blues',
                xticklabels=1 + np.arange(num_tables_to_plot),
                # yticklabels=yticklabels
                vmin=0.,
                vmax=1.
                )
    ax.set_title(r'$P(z_t|o_{\leq t})$')
    ax.set_ylabel('Observation Index')
    ax.set_xlabel('Cluster Index')

    plt.savefig(os.path.join(plot_dir, f'{inference_alg}_pred_assignments.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_sample_from_mixture_of_unigrams(assigned_table_seq_one_hot,
                                         mixture_of_unigrams,
                                         doc_samples_seq,
                                         plot_dir):

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                             gridspec_kw={"width_ratios": [0.2, 1]}
                             )
    ax = axes[0]
    sns.heatmap(data=mixture_of_unigrams['stick_weights'][:, np.newaxis],
                ax=ax,
                # cmap='jet_r',
                vmin=0.,
                vmax=1.,
                annot=True,
                cbar=False,
                # cbar_kws=dict(label='P(Topic)'),
                )
    ax.set_title('P(Topic)')
    ax.set_ylabel('Topic Index')

    # axes[1].axis('off')

    ax = axes[1]
    sns.heatmap(data=mixture_of_unigrams['topics_parameters'].T,
                ax=ax,
                # cmap='jet_r',
                # annot=True,
                vmin=0.,
                vmax=1.,
                # cbar_kws=dict(label='P(Word|Topic)'),
                )
    ax.set_title('P(Word|Topic)')
    # ax.set_ylabel('Topic Index')
    ax.set_xlabel('Word Index')
    plt.savefig(os.path.join(plot_dir, 'sample_from_mixture_of_unigrams_topics.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                             sharey=True,
                             gridspec_kw={"width_ratios": [0.8, 1]})
    ax = axes[0]
    ax.spy(assigned_table_seq_one_hot, aspect='auto')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_ylabel('Document Index')
    ax.set_xlabel('True Topic Index')
    ax = axes[1]
    sns.heatmap(data=doc_samples_seq, ax=ax,
                cbar_kws=dict(label='Words\' Counts in Doc'))
    ax.set_xlabel('Word Index')
    plt.savefig(os.path.join(plot_dir, 'sample_from_mixture_of_unigrams_docs.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()

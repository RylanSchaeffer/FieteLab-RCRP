import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns


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
                             ncols=3,
                             figsize=(12, 4))

    ax_idx = 0
    # plot ground truth data
    ax = axes[ax_idx]
    ax.scatter(sampled_mog_results['gaussian_samples_seq'][:, 0],
               sampled_mog_results['gaussian_samples_seq'][:, 1],
               c=sampled_mog_results['assigned_table_seq'])
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
    ax.scatter(sampled_mog_results['gaussian_samples_seq'][:, 0],
               sampled_mog_results['gaussian_samples_seq'][:, 1],
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


def plot_inference_algs_comparison(inference_algs_results_by_dataset: dict,
                                   sampled_mog_results_by_dataset: dict,
                                   plot_dir: str):
    num_datasets = len(inference_algs_results_by_dataset)
    num_clusters = len(np.unique(sampled_mog_results_by_dataset[0]['assigned_table_seq']))

    inference_algs = list(inference_algs_results_by_dataset[0].keys())
    scoring_metrics = inference_algs_results_by_dataset[0][inference_algs[0]]['scores_by_param'].columns.values

    # we have four dimensions of interest: inference_alg, dataset idx, scoring metric, concentration parameter

    # construct dictionary mapping from inference alg to dataframe
    # with dataset idx as rows and concentration parameters as columns
    # {inference alg: DataFrame(number of clusters)}
    num_clusters_by_dataset_by_inference_alg = {}
    for inference_alg in inference_algs:
        num_clusters_by_dataset_by_inference_alg[inference_alg] = pd.DataFrame([
            inference_algs_results_by_dataset[dataset_idx][inference_alg]['num_clusters_by_param']
            for dataset_idx in range(num_datasets)])

    plot_inference_algs_num_clusters_by_param(
        num_clusters_by_dataset_by_inference_alg=num_clusters_by_dataset_by_inference_alg,
        plot_dir=plot_dir,
        num_clusters=num_clusters)

    # construct dictionary mapping from scoring metric to inference alg
    # to dataframe with dataset idx as rows and concentration parameters as columns
    # {scoring metric: {inference alg: DataFrame(scores)}}
    scores_by_dataset_by_inference_alg_by_scoring_metric = {}
    for scoring_metric in scoring_metrics:
        scores_by_dataset_by_inference_alg_by_scoring_metric[scoring_metric] = {}
        for inference_alg in inference_algs:
            scores_by_dataset_by_inference_alg_by_scoring_metric[scoring_metric][inference_alg] = \
                pd.DataFrame(
                    [inference_algs_results_by_dataset[dataset_idx][inference_alg]['scores_by_param'][scoring_metric]
                     for dataset_idx in range(num_datasets)])

    plot_inference_algs_scores_by_param(
        scores_by_dataset_by_inference_alg_by_scoring_metric=scores_by_dataset_by_inference_alg_by_scoring_metric,
        plot_dir=plot_dir)


def plot_inference_algs_num_clusters_by_param(num_clusters_by_dataset_by_inference_alg: dict,
                                              plot_dir: str,
                                              num_clusters: int):
    for inference_alg, inference_alg_num_clusters_df in num_clusters_by_dataset_by_inference_alg.items():
        plt.errorbar(x=inference_alg_num_clusters_df.columns.values,  # concentration parameters
                     y=inference_alg_num_clusters_df.mean(),
                     yerr=inference_alg_num_clusters_df.sem(),
                     label=inference_alg)

    plt.xlabel(r'Concentration Parameter ($\alpha$ or $\lambda$)')
    plt.ylabel('Number of Clusters')
    plt.axhline(num_clusters, label='Correct Number Clusters', linestyle='--', color='k')
    plt.gca().set_ylim(bottom=0)
    plt.gca().set_xlim(left=0)
    plt.legend()

    plt.savefig(os.path.join(plot_dir, f'num_clusters_by_param.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_inference_algs_scores_by_param(scores_by_dataset_by_inference_alg_by_scoring_metric: dict,
                                        plot_dir: str):
    # for each scoring function, plot score (y) vs parameter (x)
    for scoring_metric, scores_by_dataset_by_inference_alg in scores_by_dataset_by_inference_alg_by_scoring_metric.items():
        for inference_alg_str, inference_algs_scores_df in scores_by_dataset_by_inference_alg.items():
            plt.errorbar(x=inference_algs_scores_df.columns.values,
                         y=inference_algs_scores_df.mean(),
                         yerr=inference_algs_scores_df.sem(),
                         label=inference_alg_str)
        plt.legend()
        plt.xlabel(r'Concentration Parameter ($\alpha$ or $\lambda$)')
        plt.ylabel(scoring_metric)
        plt.ylim(0., 1.)
        plt.gca().set_xlim(left=0)
        plt.savefig(os.path.join(plot_dir, f'comparison_score={scoring_metric}.png'),
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()


def plot_sample_from_mixture_of_gaussians(assigned_table_seq,
                                          gaussian_samples_seq,
                                          plot_dir):
    plt.scatter(x=gaussian_samples_seq[:, 0],
                y=gaussian_samples_seq[:, 1],
                c=assigned_table_seq)
    plt.savefig(os.path.join(plot_dir, 'sample_from_mixture_of_gaussians.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()

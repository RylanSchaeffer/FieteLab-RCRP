import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os
import pandas as pd
import seaborn as sns


def plot_inference_results(omniglot_dataset_results: dict,
                           inference_results: dict,
                           inference_alg_str: str,
                           concentration_param: float,
                           plot_dir):
    labels = omniglot_dataset_results['assigned_table_seq']
    num_obs = len(labels)
    # for obs_idx in range(num_obs):
    #     fig, axes = plt.subplots(nrows=1,
    #                              ncols=2,
    #                              figsize=(8, 5))
    #     axes[0].imshow(omniglot_dataset_results['images'][obs_idx], cmap='gray')
    #     axes[0].set_title('Image')
    #     axes[1].imshow(np.reshape(inference_results['parameters']['probs'][obs_idx],
    #                               newshape=omniglot_dataset_results['images'][obs_idx].shape),
    #                    cmap='gray')
    #     axes[1].set_title('Parameters')
    #     plt.show()
    #     break

    # params_as_images = omniglot_dataset_results['pca'].inverse_transform(
    #     inference_results['parameters']['means']).reshape((labels.shape[0], 9, 9))
    # fig, axes = plt.subplots(nrows=1,
    #                          ncols=num_obs,
    #                          figsize=(4*num_obs, 5))
    # for obs_idx in range(num_obs):
    #     axes[obs_idx].imshow(params_as_images[obs_idx],
    #                          cmap='gray')
    #     axes[obs_idx].set_title('Parameters')
    # plt.show()

    one_hot_labels = pd.get_dummies(labels)

    # sort dummies by order of class in dataset
    _, order_of_appearance = np.unique(labels, return_index=True)
    idx_order_of_appearance = np.argsort(order_of_appearance)
    one_hot_labels = one_hot_labels[one_hot_labels.columns[idx_order_of_appearance]]

    fig, axes = plt.subplots(nrows=1,
                             ncols=4,
                             figsize=(12, 5))

    ax = axes[0]
    sns.heatmap(data=one_hot_labels.values,
                ax=ax,
                mask=one_hot_labels.values == 0.,
                cmap='jet',
                cbar=False)
    ax.set_ylabel('Observation #')
    ax.set_xlabel('True Class')

    table_assignment_posteriors = np.copy(inference_results['table_assignment_posteriors'])
    posterior_cutoff = 1e-5
    table_assignment_posteriors[table_assignment_posteriors < posterior_cutoff] = np.nan
    max_posterior_index = np.argmax(np.all(np.isnan(table_assignment_posteriors), axis=0)) + 1

    ax = axes[1]
    if 'table_assignment_priors' in inference_results:
        table_assignment_priors = np.copy(inference_results['table_assignment_priors'])
        table_assignment_priors[table_assignment_priors < posterior_cutoff] = np.nan
        sns.heatmap(data=table_assignment_priors[:, :max_posterior_index],
                    ax=ax,
                    yticklabels=False,
                    mask=np.isnan(table_assignment_priors[:, :max_posterior_index]),
                    cmap='jet',
                    norm=LogNorm(),
                    vmax=1.,
                    vmin=posterior_cutoff)
        ax.set_xlabel(r'$p(z_t=k|x_{<t})$')

    # plot predicted classes
    ax = axes[2]
    sns.heatmap(data=table_assignment_posteriors[:, :max_posterior_index],
                ax=ax,
                yticklabels=False,
                mask=np.isnan(table_assignment_posteriors[:, :max_posterior_index]),
                cmap='jet',
                norm=LogNorm(),
                vmax=1.,
                vmin=posterior_cutoff)
    ax.set_xlabel(r'$p(z_t=k|x_{\leq t})$')

    ax = axes[3]
    if 'num_table_posteriors' in inference_results:
        sns.heatmap(data=inference_results['num_table_posteriors'][:, :max_posterior_index],
                    ax=ax,
                    yticklabels=False,
                    mask=inference_results['num_table_posteriors'][:, :max_posterior_index] < posterior_cutoff,
                    cmap='jet',
                    norm=LogNorm(),
                    vmax=1.,
                    vmin=posterior_cutoff)
        ax.set_xlabel(r'$p(K_t=k|x_{\leq t})$')

    plt.savefig(os.path.join(plot_dir, '{}_alpha={:.2f}_pred_assignments.png'.format(inference_alg_str,
                                                                                     concentration_param)),
                bbox_inches='tight',
                dpi=300)
    plt.show()
    plt.close()


def plot_inference_algs_comparison(omniglot_dataset_results,
                                   inference_algs_results_by_dataset_idx: dict,
                                   sampled_permutation_indices_by_dataset_idx: dict,
                                   plot_dir: str):

    num_datasets = len(inference_algs_results_by_dataset_idx)
    num_clusters = len(np.unique(omniglot_dataset_results['assigned_table_seq']))

    inference_algs = list(inference_algs_results_by_dataset_idx[0].keys())
    scoring_metrics = inference_algs_results_by_dataset_idx[0][inference_algs[0]]['scores_by_param'].columns.values

    # we have four dimensions of interest: inference_alg, dataset idx, scoring metric, concentration parameter

    # construct dictionary mapping from inference alg to dataframe
    # with dataset idx as rows and concentration parameters as columns
    # {inference alg: DataFrame(number of clusters)}
    num_clusters_by_dataset_by_inference_alg = {}
    for inference_alg in inference_algs:
        num_clusters_by_dataset_by_inference_alg[inference_alg] = pd.DataFrame([
            inference_algs_results_by_dataset_idx[dataset_idx][inference_alg]['num_clusters_by_param']
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
                    [inference_algs_results_by_dataset_idx[dataset_idx][inference_alg]['scores_by_param'][
                         scoring_metric]
                     for dataset_idx in range(num_datasets)])

    plot_inference_algs_scores_by_param(
        scores_by_dataset_by_inference_alg_by_scoring_metric=scores_by_dataset_by_inference_alg_by_scoring_metric,
        plot_dir=plot_dir)


def plot_inference_algs_num_clusters_by_param(num_clusters_by_dataset_by_inference_alg: dict,
                                              plot_dir: str,
                                              num_clusters: int):
    for inference_alg_str, inference_alg_num_clusters_df in num_clusters_by_dataset_by_inference_alg.items():
        means = inference_alg_num_clusters_df.mean()
        sems = inference_alg_num_clusters_df.sem()

        plt.plot(inference_alg_num_clusters_df.columns.values,  # concentration parameters
                 means,
                 label=inference_alg_str)

        plt.fill_between(
            x=inference_alg_num_clusters_df.columns.values,
            y1=means - sems,
            y2=means + sems,
            alpha=0.3,
            linewidth=0, )

    plt.xlabel(r'Concentration Parameter ($\alpha$ or $\lambda$)')
    plt.ylabel('Number of Clusters')
    plt.axhline(num_clusters, label='Correct Number Clusters', linestyle='--', color='k')
    plt.gca().set_ylim(bottom=1.0)
    plt.gca().set_xlim(left=0.000001)
    plt.legend()
    plt.yscale('log')
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
            means = inference_algs_scores_df.mean()
            sems = inference_algs_scores_df.sem()

            plt.plot(inference_algs_scores_df.columns.values,  # concentration parameters
                     means,
                     label=inference_alg_str)

            plt.fill_between(
                x=inference_algs_scores_df.columns.values,
                y1=means - sems,
                y2=means + sems,
                alpha=0.3,
                linewidth=0, )

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

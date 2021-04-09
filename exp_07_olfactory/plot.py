import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns


def plot_inference_results(features_df,
                           odors_df,
                           inference_results: dict,
                           inference_alg: str,
                           plot_dir):
    fig, axes = plt.subplots(nrows=1,
                             ncols=4,
                             figsize=(16, 5))

    # plot ground truth data
    ax_idx = 0
    ax = axes[ax_idx]
    sns.heatmap(data=features_df.values.T,
                ax=ax,
                yticklabels=features_df.columns.values,
                cmap='jet')
    ax.set_xlabel('Observation #')
    ax.set_title('Ground Truth Features')

    # plot ground truth labels
    ax_idx += 1
    ax = axes[ax_idx]

    one_hot_targets = pd.get_dummies(odors_df['C.A.S.'])

    # sort dummies by order of class in dataset
    _, order_of_appearance = np.unique(odors_df['C.A.S.'], return_index=True)
    idx_order_of_appearance = np.argsort(order_of_appearance)
    one_hot_targets = one_hot_targets[one_hot_targets.columns[idx_order_of_appearance]]

    sns.heatmap(data=one_hot_targets.values.T,
                ax=ax,
                mask=one_hot_targets.values.T == 0.,
                yticklabels=one_hot_targets.columns.values,
                cmap='jet')
    ax.set_xlabel('Observation #')
    ax.set_title('Ground Truth Labels')

    # plot predicted classes
    ax_idx += 1
    ax = axes[ax_idx]
    sns.heatmap(data=inference_results['table_assignment_posteriors'].T,
                ax=ax,
                mask=inference_results['table_assignment_posteriors'].T == 0,
                yticklabels=False,
                cmap='jet')
    ax.set_xlabel('Observation #')
    ax.set_title(r'$p(z_t=k|x_{<t})$')

    # plot per-class probabilities
    ax_idx += 1
    ax = axes[ax_idx]
    sns.heatmap(data=inference_results['parameters']['cluster_probs'],
                ax=ax,
                yticklabels=False,
                xticklabels=False,
                cmap='jet')
    ax.set_ylabel('k')
    ax.set_xlabel('Obs Dim')
    ax.set_title(r'$p_k$')

    plt.savefig(os.path.join(plot_dir, f'{inference_alg}_pred_assignments.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_inference_algs_comparison(features_df,
                                   odors_df,
                                   inference_algs_results_by_dataset: dict,
                                   sampled_permutation_indices_by_dataset: dict,
                                   plot_dir: str):
    num_datasets = len(inference_algs_results_by_dataset)
    num_clusters = len(odors_df['C.A.S.'].unique())

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

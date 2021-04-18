# Common plotting functions


import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def plot_inference_algs_comparison(inference_algs_results_by_dataset_idx: dict,
                                   dataset_by_dataset_idx: dict,
                                   plot_dir: str):

    num_datasets = len(inference_algs_results_by_dataset_idx)
    num_clusters = len(np.unique(dataset_by_dataset_idx[0]['assigned_table_seq']))

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

    # construct dictionary mapping from inference alg to dataframe
    # with dataset idx as rows and concentration parameters as columns
    # {inference alg: DataFrame(runtimes)}
    runtimes_by_dataset_by_inference_alg = {}
    for inference_alg in inference_algs:
        runtimes_by_dataset_by_inference_alg[inference_alg] = pd.DataFrame([
            inference_algs_results_by_dataset_idx[dataset_idx][inference_alg]['runtimes_by_param']
            for dataset_idx in range(num_datasets)])

    plot_inference_algs_runtimes_by_param(
        runtimes_by_dataset_by_inference_alg=runtimes_by_dataset_by_inference_alg,
        plot_dir=plot_dir)

    # construct dictionary mapping from scoring metric to inference alg
    # to dataframe with dataset idx as rows and concentration parameters as columns
    # {scoring metric: {inference alg: DataFrame(scores)}}
    scores_by_dataset_by_inference_alg_by_scoring_metric = {}
    for scoring_metric in scoring_metrics:
        scores_by_dataset_by_inference_alg_by_scoring_metric[scoring_metric] = {}
        for inference_alg in inference_algs:
            scores_by_dataset_by_inference_alg_by_scoring_metric[scoring_metric][inference_alg] = \
                pd.DataFrame(
                    [inference_algs_results_by_dataset_idx[dataset_idx][inference_alg]['scores_by_param'][scoring_metric]
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
            linewidth=0)

    plt.xlabel(r'Concentration Parameter ($\alpha$ or $\lambda$)')
    plt.ylabel('Number of Clusters')
    plt.axhline(num_clusters, label='Correct Number Clusters', linestyle='--', color='k')
    plt.gca().set_ylim(bottom=1.)
    plt.gca().set_xlim(left=0)
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
            plt.plot(inference_algs_scores_df.columns.values,  # concentration parameters
                     inference_algs_scores_df.mean(),
                     label=inference_alg_str)
            plt.fill_between(
                x=inference_algs_scores_df.columns.values,
                y1=inference_algs_scores_df.mean() - inference_algs_scores_df.sem(),
                y2=inference_algs_scores_df.mean() + inference_algs_scores_df.sem(),
                alpha=0.3,
                linewidth=0)

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


def plot_inference_algs_runtimes_by_param(runtimes_by_dataset_by_inference_alg: dict,
                                          plot_dir: str):

    for inference_alg_str, inference_alg_runtime_df in runtimes_by_dataset_by_inference_alg.items():
        means = inference_alg_runtime_df.mean()
        sems = inference_alg_runtime_df.sem()
        plt.plot(inference_alg_runtime_df.columns.values,  # concentration parameters
                 means,
                 label=inference_alg_str)
        plt.fill_between(
            x=inference_alg_runtime_df.columns.values,
            y1=means - sems,
            y2=means + sems,
            alpha=0.3,
            linewidth=0)

    plt.xlabel(r'Concentration Parameter ($\alpha$ or $\lambda$)')
    plt.ylabel('Runtime (s)')
    plt.gca().set_xlim(left=0)
    plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f'runtimes_by_param.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()

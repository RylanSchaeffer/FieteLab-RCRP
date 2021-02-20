import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns


def plot_inference_algs_comparison(inference_algs_results: dict,
                                   sampled_mou_results: dict,
                                   plot_dir: str):

    num_clusters = len(np.unique(sampled_mou_results['assigned_table_seq']))

    plot_inference_algs_num_clusters_by_param(
        inference_algs_results=inference_algs_results,
        plot_dir=plot_dir,
        num_clusters=num_clusters)

    plot_inference_algs_scores_by_param(
        inference_algs_results=inference_algs_results,
        plot_dir=plot_dir)


def plot_inference_algs_num_clusters_by_param(inference_algs_results,
                                              plot_dir,
                                              num_clusters):

    for inference_alg, inference_alg_results in inference_algs_results.items():
        plt.plot(list(inference_alg_results['num_clusters_by_param'].keys()),
                 list(inference_alg_results['num_clusters_by_param'].values()),
                 label=inference_alg)

    plt.xlabel(r'Concentration Parameter ($\alpha$ or $\lambda$)')
    plt.ylabel('Number of Topics')
    plt.axhline(num_clusters, label='Correct Number Topics', linestyle='--', color='k')
    plt.gca().set_ylim(bottom=0)
    plt.gca().set_xlim(left=0)
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f'num_clusters_by_param.png'),
                bbox_inches='tight',
                dpi=300)
    plt.show()
    plt.close()


def plot_inference_algs_scores_by_param(inference_algs_results: dict,
                                        plot_dir: str):

    score_strs = inference_algs_results[list(inference_algs_results.keys())[0]][
        'scores_by_param'].columns.values

    # for each scoring function, plot score (y) vs parameter (x)
    for score_str in score_strs:
        for inference_alg_str, inference_algs_values in inference_algs_results.items():
            plt.plot(inference_algs_values['scores_by_param'].index,
                     inference_algs_values['scores_by_param'][score_str],
                     label=inference_alg_str)
        plt.legend()
        plt.xlabel(r'Concentration Parameter ($\alpha$ or $\lambda$)')
        plt.ylabel(score_str)
        plt.ylim(0., 1.)
        plt.gca().set_xlim(left=0)
        plt.savefig(os.path.join(plot_dir, f'comparison_score={score_str}.png'),
                    bbox_inches='tight',
                    dpi=300)
        plt.show()
        plt.close()


def plot_inference_results(sampled_mou_results: dict,
                           inference_results: dict,
                           inference_alg: str,
                           plot_dir,
                           num_tables_to_plot: int = 10):

    assert isinstance(num_tables_to_plot, int)
    assert num_tables_to_plot > 0

    fig, axes = plt.subplots(nrows=1,
                             ncols=2,
                             figsize=(8, 4),
                             sharey=True)

    ax_idx = 0
    ax = axes[ax_idx]
    sns.heatmap(data=inference_results['table_assignment_posteriors'][:, :num_tables_to_plot],
                ax=ax,
                cmap='Blues')
    ax.set_title('True Topics')
    ax.set_ylabel('Document Index')
    ax.set_xlabel('Topic Index')

    ax_idx += 1
    ax = axes[ax_idx]
    ax.set_title(r'Inferred Topics $p(z_t|o_{\leq t})$')
    sns.heatmap(data=inference_results['table_assignment_posteriors'][:, :num_tables_to_plot],
                ax=ax,
                cmap='Blues')
    ax.set_xlabel('Topic Index')

    plt.savefig(os.path.join(plot_dir, f'{inference_alg}_pred_topics.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_sample_from_mixture_of_unigrams(assigned_table_seq_one_hot,
                                         mixture_of_unigrams,
                                         doc_samples_seq,
                                         plot_dir):

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4),
                             gridspec_kw={"width_ratios": [0.2, 1, 1]}
                             )
    ax = axes[0]
    sns.heatmap(data=mixture_of_unigrams['stick_weights'][:, np.newaxis],
                ax=ax,
                cmap='Blues',
                vmin=0.,
                vmax=1.,
                annot=True,
                cbar=False,
                # cbar_kws=dict(label='P(Topic)'),
                )
    ax.set_title('P(Topic)')
    ax.set_ylabel('Topic Index')

    ax = axes[1]
    sns.heatmap(data=mixture_of_unigrams['topics_parameters'].T,
                ax=ax,
                cmap='Blues',
                # annot=True,
                vmin=0.,
                vmax=1.,
                # cbar_kws=dict(label='P(Word|Topic)'),
                )
    ax.set_title('P(Word|Topic)')
    ax.set_xlabel('Word Index')
    ax.set_ylabel('Topic Index')

    ax = axes[2]
    sns.heatmap(data=doc_samples_seq,
                ax=ax,
                # cbar_kws=dict(label=),
                cmap='Greys',
                mask=doc_samples_seq == 0)
    ax.set_title('Word Counts Per Document')
    ax.set_xlabel('Word Index')
    ax.set_ylabel('Document Index')
    plt.savefig(os.path.join(plot_dir, 'sample_from_mixture_of_unigrams.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()

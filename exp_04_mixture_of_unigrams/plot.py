import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns


def plot_sample_from_mixture_of_unigrams(assigned_table_seq,
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
    ax.set_ylabel('Topic Index')
    ax.set_xlabel('Word Index')
    plt.savefig(os.path.join(plot_dir, 'sample_from_mixture_of_unigrams_topics.png'),
                bbox_inches='tight',
                dpi=300)
    plt.show()
    plt.close()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                             sharey=True,
                             gridspec_kw={"width_ratios": [0.8, 1]})
    ax = axes[0]
    ax.spy(assigned_table_seq, aspect='auto')
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

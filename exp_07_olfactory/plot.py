import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns


def plot_inference_results(features_df,
                           odors_df,
                           inference_results: dict,
                           inference_alg: str,
                           plot_dir,
                           num_tables_to_plot: int = 10):
    assert isinstance(num_tables_to_plot, int)
    assert num_tables_to_plot > 0

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

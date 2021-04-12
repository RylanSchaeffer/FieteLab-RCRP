import matplotlib as mpl
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.special
import scipy.stats
import seaborn as sns

alphas_color_map = {
    1.1: 'tab:blue',
    10.78: 'tab:orange',
    15.37: 'tab:purple',
    30.91: 'tab:green'
}


def plot_chinese_restaurant_table_dist_by_customer_num(analytical_table_distributions_by_alpha_by_T,
                                                       plot_dir):
    # plot how the CRT table distribution changes for T customers
    alphas = list(analytical_table_distributions_by_alpha_by_T.keys())
    T = len(analytical_table_distributions_by_alpha_by_T[alphas[0]])
    table_nums = 1 + np.arange(T)
    cmap = plt.get_cmap('jet_r')
    for alpha in alphas:
        for t in table_nums:
            plt.plot(table_nums,
                     analytical_table_distributions_by_alpha_by_T[alpha][t],
                     # label=f'T={t}',
                     color=cmap(float(t) / T))

        # https://stackoverflow.com/questions/43805821/matplotlib-add-colorbar-to-non-mappable-object
        norm = mpl.colors.Normalize(vmin=1, vmax=T)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])
        colorbar = plt.colorbar(sm,
                                ticks=np.arange(1, T + 1, 5),
                                # boundaries=np.arange(-0.05, T + 0.1, .1)
                                )
        colorbar.set_label('Number of Customers')
        plt.title(fr'Chinese Restaurant Table Distribution ($\alpha$={alpha})')
        plt.xlabel(r'Number of Tables after T Customers')
        plt.ylabel(r'P(Number of Tables after T Customers)')
        plt.savefig(os.path.join(plot_dir, f'crt_table_distribution_alpha={alpha}.png'),
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()


def plot_analytics_vs_monte_carlo_customer_tables(sampled_customer_tables_by_alpha,
                                                  analytical_customer_tables_by_alpha,
                                                  plot_dir):
    # plot analytics versus monte carlo estimates
    for alpha in sampled_customer_tables_by_alpha.keys():
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

        sampled_customer_tables = sampled_customer_tables_by_alpha[alpha]
        average_customer_tables = np.mean(sampled_customer_tables, axis=0)
        analytical_customer_tables = analytical_customer_tables_by_alpha[alpha]

        # replace 0s with nans to allow for log scaling
        average_customer_tables[average_customer_tables == 0.] = np.nan
        cutoff = np.nanmin(average_customer_tables)
        analytical_customer_tables[analytical_customer_tables < cutoff] = np.nan

        ax = axes[0]
        sns.heatmap(average_customer_tables,
                    ax=ax,
                    mask=np.isnan(average_customer_tables),
                    cmap='jet',
                    norm=LogNorm(vmin=cutoff, vmax=1., ),
                    )

        ax.set_title(rf'Monte Carlo Estimate ($\alpha$={alpha})')
        ax.set_ylabel(r'Customer Index')
        ax.set_xlabel(r'Table Index')

        ax = axes[1]
        # log_analytical_customer_tables = np.log(analytical_customer_tables)
        sns.heatmap(analytical_customer_tables,
                    ax=ax,
                    mask=np.isnan(analytical_customer_tables),
                    cmap='jet',
                    norm=LogNorm(vmin=cutoff,
                                 vmax=1., ),
                    )
        ax.set_title(rf'Analytical Prediction ($\alpha$={alpha})')
        ax.set_xlabel(r'Table Index')

        # for some reason, on OpenMind, colorbar ticks disappear without calling plt.show() first
        # plt.show()
        fig.savefig(os.path.join(plot_dir, f'analytics_vs_monte_carlo_customer_tables={alpha}.png'),
                    bbox_inches='tight',
                    dpi=300)
        plt.close()


def plot_analytics_vs_monte_carlo_table_occupancies(sampled_table_occupancies_by_alpha,
                                                    analytical_table_occupancies_by_alpha,
                                                    plot_dir):
    alphas = list(sampled_table_occupancies_by_alpha.keys())
    num_samples, T = sampled_table_occupancies_by_alpha[alphas[0]].shape
    table_nums = 1 + np.arange(T)
    fig, axes = plt.subplots(nrows=1, ncols=len(alphas), figsize=(4 * len(alphas), 4))
    for ax_idx, (alpha, crp_samples) in enumerate(sampled_table_occupancies_by_alpha.items()):
        table_cutoff = alpha * np.log(1 + T / alpha)
        empiric_table_occupancies_mean_by_repeat = np.mean(crp_samples, axis=0)
        empiric_table_occupancies_sem = scipy.stats.sem(crp_samples, axis=0)
        axes[ax_idx].set_title(rf'CRP($\alpha$={alpha})')
        for num_samples_idx in range(200):
            axes[ax_idx].plot(table_nums, crp_samples[num_samples_idx, :], alpha=0.01, color='k')
        axes[ax_idx].errorbar(x=table_nums,
                              y=empiric_table_occupancies_mean_by_repeat,
                              yerr=empiric_table_occupancies_sem,
                              # linewidth=2,
                              fmt='--',
                              color='k',
                              label=f'Empiric (N={num_samples})')
        axes[ax_idx].scatter(table_nums[:len(analytical_table_occupancies_by_alpha[alpha])],
                             analytical_table_occupancies_by_alpha[alpha],
                             # '--',
                             marker='d',
                             color=alphas_color_map[alpha],
                             # linewidth=2,
                             label=f'Analytic')
        print(f'Plotted alpha={alpha}')
        axes[ax_idx].legend()
        axes[ax_idx].set_xlabel('Table Number')
        axes[ax_idx].set_ylabel('Table Occupancy')
        axes[ax_idx].set_xlim(1, table_cutoff)

    plt.savefig(os.path.join(plot_dir, f'analytics_vs_monte_carlo_table_occupancies.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_recursion_visualization(analytical_customer_tables_by_alpha,
                                 analytical_table_distributions_by_alpha_by_T,
                                 plot_dir):
    alphas = list(analytical_customer_tables_by_alpha.keys())
    cutoff = 1e-8

    for alpha in alphas:
        fig, axes = plt.subplots(nrows=1,
                                 ncols=5,
                                 figsize=(13, 4),
                                 gridspec_kw={"width_ratios": [1, 0.1, 1, 0.1, 1]},
                                 sharex=True)

        ax = axes[0]
        cum_customer_seating_probs = np.cumsum(analytical_customer_tables_by_alpha[alpha], axis=0)
        cum_customer_seating_probs[cum_customer_seating_probs < cutoff] = np.nan

        max_table_idx = np.argmax(np.nansum(cum_customer_seating_probs, axis=0) < cutoff)
        sns.heatmap(
            data=cum_customer_seating_probs[:, :max_table_idx],
            ax=ax,
            cbar_kws=dict(label=r'$\sum_{t^{\prime} = 1}^{t-1} p(z_{t\prime} = k)$'),
            cmap='jet',
            mask=np.isnan(cum_customer_seating_probs[:, :max_table_idx]),
            norm=LogNorm(vmin=cutoff),
        )
        ax.set_xlabel('Table Index')
        ax.set_ylabel('Customer Index')
        ax.set_title('Running Sum of\nPrev. Customers\' Distributions')

        # necessary to make space for colorbar text
        axes[1].axis('off')

        ax = axes[2]
        table_distributions_by_T_array = np.stack([
            analytical_table_distributions_by_alpha_by_T[alpha][key]
            for key in sorted(analytical_table_distributions_by_alpha_by_T[alpha].keys())])
        table_distributions_by_T_array[table_distributions_by_T_array < cutoff] = np.nan
        sns.heatmap(
            data=table_distributions_by_T_array[:, :max_table_idx],
            ax=ax,
            cbar_kws=dict(label='$p(K_t = k)$'),
            cmap='jet',
            mask=np.isnan(table_distributions_by_T_array[:, :max_table_idx]),
            norm=LogNorm(vmin=cutoff, ),
        )
        ax.set_title('Distribution over\nNumber of Non-Empty Tables')
        ax.set_xlabel('Table Index')

        axes[3].axis('off')

        ax = axes[4]
        analytical_customer_tables = np.copy(analytical_customer_tables_by_alpha[alpha])
        analytical_customer_tables[analytical_customer_tables < cutoff] = np.nan
        sns.heatmap(
            data=analytical_customer_tables[:, :max_table_idx],
            ax=ax,
            cbar_kws=dict(label='$p(z_t)$'),
            cmap='jet',
            mask=np.isnan(analytical_customer_tables[:, :max_table_idx]),
            norm=LogNorm(vmin=cutoff, ),
        )
        ax.set_title('New Customer\'s Distribution')
        ax.set_xlabel('Table Index')
        plt.savefig(os.path.join(plot_dir, f'crp_recursion_alpha={alpha}.png'),
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()


def plot_analytical_vs_monte_carlo_mse(sampled_customer_tables_by_alpha_by_rep,
                                       analytical_customer_tables_by_alpha,
                                       plot_dir):
    alphas = list(sampled_customer_tables_by_alpha_by_rep.keys())
    num_reps = len(sampled_customer_tables_by_alpha_by_rep[alphas[0]])

    possible_num_samples = np.logspace(1, 4, 4).astype(np.int)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
    for alpha_idx, alpha in enumerate(alphas):

        means_per_num_samples, sems_per_num_samples = [], []
        for num_samples in possible_num_samples:
            rep_errors = []
            for rep_idx in range(num_reps):
                # TODO: refactor double alpha indexing b/c unnecessary
                rep_error = np.square(np.linalg.norm(
                    np.subtract(
                        np.mean(sampled_customer_tables_by_alpha_by_rep[alpha][rep_idx][alpha][:num_samples], axis=0),
                        analytical_customer_tables_by_alpha[alpha])
                ))
                rep_errors.append(rep_error)
            means_per_num_samples.append(np.mean(rep_errors))
            sems_per_num_samples.append(scipy.stats.sem(rep_errors))

        ax.errorbar(x=possible_num_samples,
                    y=means_per_num_samples,
                    yerr=sems_per_num_samples,
                    label=rf'$\alpha$={alpha}',
                    c=alphas_color_map[alpha])
    ax.legend(title=f'Num Repeats: {num_reps}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'$(Analytic - Empiric)^2$')
    # ax.set_ylabel(r'$\mathbb{E}_D[\sum_k (\mathbb{E}[N_{T, k}] - \frac{1}{S} \sum_{s=1}^S N_{T, k}^{(s)})^2]$')
    ax.set_xlabel('Number of Samples')
    plt.savefig(os.path.join(plot_dir, f'crp_expected_mse.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()

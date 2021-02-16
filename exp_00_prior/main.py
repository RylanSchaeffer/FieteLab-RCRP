import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.special
import scipy.stats
import seaborn as sns
from sympy.functions.combinatorial.numbers import stirling

from utils.data import vectorized_sample_sequence_from_crp

exp_dir = 'exp_00_prior'
plot_dir = os.path.join(exp_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)

T = 50  # max time
num_samples = 5000  # number of samples to draw from CRP(alpha)
alphas = [1.1, 10.01, 30.03]  # CRP parameter
alphas_color_map = {
    1.1:    'tab:blue',
    10.01:  'tab:orange',
    30.03:  'tab:green'
}
crp_samples_by_alpha = {}
for alpha in alphas:
    crp_empirics_path = os.path.join(exp_dir, f'crp_sample_{alpha}.npy')
    if os.path.isfile(crp_empirics_path):
        table_occupancies = np.load(crp_empirics_path)
        print(f'Loaded samples for {crp_empirics_path}')
        assert table_occupancies.shape == (num_samples, T)
    else:
        table_occupancies, _ = vectorized_sample_sequence_from_crp(
            T=np.full(shape=num_samples, fill_value=T),
            alpha=np.full(shape=num_samples, fill_value=alpha))
        print(f'Generated samples for {crp_empirics_path}')
        table_occupancies = np.stack(table_occupancies)
        np.save(arr=table_occupancies, file=crp_empirics_path)
    crp_samples_by_alpha[alpha] = table_occupancies


def chinese_table_restaurant_distribution(t, k, alpha):
    if k > t:
        prob = 0.
    else:
        prob = scipy.special.gamma(alpha)
        prob *= stirling(n=t, k=k, kind=1, signed=False)
        prob /= scipy.special.gamma(alpha + t)
        prob *= np.power(alpha, k)
    return prob


table_nums = 1 + np.arange(T)
table_distributions_by_alpha = {}
for alpha in alphas:
    result = np.zeros(shape=T)
    for repeat_idx in table_nums:
        result[repeat_idx - 1] = chinese_table_restaurant_distribution(t=T, k=repeat_idx, alpha=alpha)
    table_distributions_by_alpha[alpha] = result
    plt.plot(table_nums, table_distributions_by_alpha[alpha], label=f'alpha={alpha}')
plt.legend()
plt.xlabel(f'Number of Tables after T={T} customers (K_T)')
plt.ylabel('P(K_T = k)')
plt.show()

table_nums = 1 + np.arange(T)
table_distributions_by_T = {}
alpha = 30.03
cmap = plt.get_cmap('jet_r')
for t in table_nums:
    result = np.zeros(shape=T)
    for repeat_idx in np.arange(1, 1 + t):
        result[repeat_idx - 1] = chinese_table_restaurant_distribution(t=t, k=repeat_idx, alpha=alpha)
    table_distributions_by_T[t] = result
    if t == 1 or t == T:
        plt.plot(table_nums,
                 table_distributions_by_T[t],
                 label=f'T={t}',
                 color=cmap(float(t) / T))
    else:
        plt.plot(table_nums,
                 table_distributions_by_T[t],
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
plt.savefig(os.path.join(plot_dir, 'crt_table_distribution.png'),
            bbox_inches='tight',
            dpi=300)
plt.show()


def construct_analytical_customer_probs_and_table_occupancies(T, alpha):
    expected_num_tables = min(2 * int(alpha * np.log(1 + T / alpha)), T)
    customer_seating_probs = np.zeros(shape=(T + 1, expected_num_tables + 1))
    customer_seating_probs[1, 1] = 1.
    customer_seating_probs[2, 1] = 1. / (1. + alpha)
    customer_seating_probs[2, 2] = alpha / (1. + alpha)
    for customer_num in range(3, T + 1):
        customer_seating_probs[customer_num, :] += customer_seating_probs[customer_num - 1, :]
        for table_num in range(1, expected_num_tables + 1):
            diff = chinese_table_restaurant_distribution(t=customer_num - 1, k=table_num - 1, alpha=alpha) - \
                   chinese_table_restaurant_distribution(t=customer_num - 2, k=table_num - 1, alpha=alpha)
            scaled_diff = alpha * diff / (alpha + customer_num - 1)
            customer_seating_probs[customer_num, table_num] += scaled_diff
    analytical_table_occupancies = np.sum(customer_seating_probs, axis=0)
    return customer_seating_probs[1:, 1:], analytical_table_occupancies[1:]


analytical_table_occupancies_by_alpha = {}
for alpha in alphas:
    crp_analytics_path = os.path.join(exp_dir, f'crp_analytics_{alpha}.npy')
    if os.path.isfile(crp_analytics_path):
        analytical_table_occupancies = np.load(crp_analytics_path)
    else:
        customer_seating_probs, analytical_table_occupancies = construct_analytical_customer_probs_and_table_occupancies(
            T=T,
            alpha=alpha)
        np.save(arr=analytical_table_occupancies, file=crp_analytics_path)
    analytical_table_occupancies_by_alpha[alpha] = analytical_table_occupancies

table_nums = 1 + np.arange(T)
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for ax_idx, (alpha, crp_samples) in enumerate(crp_samples_by_alpha.items()):
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

plt.savefig(os.path.join(plot_dir, f'crp_table_occupancies.png'),
            bbox_inches='tight',
            dpi=300)
plt.show()


table_nums = 1 + np.arange(T)
for alpha, crp_samples in crp_samples_by_alpha.items():
    table_cutoff = alpha * np.log(1 + T / alpha)
    empiric_table_occupancies_mean_by_repeat = np.mean(crp_samples, axis=0)
    empiric_table_occupancies_sem = scipy.stats.sem(crp_samples, axis=0)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(40, 10), sharey=True)
    fig.suptitle(rf'$alpha$={alpha}')
    for num_samples_idx in range(200):
        axes[0].plot(table_nums, crp_samples[num_samples_idx, :], alpha=0.01, color='k')
    axes[0].scatter(x=table_nums,
                    y=empiric_table_occupancies_mean_by_repeat,
                    # yerr=empiric_table_occupancies_sem,
                    # linewidth=2,
                    # fmt='none',
                    label=f'Empiric (N={num_samples})')
    axes[0].scatter(table_nums[:len(analytical_table_occupancies_by_alpha[alpha])],
                    analytical_table_occupancies_by_alpha[alpha],
                    # '--',
                    marker='x',
                    # linewidth=2,
                    label=f'Analytic')
    print(f'Plotted alpha={alpha}')
    axes[0].legend()
    axes[0].set_xlabel('Table Number')
    axes[0].set_ylabel('Mean Table Occupancy')
    axes[0].set_xlim(1, table_cutoff)

    table_cutoff = 10
    table_idx = np.cumsum(np.ones_like(crp_samples[:, :table_cutoff]), axis=1)
    crp_samples_df = pd.DataFrame({
        'num_occupants': np.reshape(crp_samples[:, :table_cutoff], -1),
        'table_idx': np.reshape(table_idx, -1)})
    sns.boxenplot(x='table_idx',
                  y='num_occupants',
                  data=crp_samples_df,
                  ax=axes[1])
    axes[1].set_xlabel('Table Number')
    axes[1].set_ylabel('Number of Occupants')
    axes[1].set_xlim(-1, table_cutoff)

    sns.violinplot(x='table_idx',
                   y='num_occupants',
                   data=crp_samples_df,
                   ax=axes[2])
    axes[2].set_xlabel('Table Number')
    axes[2].set_ylabel('Number of Occupants')
    axes[2].set_xlim(-1, table_cutoff)

    plt.savefig(os.path.join(plot_dir, f'crp_{alpha}_summary.png'),
                bbox_inches='tight',
                dpi=300)
    plt.show()


num_datasets, num_tables = 10, T
possible_num_samples = np.logspace(1, 4, 4).astype(np.int)
mse_empiric_vs_analytical_per_dataset = np.zeros(shape=(len(possible_num_samples),
                                                        len(alphas),
                                                        num_datasets,
                                                        num_tables))

for num_samples_idx, num_samples in enumerate(possible_num_samples):
    for alpha_idx, alpha in enumerate(alphas):
        analytical_table_occupancies = analytical_table_occupancies_by_alpha[alpha]
        for repeat_idx in range(num_datasets):
            table_occupancies, sampled_tables = vectorized_sample_sequence_from_crp(
                T=np.full(shape=num_samples, fill_value=T),
                alpha=np.full(shape=num_samples, fill_value=alpha))
            table_occupancies = np.stack(table_occupancies)
            mse_empiric_vs_analytical_per_dataset[num_samples_idx, alpha_idx, repeat_idx, :len(analytical_table_occupancies)] = np.square(
                np.mean(table_occupancies, axis=0)[:len(analytical_table_occupancies)] - analytical_table_occupancies)
mse_empiric_vs_analytical_per_dataset = np.sum(mse_empiric_vs_analytical_per_dataset, axis=3)


mse_empiric_vs_analytical_mean = np.mean(mse_empiric_vs_analytical_per_dataset, axis=2)
mse_empiric_vs_analytical_sem = scipy.stats.sem(mse_empiric_vs_analytical_per_dataset, axis=2)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
for alpha_idx, alpha in enumerate(alphas):
    ax.errorbar(x=possible_num_samples,
                y=mse_empiric_vs_analytical_mean[:, alpha_idx],
                yerr=mse_empiric_vs_analytical_sem[:, alpha_idx],
                label=rf'$\alpha$={alpha}')
ax.legend()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel(r'$\mathbb{E}_D[\sum_k (\mathbb{E}[N_{T, k}] - \frac{1}{S} \sum_{s=1}^S N_{T, k}^{(s)})^2]$')
ax.set_xlabel('Number of Samples (S)')
plt.savefig(os.path.join(plot_dir, f'crp_expected_mse.png'),
            bbox_inches='tight',
            dpi=300)
plt.show()

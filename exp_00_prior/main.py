import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.special
import scipy.stats
import seaborn as sns
from sympy.functions.combinatorial.numbers import stirling


exp_dir = 'exp_00_prior'
plot_dir = os.path.join(exp_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)


def draw_sample_from_crp(T, alpha):
    counts = np.zeros(shape=T)
    counts[0] = 1
    for t in range(1, T):
        max_k = np.argmin(counts)  # first zero index
        freq = counts.copy()
        freq[max_k] = alpha
        probs = freq / np.sum(freq)
        z_t = np.random.choice(np.arange(max_k + 1), p=probs[:max_k + 1])
        counts[z_t] += 1
    return counts


vectorized_draw_sample_from_crp = np.vectorize(draw_sample_from_crp,
                                               otypes=[np.ndarray])

T = 50  # max time
num_samples = 5000  # number of samples to draw from CRP(alpha)
alphas = [1.1, 10.01, 30.03]  # CRP parameter
crp_samples_by_alpha = {}
for alpha in alphas:
    crp_empirics_path = os.path.join(exp_dir, f'crp_sample_{alpha}.npy')
    if os.path.isfile(crp_empirics_path):
        crp_samples = np.load(crp_empirics_path)
        print(f'Loaded samples for {crp_empirics_path}')
        assert crp_samples.shape == (num_samples, T)
    else:
        crp_samples = vectorized_draw_sample_from_crp(
            T=np.full(shape=num_samples, fill_value=T),
            alpha=np.full(shape=num_samples, fill_value=alpha))
        print(f'Generated samples for {crp_empirics_path}')
        crp_samples = np.stack(crp_samples)
        np.save(arr=crp_samples, file=crp_empirics_path)
    crp_samples_by_alpha[alpha] = crp_samples


def prob_kth_table_exists_at_time_t(t, k, alpha):
    if k > t:
        prob = 0.
    else:
        prob = 1.
        prob *= scipy.special.gamma(alpha)
        prob *= stirling(n=t, k=k, kind=1, signed=False)
        prob /= scipy.special.gamma(alpha + t)
        prob *= np.power(alpha, k)
    return prob


table_nums = 1 + np.arange(T)
table_distributions_by_alpha = {}
for alpha in alphas:
    result = np.zeros(shape=T)
    for k in table_nums:
        result[k-1] = prob_kth_table_exists_at_time_t(t=T, k=k, alpha=alpha)
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
    for k in np.arange(1, 1+t):
        result[k-1] = prob_kth_table_exists_at_time_t(t=t, k=k, alpha=alpha)
    table_distributions_by_T[t] = result
    if t == 1 or t == T:
        plt.plot(table_nums,
                 table_distributions_by_T[t],
                 label=f'T={t}',
                 color=cmap(float(t)/T))
    else:
        plt.plot(table_nums,
                 table_distributions_by_T[t],
                 # label=f'T={t}',
                 color=cmap(float(t)/T))
# https://stackoverflow.com/questions/43805821/matplotlib-add-colorbar-to-non-mappable-object
norm = mpl.colors.Normalize(vmin=1, vmax=T)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
plt.colorbar(sm,
             ticks=np.linspace(1, T, T),
             boundaries=np.arange(-0.05, T + 0.1, .1))
plt.title(fr'Chinese Restaurant Table Distribution ($\alpha$={alpha})')
plt.xlabel(r'Number of Tables after T customers')
plt.ylabel(r'P(Number of Tables after T customers)')
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
            diff = prob_kth_table_exists_at_time_t(t=customer_num - 1, k=table_num - 1, alpha=alpha) - \
                   prob_kth_table_exists_at_time_t(t=customer_num - 2, k=table_num - 1, alpha=alpha)
            scaled_diff = alpha * diff / (alpha + customer_num - 1)
            customer_seating_probs[customer_num, table_num] += scaled_diff
    analytical_table_occupancies = np.sum(customer_seating_probs, axis=0)
    return customer_seating_probs[1:, 1:], analytical_table_occupancies[1:]


# TODO: how is R code so much faster
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

# vectorized_prob_tth_customer_at_table_k = np.vectorize(prob_tth_customer_at_table_k,
#                                                        otypes=[np.ndarray])


# def compute_unsigned_stirling_nums_of_first_kind(T):
#     unsigned_stirling_matrix = np.zeros((T, T), dtype=np.float64)
#     unsigned_stirling_matrix[0, 0] = 1
#     for i in range(1, T):
#         unsigned_stirling_matrix[i, 0] = - i * unsigned_stirling_matrix[i - 1, 0]
#         for j in range(1, T):
#             unsigned_stirling_matrix[i, j] = unsigned_stirling_matrix[i - 1, j - 1] - i * unsigned_stirling_matrix[i - 1, j]
#     return np.abs(unsigned_stirling_matrix)
#
#
# unsigned_stirling_matrix = compute_unsigned_stirling_nums_of_first_kind(T=T)


table_nums = 1 + np.arange(T)
for alpha, crp_samples in crp_samples_by_alpha.items():
    table_cutoff = alpha * np.log(1 + T / alpha)
    empiric_table_occupancies_mean = np.mean(crp_samples, axis=0)
    empiric_table_occupancies_sem = scipy.stats.sem(crp_samples, axis=0)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(40, 10), sharey=True)
    fig.suptitle(f'alpha={alpha}')
    for i in range(200):
        axes[0].plot(table_nums, crp_samples[i, :], alpha=0.01, color='k')
    axes[0].scatter(x=table_nums,
                    y=empiric_table_occupancies_mean,
                    # yerr=empiric_table_occupancies_sem,
                    # linewidth=2,
                    # fmt='none',
                    label=f'Empiric (N={num_samples}, alpha={alpha})')
    axes[0].scatter(table_nums[:len(analytical_table_occupancies_by_alpha[alpha])],
                    analytical_table_occupancies_by_alpha[alpha],
                    # '--',
                    marker='x',
                    # linewidth=2,
                    label=f'Analytic (alpha={alpha}, sum={np.sum(analytical_table_occupancies_by_alpha[alpha])})')
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

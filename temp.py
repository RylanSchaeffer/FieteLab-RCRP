import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.special
import scipy.stats
from sympy.functions.combinatorial.numbers import stirling


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

T = 200  # max time
num_samples = 2000  # number of samples to draw from CRP(alpha)
alphas = [1.1, 10.01, 30.03]  # CRP parameter
crp_samples_by_alpha = {}
for alpha in alphas:
    crp_samples_path = f'crp_sample_{alpha}.npy'
    if os.path.isfile(crp_samples_path):
        crp_samples = np.load(crp_samples_path)
        print(f'Loaded samples for {crp_samples_path}')
        assert crp_samples.shape == (num_samples, T)
    else:
        crp_samples = vectorized_draw_sample_from_crp(
            T=np.full(shape=num_samples, fill_value=T),
            alpha=np.full(shape=num_samples, fill_value=alpha))
        print(f'Generated samples for {crp_samples}')
        crp_samples = np.stack(crp_samples)
        np.save(arr=crp_samples, file=crp_samples_path)
    crp_samples_by_alpha[alpha] = crp_samples


def prob_kth_table_exists_at_time_t(t, k, alpha):
    if k > t:
        prob = 0.
    else:
        prob = scipy.special.gamma(alpha)
        prob /= scipy.special.gamma(alpha + t)
        prob *= np.power(alpha, k)
        prob *= stirling(n=t, k=k, kind=1, signed=False)
    return prob


def prob_tth_customer_at_table_k(t, k, alpha):
    if t == 1:
        if k == 1:
            prob = 1.
        else:
            prob = 0.
    if t == 2:
        if k == 1:
            prob = 1. / (1. + alpha)
        elif k == 2:
            prob = alpha / (1. + alpha)
        else:
            prob = 0.
    else:
        prob = prob_tth_customer_at_table_k(t=2, k=k, alpha=alpha)
        for tprime in range(2, t):
            diff = prob_kth_table_exists_at_time_t(t=tprime, k=k-1, alpha=alpha) -\
                   prob_kth_table_exists_at_time_t(t=tprime-1, k=k - 1, alpha=alpha)
            prob = prob + alpha * diff / (alpha + t)
    return prob


# theoretical_table_occupancies = {}
# for alpha in alphas:
#     theoretical_table_occupancies[alpha] = {}
#     for k in range(1, T):
#         running_prob_sum = 0.
#         for t in range(T):
#             running_prob_sum += prob_tth_customer_at_table_k(t=t, k=k, alpha=alpha)
#         theoretical_table_occupancies[alpha][k] = running_prob_sum
# 
# 
# print(10)


# def compute_unsigned_stirling_nums_of_first_kind(T):
#     unsigned_stirling_matrix = np.zeros((T, T), dtype=np.float64)
#     unsigned_stirling_matrix[0, 0] = 1
#     for i in range(1, T):
#         unsigned_stirling_matrix[i, 0] = - i * unsigned_stirling_matrix[i - 1, 0]
#         for j in range(1, T):
#             unsigned_stirling_matrix[i, j] = unsigned_stirling_matrix[i - 1, j - 1] - i * unsigned_stirling_matrix[i - 1, j]
#     return np.abs(unsigned_stirling_matrix)


# unsigned_stirling_matrix = compute_unsigned_stirling_nums_of_first_kind(T=T)


num_traces = 200
table_nums = 1 + np.arange(T)
for alpha, crp_samples in crp_samples_by_alpha.items():
    plot_cutoff = alpha * np.log(1 + T / alpha)
    empiric_table_occupancies_mean = np.mean(crp_samples, axis=0)
    empiric_table_occupancies_sem = scipy.stats.sem(crp_samples, axis=0)
    plt.errorbar(x=table_nums,
                 y=empiric_table_occupancies_mean,
                 yerr=empiric_table_occupancies_sem,
                 label=f'Empiric (N={num_samples})')
    for i in range(num_traces):
        plt.plot(table_nums, crp_samples[i, :], alpha=0.05, color='k')
    print(f'Plotted alpha={alpha}')
    plt.legend()
    plt.title(f'alpha={alpha}')
    plt.xlabel('Table Number')
    plt.xlim(1, plot_cutoff)
    # plt.xscale('log')
    plt.ylabel('Mean Table Occupancy')
    plt.show()
    plt.savefig(f'crp_sample_{alpha}.png')

import numpy as np
import os
import pandas as pd
import scipy.stats
from sympy.functions.combinatorial.numbers import stirling

from utils.data import vectorized_sample_sequence_from_crp
from plot import *


def main():
    # set seed
    np.random.seed(1)

    # create directories
    exp_dir = 'exp_00_crp_prior'
    plot_dir = os.path.join(exp_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    T = 50  # max time
    num_samples = 5000  # number of samples to draw from CRP(alpha)
    alphas = [1.1, 10.78, 15.37, 30.91]

    sampled_table_occupancies_by_alpha, sampled_customer_tables_by_alpha = sample_from_crp(
        T=T,
        alphas=alphas,
        exp_dir=exp_dir,
        num_samples=num_samples)

    analytical_table_distributions_by_alpha_by_T = construct_analytical_crt(T=T, alphas=alphas)

    analytical_table_occupancies_by_alpha, analytical_customer_tables_by_alpha = construct_analytical_crp(
        T=T,
        alphas=alphas,
        exp_dir=exp_dir)

    # plot_chinese_restaurant_table_dist_by_customer_num(
    #     analytical_table_distributions_by_alpha_by_T=analytical_table_distributions_by_alpha_by_T,
    #     plot_dir=plot_dir)
    #
    # plot_recursion_visualization(
    #     analytical_customer_tables_by_alpha=analytical_customer_tables_by_alpha,
    #     analytical_table_distributions_by_alpha_by_T=analytical_table_distributions_by_alpha_by_T,
    #     # analytical_table_occupancies_by_alpha=analytical_table_occupancies_by_alpha,
    #     plot_dir=plot_dir)
    #
    # plot_analytics_vs_monte_carlo_table_occupancies(
    #     sampled_table_occupancies_by_alpha=sampled_table_occupancies_by_alpha,
    #     analytical_table_occupancies_by_alpha=analytical_table_occupancies_by_alpha,
    #     plot_dir=plot_dir)
    #
    # plot_analytics_vs_monte_carlo_customer_tables(
    #     sampled_customer_tables_by_alpha=sampled_customer_tables_by_alpha,
    #     analytical_customer_tables_by_alpha=analytical_customer_tables_by_alpha,
    #     plot_dir=plot_dir)

    num_reps = 2
    sampled_customer_tables_by_alpha_by_rep = {}
    for alpha in alphas:
        sampled_customer_tables_by_alpha_by_rep[alpha] = []
        for rep_idx in range(num_reps):
            _, sampled_customer_tables = sample_from_crp(
                T=T,
                alphas=alphas,
                exp_dir=exp_dir,
                num_samples=num_samples,
                rep_idx=rep_idx)
            sampled_customer_tables_by_alpha_by_rep[alpha].append(sampled_customer_tables)

    plot_analytical_vs_monte_carlo_mse(
        sampled_customer_tables_by_alpha_by_rep=sampled_customer_tables_by_alpha_by_rep,
        analytical_customer_tables_by_alpha=analytical_customer_tables_by_alpha,
        plot_dir=plot_dir)


def chinese_table_restaurant_distribution(t, k, alpha):
    if k > t:
        prob = 0.
    else:
        prob = scipy.special.gamma(alpha)
        prob *= stirling(n=t, k=k, kind=1, signed=False)
        prob /= scipy.special.gamma(alpha + t)
        prob *= np.power(alpha, k)
    return prob


def sample_from_crp(T,
                    alphas,
                    exp_dir,
                    num_samples,
                    rep_idx=0):

    # generate Monte Carlo samples from the CRP
    sampled_table_occupancies_by_alpha = {}
    sampled_customer_tables_by_alpha = {}
    for alpha in alphas:
        crp_samples_path = os.path.join(exp_dir, f'crp_sample_{alpha}_rep_idx={rep_idx}.npz')
        if os.path.isfile(crp_samples_path):
            crp_sample_data = np.load(crp_samples_path)
            table_occupancies = crp_sample_data['table_occupancies']
            customer_tables_one_hot = crp_sample_data['customer_tables_one_hot']
            print(f'Loaded samples for {crp_samples_path}')
            assert table_occupancies.shape == (num_samples, T)
        else:
            table_occupancies, customer_tables, customer_tables_one_hot = vectorized_sample_sequence_from_crp(
                T=np.full(shape=num_samples, fill_value=T),
                alpha=np.full(shape=num_samples, fill_value=alpha))
            print(f'Generated samples for {crp_samples_path}')
            table_occupancies = np.stack(table_occupancies)
            customer_tables = np.stack(customer_tables)
            customer_tables_one_hot = np.stack(customer_tables_one_hot)
            np.savez(file=crp_samples_path,
                     table_occupancies=table_occupancies,
                     customer_tables=customer_tables,
                     customer_tables_one_hot=customer_tables_one_hot)
        sampled_table_occupancies_by_alpha[alpha] = table_occupancies
        sampled_customer_tables_by_alpha[alpha] = customer_tables_one_hot

    return sampled_table_occupancies_by_alpha, sampled_customer_tables_by_alpha


def construct_analytical_crp(T, alphas, exp_dir):
    analytical_table_occupancies_by_alpha = {}
    analytical_customer_tables_by_alpha = {}

    for alpha in alphas:
        crp_analytics_path = os.path.join(exp_dir, f'crp_analytics_{alpha}.npz')
        if os.path.isfile(crp_analytics_path):
            npz_file = np.load(crp_analytics_path)
            customer_seating_probs = npz_file['customer_seating_probs']
            analytical_table_occupancies = npz_file['analytical_table_occupancies']
        else:
            customer_seating_probs, analytical_table_occupancies = \
                construct_analytical_customer_tables_and_table_occupancies(
                    T=T,
                    alpha=alpha)
            np.savez(analytical_table_occupancies=analytical_table_occupancies,
                     customer_seating_probs=customer_seating_probs,
                     file=crp_analytics_path)
        analytical_table_occupancies_by_alpha[alpha] = analytical_table_occupancies
        analytical_customer_tables_by_alpha[alpha] = customer_seating_probs

    return analytical_table_occupancies_by_alpha, analytical_customer_tables_by_alpha


def construct_analytical_crt(T,
                             alphas):
    table_nums = 1 + np.arange(T)
    table_distributions_by_alpha_by_T = {}
    for alpha in alphas:
        table_distributions_by_alpha_by_T[alpha] = {}
        for t in table_nums:
            result = np.zeros(shape=T)
            for repeat_idx in np.arange(1, 1 + t):
                result[repeat_idx - 1] = chinese_table_restaurant_distribution(
                    t=t,
                    k=repeat_idx,
                    alpha=alpha)
            table_distributions_by_alpha_by_T[alpha][t] = result
    return table_distributions_by_alpha_by_T


def construct_analytical_customer_tables_and_table_occupancies(T, alpha):
    # expected_num_tables = min(5 * int(alpha * np.log(1 + T / alpha)), T)
    expected_num_tables = T
    customer_seating_probs = np.zeros(shape=(T + 1, T + 1))
    customer_seating_probs[1, 1] = 1.
    customer_seating_probs[2, 1] = 1. / (1. + alpha)
    customer_seating_probs[2, 2] = alpha / (1. + alpha)
    # Approach 1: Recursive
    # for customer_num in range(3, T + 1):
    #     customer_seating_probs[customer_num, :] += customer_seating_probs[customer_num - 1, :]
    #     for table_num in range(1, expected_num_tables + 1):
    #         diff = chinese_table_restaurant_distribution(t=customer_num - 1, k=table_num - 1, alpha=alpha) - \
    #                chinese_table_restaurant_distribution(t=customer_num - 2, k=table_num - 1, alpha=alpha)
    #         scaled_diff = alpha * diff / (alpha + customer_num - 1)
    #         customer_seating_probs[customer_num, table_num] += scaled_diff

    # Approach 2: Iterative
    for customer_num in range(3, T + 1):
        customer_seating_probs[customer_num, :] = np.sum(customer_seating_probs[:customer_num, :],
                                                         axis=0)
        for table_num in range(1, expected_num_tables + 1):
            crt_k = chinese_table_restaurant_distribution(t=customer_num - 1, k=table_num - 1, alpha=alpha)
            customer_seating_probs[customer_num, table_num] += alpha * crt_k
        customer_seating_probs[customer_num, :] /= (alpha + customer_num - 1)
    analytical_table_occupancies = np.sum(customer_seating_probs, axis=0)
    return customer_seating_probs[1:, 1:], analytical_table_occupancies[1:]


if __name__ == '__main__':
    main()

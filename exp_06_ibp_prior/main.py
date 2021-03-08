import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats
import seaborn as sns

from utils.data import vectorized_sample_sequence_from_ibp, sample_sequence_from_ibp

np.random.seed(1)

exp_dir = 'exp_06_ibp_prior'
plot_dir = os.path.join(exp_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)

T = 50  # max time
num_samples = 5000  # number of samples to draw from IBP(alpha)
alphas = [10.37, 1.1, 15.78, 30.91]  # CRP parameter

# alpha = alphas[0]
# max_dishes = int(10 * alpha * np.sum(1 / (1 + np.arange(T))))
# dish_indices = np.arange(max_dishes + 1)
# prev_running_harmonic_sum = alpha / 1
# for t in range(2, T):
#     prev_dish_distribution = scipy.stats.poisson.pmf(
#         dish_indices,
#         mu=prev_running_harmonic_sum)
#     new_running_harmonic_sum = prev_running_harmonic_sum + alpha / t
#     new_dish_distribution = scipy.stats.poisson.pmf(
#         dish_indices,
#         mu=new_running_harmonic_sum)
#     print(10)


# draw num_samples IBP(alpha) samples
ibp_samples_by_alpha = {}
for alpha in alphas:
    ibp_empirics_path = os.path.join(exp_dir, f'ibp_sample_{alpha}.npy')
    if os.path.isfile(ibp_empirics_path):
        customers_dishes_samples = np.load(ibp_empirics_path)
        print(f'Loaded samples for {ibp_empirics_path}')
        assert customers_dishes_samples.shape[0] == num_samples
        assert customers_dishes_samples.shape[1] == T
    else:
        customers_dishes_samples = vectorized_sample_sequence_from_ibp(
            T=np.full(shape=num_samples, fill_value=T),
            alpha=np.full(shape=num_samples, fill_value=alpha))
        customers_dishes_samples = np.stack(customers_dishes_samples)
        np.save(arr=customers_dishes_samples, file=ibp_empirics_path)
        print(f'Generated samples for {ibp_empirics_path}')
    ibp_samples_by_alpha[alpha] = customers_dishes_samples


# import matplotlib.pyplot as plt
#
# plt.imshow(ibp_samples_by_alpha[10.01][0, :, :50])
# plt.ylabel('Customer Index')
# plt.xlabel('Dish Index')
# plt.show()


def construct_analytical_customers_dishes(T, alpha):
    # shape: (number of customers, number of dishes)
    # heuristic: 10 * expected number
    assert alpha > 0
    alpha = float(alpha)

    max_dishes = int(2 * alpha * np.sum(1 / (1 + np.arange(T))))
    analytical_customers_dishes = np.zeros(shape=(T + 1, max_dishes + 1))
    analytical_customer_dishes_running_sum = np.zeros(shape=(T + 1, max_dishes + 1))
    dish_indices = np.arange(max_dishes + 1)

    # customer 1 samples only new dishes
    new_dishes_rate = alpha / 1
    analytical_customers_dishes[1, :] = np.cumsum(scipy.stats.poisson.pmf(dish_indices[::-1], mu=new_dishes_rate))[::-1]
    analytical_customer_dishes_running_sum[1, :] = analytical_customers_dishes[1, :]
    total_dishes_rate_running_sum = new_dishes_rate

    # all subsequent customers sample new dishes
    for customer_num in range(2, T + 1):
        analytical_customers_dishes[customer_num, :] = analytical_customer_dishes_running_sum[customer_num - 1, :] / customer_num
        new_dishes_rate = alpha / customer_num
        cdf_lambda_t_minus_1 = scipy.stats.poisson.cdf(dish_indices, mu=total_dishes_rate_running_sum)
        cdf_lambda_t = scipy.stats.poisson.cdf(dish_indices, mu=total_dishes_rate_running_sum + new_dishes_rate)
        cdf_diff = np.subtract(cdf_lambda_t_minus_1, cdf_lambda_t)
        analytical_customers_dishes[customer_num, :] += cdf_diff
        analytical_customer_dishes_running_sum[customer_num, :] = \
            np.add(analytical_customer_dishes_running_sum[customer_num - 1, :],
                   analytical_customers_dishes[customer_num, :])
        total_dishes_rate_running_sum += new_dishes_rate

    plt.imshow(analytical_customers_dishes[1:, 1:])
    # plt.ylabel('Customer Index')
    # plt.xlabel('Dish Index')
    plt.show()

    return analytical_customers_dishes[1:, 1:], analytical_customer_dishes_running_sum[1:, 1:]


analytical_customer_dishes_by_alpha = {}
customer_seating_probs_by_alpha = {}
for alpha in alphas:
    ibp_analytics_path = os.path.join(exp_dir, f'ibp_analytics_{alpha}.npz')
    if os.path.isfile(ibp_analytics_path):
        npz_file = np.load(ibp_analytics_path)
        analytical_customers_dishes = npz_file['analytical_customers_dishes']
        analytical_customer_dishes_running_sum = npz_file['analytical_customer_dishes_running_sum']
    else:
        analytical_customers_dishes, analytical_customer_dishes_running_sum = construct_analytical_customers_dishes(
            T=T,
            alpha=alpha)
        np.savez(analytical_customers_dishes=analytical_customers_dishes,
                 analytical_customer_dishes_running_sum=analytical_customer_dishes_running_sum,
                 file=ibp_analytics_path)
    analytical_customer_dishes_by_alpha[alpha] = analytical_customers_dishes

table_distributions_by_alpha_by_T = {}
cmap = plt.get_cmap('jet_r')
for alpha in alphas:
    ibp_samples = np.mean(ibp_samples_by_alpha[alpha], axis=0)
    max_dish_idx = np.max(np.argmin(ibp_samples != 0, axis=1))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharex=True, sharey=True)

    sns.heatmap(
        data=ibp_samples[:, :max_dish_idx],
        cbar_kws=dict(label='P(Dish)'),
        cmap='jet',
        # mask=ibp_samples[:, :max_dish_idx] == 0,
        vmin=0.,
        vmax=1.,
        ax=axes[0])
    sns.heatmap(
        data=analytical_customer_dishes_by_alpha[alpha][:, :max_dish_idx],
        cbar_kws=dict(label='P(Dish)'),
        cmap='jet',
        # mask=analytical_customer_dishes_by_alpha[alpha][:, :max_dish_idx] == 0,
        vmin=0.,
        vmax=1.,
        ax=axes[1])

    # # https://stackoverflow.com/questions/43805821/matplotlib-add-colorbar-to-non-mappable-object
    # norm = mpl.colors.Normalize(vmin=1, vmax=T)
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # # sm.set_array([])
    # colorbar = plt.colorbar(sm,
    #                         ticks=np.arange(1, T + 1, 5),
    #                         # boundaries=np.arange(-0.05, T + 0.1, .1)
    #                         )
    # colorbar.set_label('Number of Customers')
    # plt.title(fr'Chinese Restaurant Table Distribution ($\alpha$={alpha})')
    axes[0].set_title(rf'Monte Carlo Estimate ($\alpha$={alpha})')
    axes[0].set_ylabel(r'Customer Index')
    axes[0].set_xlabel(r'Dish Index')
    axes[1].set_title(rf'Analytical Prediction ($\alpha$={alpha})')
    axes[1].set_xlabel(r'Dish Index')
    plt.savefig(os.path.join(plot_dir, f'empirical_ibp={alpha}.png'),
                bbox_inches='tight',
                dpi=300)
    plt.show()
    plt.close()

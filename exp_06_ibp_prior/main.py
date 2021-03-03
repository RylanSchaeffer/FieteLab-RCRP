import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

from utils.data import vectorized_sample_sequence_from_ibp, sample_sequence_from_ibp


np.random.seed(1)

exp_dir = 'exp_06_ibp_prior'
plot_dir = os.path.join(exp_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)

T = 50  # max time
num_samples = 5000  # number of samples to draw from IBP(alpha)
alphas = [1.1, 10.01, 15.51, 30.03]  # CRP parameter

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


table_nums = 1 + np.arange(T)
table_distributions_by_alpha_by_T = {}
cmap = plt.get_cmap('jet_r')
for alpha in alphas:
    ibp_samples = np.mean(ibp_samples_by_alpha[alpha], axis=0)
    sns.heatmap(
        data=ibp_samples,
        cbar_kws=dict(label='P(Dish)'),
        cmap='jet',
        mask=ibp_samples == 0,
        vmin=0.,
        vmax=1.)


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
    plt.ylabel(r'Customer Index')
    plt.xlabel(r'Dish Index')
    plt.savefig(os.path.join(plot_dir, f'empirical_ibp={alpha}.png'),
                bbox_inches='tight',
                dpi=300)
    plt.close()


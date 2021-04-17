import matplotlib.pyplot as plt
import numpy as np
import os

# common plotting functions
from utils.plot import *


def plot_num_clusters_by_num_docs(obs_indices,
                                  empiric_num_unique_clusters_by_end_index,
                                  fit_num_unique_clusters_by_end_index,
                                  fitted_alpha,
                                  plot_dir):
    plt.plot(obs_indices,
             empiric_num_unique_clusters_by_end_index,
             label='Empiric')
    plt.plot(obs_indices,
             fit_num_unique_clusters_by_end_index,
             label=f'Fit (alpha = {np.round(fitted_alpha, 2)})')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'fitted_alpha.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


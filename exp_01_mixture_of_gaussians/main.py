import numpy as np
import os

from exp_01_mixture_of_gaussians.data import generate_mixture_of_gaussians, \
    sample_sequence_from_mixture_of_gaussians
from exp_01_mixture_of_gaussians.plot import plot_sample_from_mixture_of_gaussians


def main():
    plot_dir = 'exp_01_mixture_of_gaussians/plots'
    os.makedirs(plot_dir, exist_ok=True)

    np.random.seed(1)

    # sample data
    mixture_of_gaussians = generate_mixture_of_gaussians()
    class_seq, gaussian_samples_seq = sample_sequence_from_mixture_of_gaussians(
        mixture_of_gaussians=mixture_of_gaussians,
        seq_len=500)
    plot_sample_from_mixture_of_gaussians(
        class_seq=class_seq,
        gaussian_samples_seq=gaussian_samples_seq,
        plot_dir=plot_dir)

    # inference


if __name__ == '__main__':
    main()

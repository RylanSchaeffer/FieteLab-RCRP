import numpy as np




if __name__ == '__main__':

    np.random.seed(seed=1)

    mixture_of_gaussians = generate_mixture_of_gaussians()
    class_seq, gaussian_samples_seq = sample_sequence_from_mixture_of_gaussians(
        mixture_of_gaussians=mixture_of_gaussians,
        seq_len=500)

    import exp_01_mixture_of_gaussians.plot
    exp_01_mixture_of_gaussians.plot.plot_sample_from_mixture_of_gaussians(
        assigned_table_seq=class_seq,
        gaussian_samples_seq=gaussian_samples_seq)


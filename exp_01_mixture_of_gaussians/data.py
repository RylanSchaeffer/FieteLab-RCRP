import numpy as np


def generate_mixture_of_gaussians(num_gaussians=3,
                                  gaussian_dim=2,
                                  gaussian_mean_prior_cov_scaling=3.,
                                  gaussian_cov_scaling=0.3):

    # sample Gaussians' means from prior = N(0, rho * I)
    means = np.random.multivariate_normal(
        mean=np.zeros(gaussian_dim),
        cov=gaussian_mean_prior_cov_scaling * np.eye(gaussian_dim),
        size=num_gaussians)

    # all Gaussians have same covariance
    cov = gaussian_cov_scaling * np.eye(gaussian_dim)
    covs = np.repeat(cov[np.newaxis, :, :],
                     repeats=num_gaussians,
                     axis=0)

    # list of tuples of (mean, cov)
    mixture_of_gaussians = list(zip(means, covs))

    return mixture_of_gaussians


def sample_sequence_from_mixture_of_gaussians(mixture_of_gaussians,
                                              seq_len=100):

    num_gaussians = len(mixture_of_gaussians)
    class_seq = np.random.choice(np.arange(num_gaussians),
                                 replace=True,
                                 size=seq_len)

    # create sequence of (mean_t, cov_t) using sequence of sampled classes
    gaussian_params_seq = [mixture_of_gaussians[tth_class] for tth_class in class_seq]

    # create sequence of Gaussian samples from (mean_t, cov_t)
    gaussian_samples_seq = np.array([
        np.random.multivariate_normal(mean=mean_t, cov=cov_t)
        for mean_t, cov_t in gaussian_params_seq])

    return class_seq, gaussian_samples_seq


if __name__ == '__main__':

    np.random.seed(seed=1)

    mixture_of_gaussians = generate_mixture_of_gaussians()
    class_seq, gaussian_samples_seq = sample_sequence_from_mixture_of_gaussians(
        mixture_of_gaussians=mixture_of_gaussians,
        seq_len=500)

    import exp_01_mixture_of_gaussians.plot
    exp_01_mixture_of_gaussians.plot.plot_sample_from_mixture_of_gaussians(
        class_seq=class_seq,
        gaussian_samples_seq=gaussian_samples_seq)


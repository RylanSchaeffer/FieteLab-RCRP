import joblib
import pandas as pd
import scipy
import scipy.special

from exp_02_mixture_of_unigrams.plot import *

from utils.data import sample_sequence_from_mixture_of_unigrams
from utils.helpers import assert_no_nan_no_inf
from utils.inference_mix_of_unigram import bayesian_recursion, expectation_maximization, sampling_hmc_gibbs, \
    stochastic_variational_inference
from utils.metrics import score_predicted_clusters


def main():
    plot_dir = 'exp_02_mixture_of_unigrams/plots'
    os.makedirs(plot_dir, exist_ok=True)
    np.random.seed(1)

    num_datasets = 5
    inference_algs_results_by_dataset = {}
    sampled_mog_results_by_dataset = {}
    # generate lots of datasets and record performance
    for dataset_idx in range(num_datasets):
        dataset_dir = os.path.join(plot_dir, f'dataset={dataset_idx}')
        dataset_results_path = os.path.join(dataset_dir, 'dataset_results.joblib')

        if os.path.isfile(dataset_results_path):
            # load from disk if exists
            dataset_results = joblib.load(dataset_results_path)
        else:
            # otherwise, generate anew
            dataset_inference_algs_results, dataset_sampled_mog_results = \
                run_one_dataset(plot_dir=dataset_dir)
            dataset_results = dict(
                dataset_inference_algs_results=dataset_inference_algs_results,
                dataset_sampled_mog_results=dataset_sampled_mog_results,
            )
            joblib.dump(dataset_results, dataset_results_path)

            # delete variables from memory and perform fresh read from disk
            del dataset_inference_algs_results, dataset_sampled_mog_results
            del dataset_results
            dataset_results = joblib.load(dataset_results_path)

        inference_algs_results_by_dataset[dataset_idx] = dataset_results['dataset_inference_algs_results']
        sampled_mog_results_by_dataset[dataset_idx] = dataset_results['dataset_sampled_mog_results']

    plot_inference_algs_comparison(
        plot_dir=plot_dir,
        inference_algs_results_by_dataset=inference_algs_results_by_dataset,
        sampled_mog_results_by_dataset=sampled_mog_results_by_dataset)


def run_one_dataset(plot_dir,
                    dp_concentration_param: float = 4.6,
                    prior_over_topic_parameters: float = 0.3):
    # sample data
    sampled_mou_results = sample_sequence_from_mixture_of_unigrams(
        seq_len=150,
        num_topics=7,
        unigram_params=dict(dp_concentration_param=dp_concentration_param,
                            prior_over_topic_parameters=prior_over_topic_parameters))

    variational_bayes_results = run_and_plot_stochastic_variational_inference(
        sampled_mou_results=sampled_mou_results,
        plot_dir=plot_dir)

    bayesian_recursion_results = run_and_plot_bayesian_recursion(
        sampled_mou_results=sampled_mou_results,
        plot_dir=plot_dir)

    hmc_gibbs_5000_samples_results = run_and_plot_hmc_gibbs_sampling(
        sampled_mou_results=sampled_mou_results,
        plot_dir=plot_dir,
        num_samples=5000)

    hmc_gibbs_20000_samples_results = run_and_plot_hmc_gibbs_sampling(
        sampled_mou_results=sampled_mou_results,
        plot_dir=plot_dir,
        num_samples=20000)

    inference_algs_results = {
        # 'Bayesian Recursion': bayesian_recursion_results,
        'HMC-Gibbs (5k Samples)': hmc_gibbs_5000_samples_results,
        'HMC-Gibbs (20k Samples)': hmc_gibbs_20000_samples_results,
    }
    for num_iter in [15, 5, 1]:
        inference_algs_results[f'Expectation Maximization (Num Iter={num_iter})'] = \
            run_and_plot_expectation_maximization(
                sampled_mou_results=sampled_mou_results,
                plot_dir=plot_dir,
                num_iter=num_iter)

    return inference_algs_results, sampled_mou_results


def run_and_plot_bayesian_recursion(sampled_mou_results,
                                    plot_dir):
    def likelihood_fn(doc, parameters):
        # create new mean for new table, centered at that point
        # initialize possible new cluster with parameters matching doc word count
        # technically, this should be an assignment, but we can't permit zero values
        # so we initialized with a small positive value and add to that initial value
        # create new mean for new table, centered at that point
        parameters['topics_concentrations'] = np.vstack([
            parameters['topics_concentrations'],
            doc[np.newaxis, :] + parameters['epsilon']])

        # draw multinomial parameters
        # see https://en.wikipedia.org/wiki/Dirichlet_distribution#Conjugate_to_categorical/multinomial
        # however, again, we have floating point issues
        # add epsilon, and renormalize
        # TODO: we want the likelihood for every topic p(d_t|nhat, z_t = k) = Dirichlet-Multinomial(nhat_k)
        # nhat is the running pseudocounts for the Dirichlet distribution

        # Approach 1: Multinomial-Dirichlet
        words_in_doc = np.sum(doc)
        log_numerator = np.log(words_in_doc) + np.log(scipy.special.beta(
            np.sum(parameters['topics_concentrations'], axis=1),
            words_in_doc))
        # shape (doc idx, vocab size)
        beta_terms = scipy.special.beta(
            parameters['topics_concentrations'],
            doc)
        log_x_times_beta_terms = np.add(
            np.log(beta_terms),
            np.log(doc))
        # beta numerically can create infs if x is 0, even though 0*Beta(., 0) should be 0
        # consequently, filter these out by setting equal to 0
        log_x_times_beta_terms[np.isnan(log_x_times_beta_terms)] = 0.
        # shape (doc idx, )
        log_denominator = np.sum(log_x_times_beta_terms, axis=1)
        assert_no_nan_no_inf(log_denominator)
        log_like = log_numerator - log_denominator
        assert_no_nan_no_inf(log_like)
        likelihoods = np.exp(log_like)

        # Approach 2: Sampling
        # multinomial_parameters = np.apply_along_axis(
        #     func1d=np.random.dirichlet,
        #     axis=1,
        #     arr=parameters['inferred_topic_dirichlet_parameters'][:doc_idx + 1, :])
        # multinomial_parameters += epsilon
        # multinomial_parameters = np.divide(multinomial_parameters,
        #                                    np.sum(multinomial_parameters, axis=1)[:, np.newaxis])
        # assert np.allclose(np.sum(multinomial_parameters, axis=1), 1)
        # like1 = scipy.stats.multinomial.pmf(x=doc, n=np.sum(doc), p=multinomial_parameters)

        # Approach 3: Dirichlet Mean
        # multinomial_parameters = np.divide(
        #     parameters['inferred_topic_dirichlet_parameters'][:doc_idx+1, :],
        #     np.sum(parameters['inferred_topic_dirichlet_parameters'][:doc_idx + 1, :], axis=1)[:, np.newaxis])
        # assert np.allclose(np.sum(multinomial_parameters, axis=1), 1)
        # likelihoods = scipy.stats.multinomial.pmf(x=doc, n=np.sum(doc), p=multinomial_parameters)

        assert_no_nan_no_inf(likelihoods)

        return likelihoods, parameters

    def update_parameters_fn(doc,
                             table_assignment_posteriors_running_sum,
                             table_assignment_posterior,
                             parameters):
        # update parameters based on Dirichlet prior and Multinomial likelihood
        # floating point errors are common here because such small values!
        probability_prefactor = np.divide(table_assignment_posterior,
                                          table_assignment_posteriors_running_sum)
        probability_prefactor[np.isnan(probability_prefactor)] = 0.
        assert_no_nan_no_inf(probability_prefactor)

        inferred_topic_parameters_updates = np.multiply(
            probability_prefactor[:, np.newaxis],
            doc)
        # multiply the last parameter update by 0 so we don't create a point and double count that obs
        inferred_topic_parameters_updates[-1, :] *= 0.
        assert_no_nan_no_inf(inferred_topic_parameters_updates)
        parameters['topics_concentrations'] += inferred_topic_parameters_updates
        assert_no_nan_no_inf(parameters['topics_concentrations'])

        return parameters

    alphas = 0.01 + np.arange(0., 10.01, 0.5)
    bayesian_recursion_plot_dir = os.path.join(plot_dir, 'bayesian_recursion')
    os.makedirs(bayesian_recursion_plot_dir, exist_ok=True)
    num_clusters_by_alpha = {}
    scores_by_alpha = {}
    for alpha in alphas:
        bayesian_recursion_results = bayesian_recursion(
            docs=sampled_mou_results['doc_samples_seq'],
            alpha=alpha,
            likelihood_fn=likelihood_fn,
            update_parameters_fn=update_parameters_fn)

        # record scores
        scores, pred_cluster_labels = score_predicted_clusters(
            true_cluster_labels=sampled_mou_results['assigned_table_seq'],
            table_assignment_posteriors=bayesian_recursion_results['table_assignment_posteriors'])
        scores_by_alpha[alpha] = scores

        # count number of clusters
        num_clusters_by_alpha[alpha] = len(np.unique(pred_cluster_labels))

        # plot_inference_results(
        #     sampled_mou_results=sampled_mou_results,
        #     inference_results=bayesian_recursion_results,
        #     inference_alg='bayesian_recursion_alpha={:.2f}'.format(alpha),
        #     plot_dir=bayesian_recursion_plot_dir)

        print('Finished Bayesian recursion alpha={:.2f}'.format(alpha))

    bayesian_recursion_results = dict(
        num_clusters_by_param=num_clusters_by_alpha,
        scores_by_param=pd.DataFrame(scores_by_alpha).T,
    )

    return bayesian_recursion_results


def run_and_plot_hmc_gibbs_sampling(sampled_mou_results,
                                    plot_dir,
                                    num_samples: int = 5000):
    hmc_gibbs_sampling_plot_dir = os.path.join(plot_dir,
                                               f'hmc_gibbs_sampling_nsamples={num_samples}')
    os.makedirs(hmc_gibbs_sampling_plot_dir, exist_ok=True)
    num_clusters_by_alpha = {}
    scores_by_alpha = {}
    alphas = np.arange(0.01, 5.01, 0.2)
    for alpha in alphas:
        sampling_hmc_gibbs_results = sampling_hmc_gibbs(
            docs=sampled_mou_results['doc_samples_seq'],
            num_samples=num_samples,
            alpha=alpha, )

        # # score clusters
        scores, pred_cluster_labels = score_predicted_clusters(
            true_cluster_labels=sampled_mou_results['assigned_table_seq'],
            table_assignment_posteriors=sampling_hmc_gibbs_results['table_assignment_posteriors'])
        scores_by_alpha[alpha] = scores

        # count number of clusters
        num_clusters_by_alpha[alpha] = len(np.unique(pred_cluster_labels))

        # plot_inference_results(
        #     sampled_mou_results=sampled_mou_results,
        #     inference_results=sampling_hmc_gibbs_results,
        #     inference_alg='hmc_gibbs={:.2f}'.format(alpha),
        #     plot_dir=hmc_gibbs_sampling_plot_dir)

    sampling_hmc_gibbs_results = dict(
        num_clusters_by_param=num_clusters_by_alpha,
        scores_by_param=pd.DataFrame(scores_by_alpha).T
    )

    return sampling_hmc_gibbs_results


def run_and_plot_expectation_maximization(sampled_mou_results: dict,
                                          plot_dir: str,
                                          num_iter: int):
    alphas = 0.01 + np.arange(0., 10.01, 0.5)
    expectation_maximization_plot_dir = os.path.join(plot_dir, 'expectation_maximization')
    os.makedirs(expectation_maximization_plot_dir, exist_ok=True)
    num_clusters_by_alpha = {}
    scores_by_alpha = {}
    for alpha in alphas:
        expectation_maximization_results = expectation_maximization(
            docs=sampled_mou_results['doc_samples_seq'],
            alpha=alpha,
            num_iter=num_iter)

        # record scores
        scores, pred_cluster_labels = score_predicted_clusters(
            true_cluster_labels=sampled_mou_results['assigned_table_seq'],
            table_assignment_posteriors=expectation_maximization_results['table_assignment_posteriors'])
        scores_by_alpha[alpha] = scores

        # count number of clusters
        num_clusters_by_alpha[alpha] = len(np.unique(pred_cluster_labels))

        # plot_inference_results(
        #     sampled_mou_results=sampled_mou_results,
        #     inference_results=expectation_maximization_results,
        #     inference_alg='bayesian_recursion_alpha={:.2f}'.format(alpha),
        #     plot_dir=expectation_maximization_plot_dir)

        print('Finished Expectation Maximization alpha={:.2f}'.format(alpha))

    expectation_maximization_results = dict(
        num_clusters_by_param=num_clusters_by_alpha,
        scores_by_param=pd.DataFrame(scores_by_alpha).T,
    )

    return expectation_maximization_results


def run_and_plot_stochastic_variational_inference(sampled_mou_results,
                                                  plot_dir,
                                                  num_steps: int = 5000):
    alphas = 0.01 + np.arange(0., 5.01, 0.1)
    svi_plot_dir = os.path.join(plot_dir, 'svi')
    os.makedirs(svi_plot_dir, exist_ok=True)
    num_clusters_by_alpha = {}
    scores_by_alpha = {}
    for alpha in alphas:
        stochastic_variational_inference_results = stochastic_variational_inference(
            docs=sampled_mou_results['doc_samples_seq'],
            num_steps=num_steps,
            alpha=alpha)

        # score clusters
        scores, pred_cluster_labels = score_predicted_clusters(
            true_cluster_labels=sampled_mou_results['assigned_table_seq'],
            table_assignment_posteriors=stochastic_variational_inference_results['table_assignment_posteriors'])
        scores_by_alpha[alpha] = scores

        # count number of clusters
        num_clusters_by_alpha[alpha] = len(np.unique(pred_cluster_labels))

        # plot_inference_results(
        #     sampled_mog_results=sampled_mou_results,
        #     inference_results=stochastic_variational_inference_results,
        #     inference_alg='variational_bayes={:.2f}'.format(alpha),
        #     plot_dir=svi_plot_dir)

        print('Finished Variational Bayes alpha={:.2f}'.format(alpha))

    stochastic_variational_inference_results = dict(
        num_clusters_by_param=num_clusters_by_alpha,
        scores_by_param=pd.DataFrame(scores_by_alpha).T,
    )

    return stochastic_variational_inference_results


if __name__ == '__main__':
    main()

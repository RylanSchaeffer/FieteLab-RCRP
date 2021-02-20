import numpy as np
import matplotlib.pyplot as plt
import os
import scipy as sp
import scipy.special
import scipy.stats
import scipy.optimize

from utils.helpers import assert_no_nan_no_inf
import utils.metrics

np.random.seed(0)

plot_dir = os.path.join('exp_04_mixture_of_unigrams', 'plots')
os.makedirs(plot_dir, exist_ok=True)


def run_dp_expt(alpha):
    # alpha = 2
    beta = 0.8
    vocab_dim = 10  # vocabulary size
    doc_len = 120  # len of each document
    num_docs = 250  # num_docs

    # could truncate, if we want
    num_topics = num_docs  # 50 * np.log(num_docs).astype('int')

    beta_samples = np.random.beta(1, alpha, size=num_topics)
    stick_weights = np.zeros(shape=num_topics, dtype=np.float64)
    remaining_stick = 1.
    for doc_idx in range(num_topics):
        stick_weights[doc_idx] = beta_samples[doc_idx] * remaining_stick
        remaining_stick *= (1. - beta_samples[doc_idx])
    # ordinarily, we'd set the last stick weight so that the total stick weights sum to 1.
    # However, floating-point errors can make this last value negative (yeah, I was surprised too)
    # so I only do this if the sum of the weights isn't sufficiently close to 1
    try:
        assert np.allclose(np.sum(stick_weights), 1.)
    except AssertionError:
        stick_weights[-1] = 1 - np.sum(stick_weights[:-1])
    assert np.alltrue(stick_weights >= 0.)
    assert np.allclose(np.sum(stick_weights), 1.)

    topic_parameters = np.zeros((num_docs, vocab_dim))
    for doc_idx in range(num_docs):
        gam_t = np.random.gamma(beta, size=vocab_dim)
        topic_parameters[doc_idx, :] = gam_t / np.sum(gam_t)

    # draw a Sample of num_docs documents each of size doc_len
    docs = np.zeros(shape=(num_docs, vocab_dim))
    topic_id = np.random.multinomial(1, stick_weights, size=num_docs)
    topic_id_max = np.argmax(topic_id, axis=1)
    num_topics_in_sample = len(np.unique(topic_id_max))

    # plot comparison between true doc probabilities and sampled doc probabilities
    """
    plt.plot(np.arange(num_topics), stick_weights, label='Topic Probabilities')
    plt.plot(np.arange(num_topics), np.sum(topic_id, axis=0) / num_docs, label='Sampled Topic Probabilities')
    plt.title(rf'$\alpha$={alpha}')
    plt.xlabel('Topic Index')
    plt.ylabel('Topic Probability Mass')
    plt.xscale('log')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f'topic_sample_probs_for_a={alpha}.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    """

    print(f'Number of unique topic_parameters sampled: {num_topics_in_sample}')
    assert topic_id_max.shape == (num_docs,)
    for doc_idx in range(num_docs):
        topic_idx = topic_id_max[doc_idx]
        arr = np.random.multinomial(doc_len, topic_parameters[topic_idx, :], size=1)
        docs[doc_idx, :] = arr

    """
    plt.imshow(topic_parameters.num_docs)
    plt.title('Topics')
    plt.xlabel(r'$k$')
    plt.ylabel(r'Words')
    plt.show()
    """

    table_assignment_priors = np.zeros((num_docs, num_topics), dtype=np.float64)
    table_assignment_priors[0, 0] = 1.
    table_assignment_posteriors = np.zeros((num_docs, num_topics), dtype=np.float64)
    table_assignment_posteriors_running_sum = np.zeros_like(table_assignment_posteriors)
    num_table_posteriors = np.zeros(shape=(num_docs, num_docs))

    epsilon = 1e-5
    inferred_topic_dirichlet_parameters = np.full(shape=(num_topics, vocab_dim),
                                                  fill_value=epsilon)  # need small value to ensure non-zero

    for doc_idx, doc in enumerate(docs):

        # initialize possible new cluster with parameters matching doc word count
        # technically, this should be an assignment, but we can't permit zero values
        # so we initialized with a small positive value and add to that initial value
        inferred_topic_dirichlet_parameters[doc_idx, :] += doc

        # draw multinomial parameters
        # see https://en.wikipedia.org/wiki/Dirichlet_distribution#Conjugate_to_categorical/multinomial
        # however, again, we have floating point issues
        # add epsilon, and renormalize
        # TODO: we want the likelihood for every topic p(d_t|nhat, z_t = k) = Dirichlet-Multinomial(nhat_k)
        # nhat is the running pseudocounts for the Dirichlet distribution

        # Approach 1: Multinomial-Dirichlet
        words_in_doc = np.sum(doc)
        log_numerator = np.log(words_in_doc) + np.log(scipy.special.beta(
            np.sum(inferred_topic_dirichlet_parameters[:doc_idx + 1, :], axis=1),
            words_in_doc))
        # shape (doc idx, vocab size)
        beta_terms = scipy.special.beta(
            inferred_topic_dirichlet_parameters[:doc_idx + 1, :],
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
        like = np.exp(log_like)

        # Approach 2: Sampling
        # multinomial_parameters = np.apply_along_axis(
        #     func1d=np.random.dirichlet,
        #     axis=1,
        #     arr=inferred_topic_dirichlet_parameters[:doc_idx + 1, :])
        # multinomial_parameters += epsilon
        # multinomial_parameters = np.divide(multinomial_parameters,
        #                                    np.sum(multinomial_parameters, axis=1)[:, np.newaxis])
        # assert np.allclose(np.sum(multinomial_parameters, axis=1), 1)
        # like1 = scipy.stats.multinomial.pmf(x=doc, n=np.sum(doc), p=multinomial_parameters)

        # Approach 3: Dirichlet Mean
        # multinomial_parameters = np.divide(
        #     inferred_topic_dirichlet_parameters[:doc_idx+1, :],
        #     np.sum(inferred_topic_dirichlet_parameters[:doc_idx + 1, :], axis=1)[:, np.newaxis])
        # assert np.allclose(np.sum(multinomial_parameters, axis=1), 1)
        # like = scipy.stats.multinomial.pmf(x=doc, n=np.sum(doc), p=multinomial_parameters)

        assert_no_nan_no_inf(like)
        if doc_idx == 0:
            # first customer has to go at first table
            table_assignment_posterior = np.array([1.])
            table_assignment_posteriors[doc_idx, 0] = table_assignment_posterior
            num_table_posteriors[0, 0] = 1.
            # num_topics_idx += 1
        else:
            table_assignment_prior = np.copy(table_assignment_posteriors_running_sum[doc_idx - 1, :len(like)])
            # we don't subtract 1 because Python uses 0-based indexing
            assert np.allclose(np.sum(table_assignment_prior), doc_idx)
            # right shift table posterior by 1
            table_assignment_prior[1:] += alpha * np.copy(num_table_posteriors[doc_idx - 1, :len(like) - 1])
            table_assignment_prior /= (alpha + doc_idx)
            assert np.allclose(np.sum(table_assignment_prior), 1.)

            # record latent prior
            table_assignment_priors[doc_idx, :len(table_assignment_prior)] = table_assignment_prior

            unnormalized_table_assignment_posterior = np.multiply(like, table_assignment_prior)
            table_assignment_posterior = unnormalized_table_assignment_posterior / np.sum(
                unnormalized_table_assignment_posterior)
            assert np.allclose(np.sum(table_assignment_posterior), 1.)

            # # truncate to speed up. otherwise too slow.
            # if table_assignment_posterior[-1] > 1e-10:
            #     num_topics_idx += 1
            # else:
            #     # assign posterior to zero
            #     table_assignment_posterior[-1] = 0
            #     # reset this parameter
            #     inferred_topic_dirichlet_parameters[num_topics_idx, :] = epsilon

            # record latent posterior
            table_assignment_posteriors[doc_idx, :len(table_assignment_posterior)] = table_assignment_posterior

            # update posterior over number of tables
            for k1, p_z_t_equals_k1 in enumerate(table_assignment_posteriors[doc_idx, :doc_idx + 1]):
                for k2, p_prev_num_tables_equals_k2 in enumerate(num_table_posteriors[doc_idx - 1, :doc_idx + 1]):
                    # exclude cases of placing customer at impossible table
                    if k1 > k2 + 1:
                        continue
                    # customer allocated to previous table
                    elif k1 <= k2:
                        num_table_posteriors[doc_idx, k2] += p_z_t_equals_k1 * p_prev_num_tables_equals_k2
                    # create new table
                    elif k1 == k2 + 1:
                        num_table_posteriors[doc_idx, k1] += p_z_t_equals_k1 * p_prev_num_tables_equals_k2
                    else:
                        raise ValueError
            num_table_posteriors[doc_idx, :] /= np.sum(num_table_posteriors[doc_idx, :])
            assert np.allclose(np.sum(num_table_posteriors[doc_idx, :]), 1.)

        # update running sum of posteriors
        table_assignment_posteriors_running_sum[doc_idx, :] = table_assignment_posteriors_running_sum[doc_idx - 1, :] + \
                                                              table_assignment_posteriors[doc_idx, :]
        assert np.allclose(np.sum(table_assignment_posteriors_running_sum[doc_idx, :]),
                           doc_idx + 1)

        # update parameters, except for the first observation
        if doc_idx == 0:
            continue

        # update parameters based on Dirichlet prior and Multinomial likelihood
        # floating point errors are common here because such small values!
        probability_prefactor = np.divide(table_assignment_posterior,
                                          table_assignment_posteriors_running_sum[doc_idx,
                                          :len(table_assignment_posterior)])
        probability_prefactor[np.isnan(probability_prefactor)] = 0.
        assert_no_nan_no_inf(probability_prefactor)

        inferred_topic_parameters_updates = np.multiply(
            probability_prefactor[:, np.newaxis],
            doc)
        inferred_topic_dirichlet_parameters[:doc_idx + 1, :] += inferred_topic_parameters_updates

    # Blake's equivalent function
    # scores_results = score_util.score_predicted_clusters(
    #     table_assignment_posteriors=soft_assign,
    #     true_cluster_labels=topic_id_max)

    scores_results, pred_cluster_labels = utils.metrics.score_predicted_clusters(
        table_assignment_posteriors=table_assignment_posteriors,
        true_cluster_labels=topic_id_max)

    """
    plt.subplot(121)
    plt.imshow( table_assignment_priors[0:,0:50], cmap = 'rainbow')
    plt.title('Rylan Assignment Posterior')
    plt.subplot(122)
    plt.imshow(topic_id[0:,0:50], cmap= 'rainbow')
    plt.title('True Topic Assignment')
    plt.savefig('assignment_fig_recursion.pdf')
    plt.show()
    """

    return scores_results


def EM(alpha):
    beta = 0.8
    V = 6  # vocabulary size
    M = 50  # len of each document
    N_doc = 400  # num_doc
    T = N_doc
    # T = N_doc
    np.random.seed(0)

    num_topic = 50 * np.log(T).astype('int')
    stick_weights = [np.random.beta(1, alpha)]
    for t in range(1, num_topic):
        ans_t = np.random.beta(1, alpha)
        for i in range(t):
            ans_t *= (1 - stick_weights[i])
        stick_weights += [ans_t]

    stick_weights = np.array(stick_weights)
    stick_weights = stick_weights / stick_weights.sum()
    print(stick_weights)

    topics = np.zeros((T, V))
    for t in range(T):
        gam_t = np.random.gamma(beta, size=V)
        topics[t, :] = gam_t / np.sum(gam_t)

    print(topics[0, :])

    # draw a Sample of T documents each of size M
    docs = np.zeros((N_doc, V))
    topic_id = np.random.multinomial(1, stick_weights, size=N_doc)
    topic_id_max = np.argmax(topic_id, axis=1)
    for t in range(N_doc):
        k = topic_id_max[t]
        arr = np.random.multinomial(M, topics[k, :], size=1)
        docs[t, :] = arr

    print(docs)

    num_topics = T
    P = np.zeros((num_topics, V))
    for k in range(num_topics):
        arr = np.random.gamma(beta, size=V)
        arr = arr / arr.sum()
        P[k, :] = arr

    latent_post = np.zeros((N_doc, num_topics))
    hard_assign = np.zeros(N_doc)
    # offline EM
    errs = []
    err_topic = []

    for n in range(20):
        # num_docs x num_topics
        log_like = docs @ np.log(P).T
        like_norm = np.exp(log_like)
        soft_assign = like_norm / like_norm.sum(axis=1)[:, np.newaxis]  # normalized distribution

        # import scipy.stats
        # like_scipy = scipy.stats.multinomial.pmf(x=docs[0, :],
        #                                          n=np.sum(docs[0, :]).astype(np.int),
        #                                          p=P[0, :])

        counts_soft = soft_assign.T @ docs
        P = counts_soft + beta
        P = P / P.sum(axis=1)[:, np.newaxis]

    res = score_util.score_predicted_clusters(soft_assign, topic_id_max)
    return res


alpha_vals = np.linspace(0.2, 50, 10)
all_res = [run_dp_expt(alpha) for alpha in alpha_vals]
# all_res_EM = [EM(alpha) for alpha in alpha_vals]
e1 = [r['Normalized Mutual Info Score'] for r in all_res]
# e1_EM = [r['Normalized Mutual Info Score'] for r in all_res_EM]
plt.plot(alpha_vals, e1, label='Bayesian Recursion')
# plt.plot(alpha_vals, e1_EM, label='EM (Offline)')
plt.xlabel(r'$\alpha$', fontsize=20)
plt.ylabel(r'NMI', fontsize=20)
plt.title('Topic Modeling Normalized Information', fontsize=20)
plt.legend()
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig('NMI_DP_EM_compare.pdf')
plt.show()

e1 = [r['Adjusted Mutual Info Score'] for r in all_res]
# e1_EM = [r['Adjusted Mutual Info Score'] for r in all_res_EM]
plt.plot(alpha_vals, e1, label='Bayesian Recursion')
# plt.plot(alpha_vals, e1_EM, label='EM (Offline)')
plt.xlabel(r'$\alpha$', fontsize=20)
plt.title('Topic Modeling Normalized Information')
plt.ylabel(r'AMI', fontsize=20)
# plt.ylim([0, 1])
plt.legend()
plt.tight_layout()
plt.savefig('AMI_DP_EM_compare.pdf')
plt.show()

"""
print("ONLINE DP ALGO SCORES")
print(res)

num_topic_vals = [5, 10, 50, 100]
for num_topics in num_topic_vals:
    all_soft_assign = np.zeros((N_doc, num_topics))
    P_online = np.zeros((num_topics, V))
    for k in range(num_topics):
        arr = np.random.gamma(beta, size=V)
        arr = arr/arr.sum()
        P_online[k,:] =arr

    # online EM
    for t in range(N_doc):
        log_like = np.log(P_online) @ docs[t,:] # likelihood over topics for document t
        soft_assign = np.exp(log_like)
        soft_assign = soft_assign/np.sum(soft_assign)
        all_soft_assign[t,:] = soft_assign
        counts = all_soft_assign[0:t,:].T @ docs[0:t,:]
        P_online = counts + beta
        P_online = P_online / P_online.sum(axis=1)[:,np.newaxis]

    res = score_util.score_predicted_clusters(all_soft_assign, topic_id_max)
    print("ONLINE EM SCORE")
    print(res)

plt.subplot(121)
plt.imshow( np.log(all_soft_assign[0:,0:100]), cmap = 'rainbow')
plt.title('EM Assignment Posterior')
plt.subplot(122)
plt.imshow(np.log(topic_id[0:,0:100]), cmap= 'rainbow')
plt.title('True Topic Assignment')
plt.savefig('EM_assignment.pdf')
plt.show()
"""

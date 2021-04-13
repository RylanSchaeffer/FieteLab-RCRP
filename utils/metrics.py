import numpy as np
from sklearn.metrics import adjusted_rand_score, rand_score, adjusted_mutual_info_score, \
    normalized_mutual_info_score


def score_predicted_clusters(table_assignment_posteriors, true_cluster_labels):
    # table assignment posteriors is square matrix
    # first dimension is num obs, second dimension is number clusters
    # (i, j) element is probability the ith observation belongs to jth cluster
    # true_cluster_labels: integer classes with shape (num obs, )

    pred_cluster_labels = np.argmax(table_assignment_posteriors,
                                    axis=1)

    rnd_score = rand_score(labels_pred=pred_cluster_labels,
                           labels_true=true_cluster_labels)

    adj_rnd_score = adjusted_rand_score(labels_pred=pred_cluster_labels,
                                        labels_true=true_cluster_labels)

    adj_mut_inf_score = adjusted_mutual_info_score(labels_pred=pred_cluster_labels,
                                                   labels_true=true_cluster_labels)

    norm_mut_inf_score = normalized_mutual_info_score(labels_pred=pred_cluster_labels,
                                                      labels_true=true_cluster_labels)

    scores_results = {
        'Rand Score': rnd_score,
        'Adjusted Rand Score': adj_rnd_score,
        'Adjusted Mutual Info Score': adj_mut_inf_score,
        'Normalized Mutual Info Score': norm_mut_inf_score,
    }

    return scores_results, pred_cluster_labels

# def compute_entropy(p):
#     entropy_p = np.multiply(p, np.log2(p))
#     # we want 0 * log(0) to be 0
#     entropy_p[np.isnan(entropy_p)] = 0
#     entropy_p = -np.sum(entropy_p)
#     return entropy_p
#
#
# def compute_mutual_information(p, q):
#     raise NotImplementedError
#
#
# def normalized_mutual_information(assigned_table_seq_one_hot,
#                                   table_posteriors_one_hot):
#     num_obs = assigned_table_seq_one_hot.shape[0]
#     assert assigned_table_seq_one_hot.shape == table_posteriors_one_hot.shape
#
#     p = np.sum(assigned_table_seq_one_hot, axis=0)
#     p = p / np.sum(p)
#
#     q = np.sum(table_posteriors_one_hot, axis=0),
#     q = q / np.sum(q)
#
#     entropy_p = compute_entropy(p)
#     entropy_q = compute_entropy(q)
#     mi_pq = compute_mutual_information(p, q)
#     return 2 * mi_pq / (entropy_p + entropy_q)

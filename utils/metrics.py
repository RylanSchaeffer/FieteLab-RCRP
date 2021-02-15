import numpy as np
from sklearn.feature_selection import mutual_info_classif


def compute_entropy(p):
    entropy_p = np.multiply(p, np.log2(p))
    # we want 0 * log(0) to be 0
    entropy_p[np.isnan(entropy_p)] = 0
    entropy_p = -np.sum(entropy_p)
    return entropy_p


def compute_mutual_information(p, q):
    raise NotImplementedError


def normalized_mutual_information(assigned_table_seq_one_hot,
                                  table_posteriors_one_hot):

    num_obs = assigned_table_seq_one_hot.shape[0]
    assert assigned_table_seq_one_hot.shape == table_posteriors_one_hot.shape

    p = np.sum(assigned_table_seq_one_hot, axis=0)
    p = p / np.sum(p)

    q = np.sum(table_posteriors_one_hot, axis=0),
    q = q / np.sum(q)

    entropy_p = compute_entropy(p)
    entropy_q = compute_entropy(q)
    mi_pq = compute_mutual_information(p, q)
    return 2 * mi_pq / (entropy_p + entropy_q)

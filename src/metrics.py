import numpy as np
import torch


def explanation_accuracy(ground_truth, explanation, k="auto"):
    """
    Compute the accuracy of an explanation compared to the ground truth.

    Parameters:
        ground_truth (np.ndarray): The ground truth explanation.
        explanation (np.ndarray): The explanation to evaluate.

    Returns:
        float: The accuracy of the explanation.
    """
    if explanation.shape != ground_truth.shape:
        raise ValueError(
            "The explanation and ground truth must have the same shape."
        )
    if type(explanation) is torch.Tensor:
        explanation = explanation.cpu().detach().numpy()
    if type(ground_truth) is torch.Tensor:
        ground_truth = ground_truth.cpu().detach().numpy()
    accuracy = []
    for row in range(ground_truth.shape[0]):
        if k == "auto":
            k_ = int(np.sum(ground_truth[row]))
        else:
            k_ = k
        if k_ == 0:
            continue
        # Sort the explanation by importance
        sorted_indices = np.argsort(explanation[row])[::-1]
        instance_explanation = np.zeros_like(explanation[row])
        instance_explanation[sorted_indices[:k_]] = 1

        # Compute the accuracy: intersection of topk method and ground truth divided by k
        instance_accuracy = (
            np.sum(ground_truth[row] * instance_explanation) / k_
        )
        accuracy.append(instance_accuracy)
    return np.mean(accuracy)


def dcg_score_matrix_p(importance_scores, relevance_matrix, p):
    """
    Compute the DCG scores at a given cutoff rank p.
    """
    importance_scores = np.array(importance_scores)
    relevance_matrix = np.array(relevance_matrix)
    importance_scores = importance_scores.squeeze()
    relevance_matrix = relevance_matrix.squeeze()
    assert (
        importance_scores.shape == relevance_matrix.shape
    ), "importance_scores and relevance_matrix must have the same shape"

    # Sort relevance based on importance scores
    if len(importance_scores.shape) == 1:
        importance_scores = importance_scores.reshape(1, -1)
        relevance_matrix = relevance_matrix.reshape(1, -1)

    sorted_indices = np.argsort(importance_scores, axis=1)[:, ::-1]
    sorted_relevance = np.take_along_axis(
        relevance_matrix, sorted_indices, axis=1
    )

    # Consider only the top p items
    sorted_relevance_p = sorted_relevance[:, :p]
    ranks = np.arange(1, p + 1)  # Ranks from 1 to p

    # Compute DCG scores
    dcg_scores = np.sum(sorted_relevance_p / np.log2(ranks + 1), axis=1)

    return dcg_scores


def idcg_score_matrix_p(relevance_matrix, p):
    """
    Compute the IDCG scores at a given cutoff rank p.
    """
    if len(relevance_matrix.shape) == 1:
        relevance_matrix = relevance_matrix.reshape(1, -1)
    relevance_matrix = np.array(relevance_matrix)
    sorted_relevance = np.sort(relevance_matrix, axis=1)[:, ::-1]

    # Consider only the top p items
    sorted_relevance_p = sorted_relevance[:, :p]
    ranks = np.arange(1, p + 1)  # Ranks from 1 to p

    # Compute IDCG scores
    idcg_scores = np.sum(sorted_relevance_p / np.log2(ranks + 1), axis=1)

    return idcg_scores


def nDCG_(importance_scores, relevance_matrix, p):
    """
    Compute the nDCG scores at a given cutoff rank p.
    """
    dcg_scores_p = dcg_score_matrix_p(importance_scores, relevance_matrix, p)
    idcg_scores_p = idcg_score_matrix_p(relevance_matrix, p)

    # Compute normalized DCG
    ndcg_scores_p = np.zeros_like(dcg_scores_p)
    for i in range(len(dcg_scores_p)):
        if idcg_scores_p[i] == 0:
            ndcg_scores_p[i] = 0
        else:
            ndcg_scores_p[i] = dcg_scores_p[i] / idcg_scores_p[i]

    return ndcg_scores_p


def nDCG_p(importance_scores, relevance_matrix):
    nDCG_scores = []
    if len(importance_scores.shape) == 1:
        importance_scores = importance_scores.reshape(1, -1)
    for i in range(importance_scores.shape[0]):
        p = int(np.sum(relevance_matrix[i]))
        nDCG_scores.append(nDCG_(importance_scores[i], relevance_matrix[i], p))
    return np.array(nDCG_scores)

import torch
import numpy as np


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


def dcg_score_matrix(importance_scores, relevance_matrix):
    importance_scores = np.array(importance_scores)
    relevance_matrix = np.array(relevance_matrix)
    importance_scores = importance_scores.squeeze()
    relevance_matrix = relevance_matrix.squeeze()
    assert (
        importance_scores.shape == relevance_matrix.shape
    ), "importance_scores and relevance_matrix must have the same shape"

    sorted_indices = np.argsort(importance_scores, axis=1)[:, ::-1]
    sorted_relevance = np.take_along_axis(
        relevance_matrix, sorted_indices, axis=1
    )
    ranks = np.arange(1, importance_scores.shape[1] + 1)
    dcg_scores = np.sum(sorted_relevance / np.log2(ranks + 1), axis=1)

    return dcg_scores


def idcg_score_matrix(relevance_matrix):
    relevance_matrix = np.array(relevance_matrix)
    sorted_relevance = np.sort(relevance_matrix, axis=1)[:, ::-1]
    ranks = np.arange(1, relevance_matrix.shape[1] + 1)
    idcg_scores = np.sum(sorted_relevance / np.log2(ranks + 1), axis=1)
    return idcg_scores


def nDCG(importance_scores, relevance_matrix):
    dcg_scores = dcg_score_matrix(importance_scores, relevance_matrix)
    idcg_scores = idcg_score_matrix(relevance_matrix)
    ndcg_scores = np.zeros_like(dcg_scores)
    for i in range(len(dcg_scores)):
        if idcg_scores[i] == 0:
            ndcg_scores[i] = 0
        else:
            ndcg_scores[i] = dcg_scores[i] / idcg_scores[i]
    return ndcg_scores

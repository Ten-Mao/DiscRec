import numpy as np
import torch


def ndcg_at_k(pred, tgt, k):
    """
    Calculate NDCG at K using NumPy.

    Args:
    - pred: Array of shape [B, K], predicted ranking for each user
    - tgt: Array of shape [B], ground truth relevant item for each user
    - k: int, rank position for NDCG

    Returns:
    - ndcg: NDCG score at K for the batch
    """
    top_k_preds = pred[:, :k]
    relevant_mask = top_k_preds == tgt[:, None]
    dcg_scores = relevant_mask.astype(np.float32) / np.log2(np.arange(2, k + 2))
    dcg = np.sum(dcg_scores, axis=1)
    idcg = 1.0
    ndcg = dcg / idcg
    return np.mean(ndcg)

def recall_at_k(pred, tgt, k):
    """
    Calculate Recall at K using NumPy.

    Args:
    - pred: Array of shape [B, K], predicted ranking for each user
    - tgt: Array of shape [B], ground truth relevant item for each user
    - k: int, rank position for recall

    Returns:
    - recall: Recall score at K for the batch
    """
    top_k_preds = pred[:, :k]
    relevant_mask = np.any(top_k_preds == tgt[:, None], axis=1)
    recall = relevant_mask.astype(np.float32)
    return np.mean(recall)

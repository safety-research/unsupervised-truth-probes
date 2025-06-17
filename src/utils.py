from pathlib import Path
from typing import List, Tuple

import torch


def create_k_fold_splits(
    n_samples: int, k_folds: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create k-fold cross-validation splits.

    Args:
        n_samples: Total number of samples
        k_folds: Number of folds

    Returns:
        Tuple of (fold_indices, non_fold_indices) where:
        - fold_indices: Tensor of shape (fold_size, k_folds) containing indices for each fold
        - non_fold_indices: Tensor of shape (n_use - fold_size, k_folds) containing
          indices NOT in each fold (i.e., training indices for each fold)
        Note: If n_samples % k_folds != 0, the last samples are dropped
    """
    # Create shuffled indices
    indices = torch.randperm(n_samples)

    # Calculate how many samples to use (drop remainder)
    fold_size = n_samples // k_folds
    n_use = fold_size * k_folds

    # Take only the samples we can evenly divide
    indices = indices[:n_use]

    # Reshape into (k_folds, fold_size)
    fold_indices = indices.view(k_folds, fold_size)

    # Create non-fold indices for each fold
    non_fold_indices = []
    for k in range(k_folds):
        # Get all indices except those in fold k
        mask = torch.ones(k_folds, dtype=torch.bool)
        mask[k] = False
        non_fold_k = fold_indices[mask].flatten()
        non_fold_indices.append(non_fold_k)

    # Stack into tensor of shape (k_folds, n_use - fold_size)
    non_fold_indices = torch.stack(non_fold_indices)

    return fold_indices.permute(1, 0), non_fold_indices.permute(1, 0)


def get_project_root() -> Path:
    """Returns the project root directory."""
    return Path(__file__).resolve().parent.parent

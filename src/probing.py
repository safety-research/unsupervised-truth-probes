from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm.auto import tqdm

from .utils import create_k_fold_splits


class Probe(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expects a tensor of shape (N, ..., D) and outputs (N, C)
        raise NotImplementedError


class LinearProbe(Probe):
    def __init__(self, d_model: int, num_classes: int, enable_bias: bool = True):
        super().__init__(num_classes)
        self.linear = nn.Linear(d_model, num_classes, bias=enable_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class NonlinearProbe(Probe):
    def __init__(
        self, d_model: int, num_classes: int, d_mlp: int = 512, dropout: float = 0.1
    ):
        super().__init__(num_classes)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_mlp, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MultiLinearProbe(Probe):
    def __init__(
        self, d_model: int, num_classes: int, num_probes: int, enable_bias: bool = True
    ):
        super().__init__(num_classes)
        self.num_probes = num_probes
        # K parallel linear layers
        self.probes = nn.ModuleList(
            [
                nn.Linear(d_model, num_classes, bias=enable_bias)
                for _ in range(num_probes)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (N, K, D), Output: (N, K, C)
        N, K, D = x.shape
        assert K == self.num_probes, f"Expected {self.num_probes} probes, got {K}"

        # Stack all weights: (K, D, C)
        weights = torch.stack([probe.weight.T for probe in self.probes], dim=0)

        # Vectorized matrix multiplication: (N, K, D) x (K, D, C) -> (N, K, C)
        output = torch.einsum("nkd,kdc->nkc", x, weights)

        # Add bias if enabled
        if self.probes[0].bias is not None:
            biases = torch.stack([probe.bias for probe in self.probes], dim=0)  # (K, C)
            output = output + biases.unsqueeze(0)  # Broadcast to (N, K, C)

        return output


def train_supervised_probe(
    probe: Probe,
    X: torch.Tensor,  # (N, D) or (N, K, D)
    y: torch.Tensor,  # (N,) or (N, K)
    num_epochs: int = 128,
    lr: float = 0.001,
    batch_size: int = None,
    loss_fn=F.cross_entropy,
    weight_decay: float = 0.01,
    verbose: bool = True,
):
    """Train probe with supervised learning."""
    device = next(probe.parameters()).device
    X, y = X.to(device), y.to(device)
    N = X.shape[0]

    optimizer = optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)

    # Create epoch progress bar
    epoch_pbar = tqdm(
        range(num_epochs), desc="Training", position=0, disable=not verbose
    )
    for epoch in epoch_pbar:
        # Shuffle data
        indices = torch.randperm(N, device=device)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # Full batch if batch_size is None
        if batch_size is None:
            batch_size = N

        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, N, batch_size):
            optimizer.zero_grad()

            # Get batch
            batch_X = X_shuffled[i : i + batch_size]
            batch_y = y_shuffled[i : i + batch_size]

            logits = probe(batch_X)
            loss = loss_fn(logits, batch_y)

            loss.backward()
            optimizer.step()

            # Update progress bar
            epoch_loss += loss.item()
            num_batches += 1
            epoch_pbar.set_postfix({"epoch_loss": f"{epoch_loss/num_batches:.4f}"})

    epoch_pbar.close()
    return probe


def train_ccs_probe(
    probe: Probe,
    X_pos: torch.Tensor,  # (N, D) - "Question? Yes" features
    X_neg: torch.Tensor,  # (N, D) - "Question? No" features
    num_epochs: int = 1000,
    lr: float = 0.01,
    batch_size: int = -1,
    weight_decay: float = 0.01,
    verbose: bool = False,
):
    """Train probe with CCS (unsupervised consistency loss)."""
    assert probe.num_classes == 1, "CCS requires binary probe with single output"
    assert (
        X_pos.shape[0] == X_neg.shape[0]
    ), "CCS requires same number of positive and negative examples"

    if len(X_pos) == 0:
        raise ValueError("Cannot train on empty dataset")

    device = next(probe.parameters()).device
    X_pos, X_neg = X_pos.to(device), X_neg.to(device)

    # Create a fresh probe for this run
    current_probe = LinearProbe(X_pos.shape[-1], 1).to(device)

    # Random initialization with unit norm (as mentioned in the paper)
    with torch.no_grad():
        current_probe.linear.weight.data = torch.randn_like(
            current_probe.linear.weight.data
        )
        current_probe.linear.weight.data = (
            current_probe.linear.weight.data / current_probe.linear.weight.data.norm()
        )
        if current_probe.linear.bias is not None:
            current_probe.linear.bias.data.zero_()

    # Training setup
    optimizer = optim.AdamW(
        current_probe.parameters(), lr=lr, weight_decay=weight_decay
    )

    # Use full batch if batch_size not specified
    actual_batch_size = len(X_pos) if batch_size == -1 else batch_size
    n_batches = (len(X_pos) + actual_batch_size - 1) // actual_batch_size

    # Training loop
    epoch_pbar = tqdm(range(num_epochs), desc="Training CCS", disable=not verbose)

    for epoch in epoch_pbar:
        # Shuffle data each epoch
        permutation = torch.randperm(len(X_pos), device=X_pos.device)
        X_pos_shuffled = X_pos[permutation]
        X_neg_shuffled = X_neg[permutation]

        epoch_loss = 0.0
        epoch_consistency_loss = 0.0
        epoch_confidence_loss = 0.0

        for batch_idx in range(n_batches):
            start_idx = batch_idx * actual_batch_size
            end_idx = min(start_idx + actual_batch_size, len(X_pos))

            # Get batch
            X_pos_batch = X_pos_shuffled[start_idx:end_idx]
            X_neg_batch = X_neg_shuffled[start_idx:end_idx]

            optimizer.zero_grad()

            # Get probabilities
            logits_pos = current_probe(X_pos_batch).squeeze(-1)
            logits_neg = current_probe(X_neg_batch).squeeze(-1)

            p_pos = torch.sigmoid(logits_pos)
            p_neg = torch.sigmoid(logits_neg)

            # CCS loss components
            # Consistency: P(True|pos) should equal 1 - P(True|neg)
            consistency_loss = ((p_pos - (1 - p_neg)) ** 2).mean()

            # Confidence: min{p(x+), p(x-)}²
            # This is the CORRECTED version from the paper
            confidence_loss = (torch.min(p_pos, p_neg) ** 2).mean()

            # Combined loss
            loss = consistency_loss + confidence_loss

            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(current_probe.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()
            epoch_consistency_loss += consistency_loss.item()
            epoch_confidence_loss += confidence_loss.item()

        # Update progress bar
        if verbose:
            avg_loss = epoch_loss / n_batches
            avg_consistency = epoch_consistency_loss / n_batches
            avg_confidence = epoch_confidence_loss / n_batches

            epoch_pbar.set_postfix(
                {
                    "loss": f"{avg_loss:.6f}",
                    "cons": f"{avg_consistency:.6f}",
                    "conf": f"{avg_confidence:.6f}",
                }
            )

    # Calculate final loss
    with torch.no_grad():
        logits_pos = current_probe(X_pos).squeeze(-1)
        logits_neg = current_probe(X_neg).squeeze(-1)
        p_pos = torch.sigmoid(logits_pos)
        p_neg = torch.sigmoid(logits_neg)

        consistency_loss = ((p_pos - (1 - p_neg)) ** 2).mean()
        confidence_loss = (torch.min(p_pos, p_neg) ** 2).mean()
        final_loss = consistency_loss + confidence_loss

    if verbose:
        print(f"Final loss = {final_loss.item():.6f}")

    return current_probe
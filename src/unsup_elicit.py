from src.probing import *
from src.utils import *


def cross_entropy_last_dim(input, target):
    """
    Cross entropy where class dimension is last.

    Args:
        input: (..., n_classes) - logits with classes at the end
        target: (...) - target class indices
    """
    log_probs = F.log_softmax(input, dim=-1)

    # Gather the log probabilities for the target classes
    target_log_probs = torch.gather(log_probs, dim=-1, index=target.unsqueeze(-1))
    target_log_probs = target_log_probs.squeeze(-1)

    # Return negative log likelihood (cross entropy)
    return -target_log_probs.mean()


def compute_kfold_loss(
    X: torch.Tensor,
    y: torch.Tensor,
    k_folds: int = 5,
    specialist_probe_kwargs: dict = {"num_epochs": 256, "lr": 5e-2},
) -> float:
    """
    Compute the k-fold loss for a given X and y.
    """
    if len(X) < k_folds:
        raise ValueError(
            f"Number of samples ({len(X)}) must be greater than or equal to k_folds ({k_folds})"
        )

    n_classes = int(y.max().item()) + 1

    # Get k-fold splits
    fold_indices, train_indices = create_k_fold_splits(len(X), k_folds)
    X_train = X[train_indices]
    X_test = X[fold_indices]
    y_train = y[train_indices]
    y_test = y[fold_indices]

    # Train k specialist probes in parallel
    specialist_probes = MultiLinearProbe(X_train.shape[-1], n_classes, k_folds).to(
        X.device
    )
    specialist_probes = train_supervised_probe(
        specialist_probes,
        X_train,
        y_train,
        verbose=False,
        loss_fn=cross_entropy_last_dim,
        **specialist_probe_kwargs,
    )

    # Get predictions and compute loss
    y_test_pred = specialist_probes(X_test)  # Shape: (fold_size, k_folds, n_classes)

    # Average loss across all folds
    loss = cross_entropy_last_dim(y_test_pred, y_test)
    return loss.item()


def get_relabelings(
    X_pos: torch.Tensor,
    X_neg: torch.Tensor,
    y: torch.Tensor,
    n_relabelings: int = 10,
    specialist_probe_kwargs: dict = {"num_epochs": 256, "lr": 5e-2},
):
    """
    Generate multiple candidate relabelings using parallel co-training.
    Enforces consistency: for each (pos, neg) pair, exactly one gets label=1.
    """
    n_samples = len(X_pos)
    device = X_pos.device

    assert (
        len(X_pos) == len(X_neg) == len(y)
    ), "X_pos, X_neg, and y must have same length"

    candidate_relabelings = [y.clone()]  # Original labeling

    if n_relabelings <= 0:
        raise ValueError(f"n_relabelings must be greater than 0, got {n_relabelings}")

    # Create combined dataset: [X_pos, X_neg] with corresponding labels
    X_combined = torch.cat([X_pos, X_neg], dim=0)  # (2*N, D)
    y_combined = torch.cat(
        [y, 1 - y], dim=0
    ).float()  # (2*N,) - convert to float for BCE

    # Generate all random splits at once (split the pairs, not individual samples)
    all_indices = torch.stack(
        [torch.randperm(n_samples, device=device) for _ in range(n_relabelings)]
    ).permute(1, 0)
    fold_0_indices, fold_1_indices = (
        all_indices[: n_samples // 2, :],
        all_indices[n_samples // 2 :, :],
    )  # each is (n_samples // 2, n_relabelings)

    # Convert pair indices to full indices (both pos and neg)
    fold_0_full = torch.cat(
        [fold_0_indices, fold_0_indices + n_samples], dim=0
    )  # (n_samples, n_relabelings)
    fold_1_full = torch.cat(
        [fold_1_indices, fold_1_indices + n_samples], dim=0
    )  # (n_samples, n_relabelings)

    # Phase 1: Train on fold_0, predict fold_1
    specialist_probes = MultiLinearProbe(X_combined.shape[-1], 1, n_relabelings).to(
        device
    )
    specialist_probes = train_supervised_probe(
        specialist_probes,
        X_combined[fold_0_full],
        y_combined[fold_0_full].unsqueeze(-1),
        verbose=False,
        loss_fn=F.binary_cross_entropy_with_logits,
        **specialist_probe_kwargs,
    )

    # Get predictions for fold_1
    with torch.no_grad():
        logits_phase1 = specialist_probes(
            X_combined[fold_1_full]
        )  # (n_samples, n_relabelings, 1)
        probs_phase1 = torch.sigmoid(
            logits_phase1.squeeze(-1)
        )  # (n_samples, n_relabelings)

    # Enforce consistency: for each pair, pick pos or neg based on higher probability
    fold_1_size = fold_1_indices.shape[0]
    pos_probs = probs_phase1[:fold_1_size, :]  # (n_samples // 2, n_relabelings)
    neg_probs = probs_phase1[fold_1_size:, :]  # (n_samples // 2, n_relabelings)

    # Consistent labeling: if pos_prob > neg_prob, then pos=1, neg=0
    pos_wins = pos_probs > neg_probs  # (n_samples // 2, n_relabelings)
    consistent_labels_phase1 = torch.cat(
        [
            pos_wins.float(),  # pos labels
            (~pos_wins).float(),  # neg labels (opposite of pos)
        ],
        dim=0,
    )  # (2*fold_1_size, n_relabelings)

    # Phase 2: Train on fold_1 (with consistent labels), predict fold_0
    specialist_probes_2 = MultiLinearProbe(X_combined.shape[-1], 1, n_relabelings).to(
        device
    )
    specialist_probes_2 = train_supervised_probe(
        specialist_probes_2,
        X_combined[fold_1_full],
        consistent_labels_phase1.unsqueeze(-1),
        verbose=False,
        loss_fn=F.binary_cross_entropy_with_logits,
        **specialist_probe_kwargs,
    )

    # Get predictions for fold_0
    with torch.no_grad():
        logits_phase2 = specialist_probes_2(
            X_combined[fold_0_full]
        )  # (n_samples, n_relabelings, 1)
        probs_phase2 = torch.sigmoid(
            logits_phase2.squeeze(-1)
        )  # (n_samples, n_relabelings)

    # Enforce consistency for phase 2
    fold_0_size = fold_0_indices.shape[0]
    pos_probs_2 = probs_phase2[:fold_0_size, :]
    neg_probs_2 = probs_phase2[fold_0_size:, :]

    pos_wins_2 = pos_probs_2 > neg_probs_2
    consistent_labels_phase2 = torch.cat(
        [pos_wins_2.float(), (~pos_wins_2).float()], dim=0
    )  # (2*fold_0_size, n_relabelings)

    # Combine predictions into full relabelings
    all_new_labelings = (
        y.unsqueeze(0).expand(n_relabelings, -1).clone()
    )  # (n_relabelings, n_samples)

    # Use scatter_ for vectorized assignment
    all_new_labelings.scatter_(
        dim=1,
        index=fold_1_indices.T,  # (n_relabelings, fold_1_size)
        src=consistent_labels_phase1[
            :fold_1_size, :
        ].T.long(),  # (n_relabelings, fold_1_size)
    )

    all_new_labelings.scatter_(
        dim=1,
        index=fold_0_indices.T,  # (n_relabelings, fold_0_size)
        src=consistent_labels_phase2[
            :fold_0_size, :
        ].T.long(),  # (n_relabelings, fold_0_size)
    )

    # Convert to list and add to candidates
    candidate_relabelings.extend([all_new_labelings[i] for i in range(n_relabelings)])
    return candidate_relabelings


def train_fabien_probe(
    probe: Probe,
    X_pos: torch.Tensor,  # (N, D) - "Question? Yes" features
    X_neg: torch.Tensor,  # (N, D) - "Question? No" features
    n_iterations: int = 20,
    n_relabelings: int = 10,
    specialist_probe_kwargs: dict = {"num_epochs": 256, "lr": 5e-2},
    verbose: bool = True,
):
    """
    Train probe with Fabien's co-training algorithm.

    Uses iterative co-training to discover consistent labels, then trains
    the probe on the discovered labels.
    """
    n_samples = len(X_pos)
    device = X_pos.device

    assert len(X_pos) == len(X_neg), "X_pos and X_neg must have same length"
    assert (
        probe.num_classes == 1
    ), "Fabien probe requires binary probe with single output"

    # Initialize with random labels
    y = torch.randint(0, 2, (n_samples,), device=device)

    if verbose:
        print(
            f"Starting Fabien co-training: {n_samples} samples, {n_iterations} iterations"
        )

    # Main co-training loop
    pbar = tqdm(range(n_iterations), desc="Fabien Co-Training", disable=not verbose)

    for iteration in pbar:
        # Generate candidate relabelings through co-training
        relabelings = get_relabelings(
            X_pos,
            X_neg,
            y,
            n_relabelings=n_relabelings,
            specialist_probe_kwargs=specialist_probe_kwargs,
        )

        # Evaluate coherence of each candidate relabeling
        best_loss = float("inf")
        best_relabeling = None

        for relabeling in relabelings:
            # Create combined dataset for coherence evaluation
            X_combined = torch.cat([X_pos, X_neg], dim=0)
            y_combined = torch.cat([relabeling, 1 - relabeling], dim=0)

            # Compute coherence (lower = more self-consistent)
            try:
                loss = compute_kfold_loss(X_combined, y_combined)
                if loss < best_loss:
                    best_loss = loss
                    best_relabeling = relabeling
            except:
                continue  # Skip if coherence computation fails

        if best_relabeling is None:
            if verbose:
                tqdm.write(
                    f"Warning: No valid relabeling found at iteration {iteration}"
                )
            break

        # Update to best relabeling
        y = best_relabeling

        # Update progress
        pbar.set_postfix({"Loss": f"{best_loss:.4f}"})

    pbar.close()

    # Train final probe on discovered labels
    X_combined = torch.cat([X_pos, X_neg], dim=0)  # (2N, D)
    y_combined = torch.cat([y.float(), (1 - y).float()], dim=0)  # (2N,)

    # Train the probe using supervised learning
    trained_probe = train_supervised_probe(
        probe,
        X_combined,
        y_combined.unsqueeze(-1),  # (2N, 1) for binary classification
        verbose=verbose,
        loss_fn=F.binary_cross_entropy_with_logits,
        **specialist_probe_kwargs,
    )

    if verbose:
        final_pos = y.sum().item()
        print(f"Training complete. Final labels: {final_pos}/{n_samples} positive")

    return trained_probe


def label_new_samples_parallel(
    X_pos_active: torch.Tensor,  # Current active pos samples
    X_neg_active: torch.Tensor,  # Current active neg samples
    y_active: torch.Tensor,  # Current active labels
    X_pos_new: torch.Tensor,  # New pos samples to label
    X_neg_new: torch.Tensor,  # New neg samples to label
    n_relabelings: int,
    specialist_probe_kwargs: dict,
) -> torch.Tensor:
    """Label new samples using multiple parallel specialist probes"""

    device = X_pos_active.device
    n_new = len(X_pos_new)

    # Create multiple random bootstrap samples of active set
    bootstrap_indices = torch.stack(
        [
            torch.randint(0, len(X_pos_active), (len(X_pos_active),), device=device)
            for _ in range(n_relabelings)
        ]
    ).T  # (n_active, n_relabelings)

    # Prepare training data for parallel probes
    X_active_combined = torch.cat([X_pos_active, X_neg_active], dim=0)
    y_active_combined = torch.cat([y_active.float(), (1 - y_active).float()], dim=0)

    # Bootstrap the combined data
    combined_indices = torch.cat(
        [bootstrap_indices, bootstrap_indices + len(X_pos_active)], dim=0
    )  # (2*n_active, n_relabelings)

    # Train multiple specialist probes in parallel
    specialist_probes = MultiLinearProbe(
        X_active_combined.shape[-1], 1, n_relabelings
    ).to(device)
    specialist_probes = train_supervised_probe(
        specialist_probes,
        X_active_combined[combined_indices],  # (2*n_active, n_relabelings, dim)
        y_active_combined[combined_indices].unsqueeze(
            -1
        ),  # (2*n_active, n_relabelings, 1)
        verbose=False,
        loss_fn=F.binary_cross_entropy_with_logits,
        **specialist_probe_kwargs,
    )

    # Get predictions for new samples
    X_new_combined = torch.cat([X_pos_new, X_neg_new], dim=0)  # (2*n_new, dim)

    # FIX: Reshape for MultiLinearProbe - expand to (2*n_new, n_relabelings, dim)
    X_new_expanded = X_new_combined.unsqueeze(1).expand(-1, n_relabelings, -1)

    with torch.no_grad():
        logits = specialist_probes(X_new_expanded)  # (2*n_new, n_relabelings, 1)
        probs = torch.sigmoid(logits.squeeze(-1))  # (2*n_new, n_relabelings)

    # Apply consistency constraint (same as your original approach)
    pos_probs = probs[:n_new, :]  # (n_new, n_relabelings)
    neg_probs = probs[n_new:, :]  # (n_new, n_relabelings)

    pos_wins = pos_probs > neg_probs  # (n_new, n_relabelings)

    # Take majority vote across all specialist probes
    final_labels = (pos_wins.float().mean(dim=1) > 0.5).long()

    return final_labels


def relabel_active_set_parallel(
    X_pos_active: torch.Tensor,
    X_neg_active: torch.Tensor,
    y_active: torch.Tensor,
    n_iterations: int,
    n_relabelings: int,
    specialist_probe_kwargs: dict,
) -> torch.Tensor:
    """Iteratively relabel active set using parallel co-training"""

    current_labels = y_active.clone()

    for iteration in range(n_iterations):
        # Generate candidate relabelings using your existing parallel approach
        candidate_relabelings = get_relabelings(
            X_pos_active,
            X_neg_active,
            current_labels,
            n_relabelings=n_relabelings,
            specialist_probe_kwargs=specialist_probe_kwargs,
        )

        # Evaluate coherence of each candidate
        best_loss = float("inf")
        best_relabeling = None

        for relabeling in candidate_relabelings:
            X_combined = torch.cat([X_pos_active, X_neg_active], dim=0)
            y_combined = torch.cat([relabeling, 1 - relabeling], dim=0)

            try:
                loss = compute_kfold_loss(X_combined, y_combined)
                if loss < best_loss:
                    best_loss = loss
                    best_relabeling = relabeling
            except:
                continue

        if best_relabeling is not None:
            current_labels = best_relabeling
        else:
            break  # No improvement found

    return current_labels


def train_fabien_probe_incremental(
    probe: Probe,
    X_pos: torch.Tensor,  # (N, D) - "Question? Yes" features
    X_neg: torch.Tensor,  # (N, D) - "Question? No" features
    start_k: int = 3,
    increase_per_turn: int = 2,
    n_relabelings: int = 10,
    relabel_iterations_per_turn: int = 10,
    specialist_probe_kwargs: dict = {"num_epochs": 256, "lr": 5e-2},
    verbose: bool = True,
):
    """
    Train probe with Fabien's incremental co-training algorithm.

    Starts with a small subset and incrementally adds more samples,
    using parallel co-training to discover consistent labels.
    """
    n_samples = len(X_pos)
    device = X_pos.device

    assert len(X_pos) == len(X_neg), "X_pos and X_neg must have same length"
    assert (
        probe.num_classes == 1
    ), "Fabien probe requires binary probe with single output"
    assert (
        start_k <= n_samples
    ), f"start_k ({start_k}) must be <= n_samples ({n_samples})"

    # Start with small random subset
    all_indices = torch.randperm(n_samples, device=device)
    active_indices = all_indices[:start_k]
    remaining_indices = all_indices[start_k:]

    # Initialize random labels for active set
    y_active = torch.randint(0, 2, (start_k,), device=device)

    if verbose:
        print(
            f"Starting incremental Fabien co-training: {start_k} initial samples, "
            f"adding {increase_per_turn} per turn"
        )

    # Main incremental loop
    pbar = tqdm(
        total=n_samples,
        initial=start_k,
        desc="Incremental Co-Training",
        disable=not verbose,
    )

    while len(remaining_indices) > 0:
        # Phase 1: Add new samples and label them
        n_to_add = min(increase_per_turn, len(remaining_indices))
        new_indices = remaining_indices[:n_to_add]
        remaining_indices = remaining_indices[n_to_add:]

        # Label new samples using current active set (parallel)
        new_labels = label_new_samples_parallel(
            X_pos[active_indices],
            X_neg[active_indices],
            y_active,
            X_pos[new_indices],
            X_neg[new_indices],
            n_relabelings,
            specialist_probe_kwargs,
        )

        # Add to active set
        active_indices = torch.cat([active_indices, new_indices])
        y_active = torch.cat([y_active, new_labels])

        # Phase 2: Relabel active set iteratively (parallel)
        y_active = relabel_active_set_parallel(
            X_pos[active_indices],
            X_neg[active_indices],
            y_active,
            relabel_iterations_per_turn,
            n_relabelings,
            specialist_probe_kwargs,
        )

        # Update progress
        if verbose:
            try:
                coherence = compute_kfold_loss(
                    torch.cat([X_pos[active_indices], X_neg[active_indices]]),
                    torch.cat([y_active, 1 - y_active]),
                )
                final_pos = y_active.sum().item()
                pbar.set_postfix(
                    {
                        "Coherence": f"{coherence:.4f}",
                        "Pos Labels": f"{final_pos}/{len(active_indices)}",
                    }
                )
            except:
                pbar.set_postfix({"Coherence": "N/A"})

        pbar.update(n_to_add)

    pbar.close()

    # Final training on all discovered labels
    X_combined = torch.cat([X_pos, X_neg], dim=0)  # (2N, D)
    y_combined = torch.cat([y_active.float(), (1 - y_active).float()], dim=0)  # (2N,)

    # Train the probe using supervised learning
    trained_probe = train_supervised_probe(
        probe,
        X_combined,
        y_combined.unsqueeze(-1),  # (2N, 1) for binary classification
        verbose=verbose,
        loss_fn=F.binary_cross_entropy_with_logits,
        **specialist_probe_kwargs,
    )

    if verbose:
        final_pos = y_active.sum().item()
        print(f"Training complete. Final labels: {final_pos}/{n_samples} positive")

    return trained_probe

import json
import os
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from promptsource.templates import DatasetTemplates
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .evals import CCSDataset
from .probing import LinearProbe, train_ccs_probe, train_supervised_probe
from .unsup_elicit import train_fabien_probe, train_fabien_probe_exponential
from .utils import get_project_root

LOADED_MODELS = {}
LOADED_TOKENIZERS = {}


def get_results_on_dataset(
    dataset: CCSDataset,
    model_name: str,
    layer_idx: int = -1,
    test_size: float = 0.5,
    seed: int = 42,
    batch_size: int = 32,
    run_methods: List[str] = [
        "supervised",
        "model_logits",
        "ccs",
        "fabiens_method",
        "pca",
        "random"
    ],
) -> List[Dict]:
    """
    Evaluate different methods on a formatted dataset.

    Args:
        dataset: CCSDataset object containing prompts and labels
        model_name: HuggingFace model name (e.g., "gpt2")
        layer_idx: Layer to extract activations from (-1 for last layer)
        test_size: Fraction for test set
        seed: Random seed for reproducibility
        batch_size: Batch size for processing
        run_methods: List of methods to evaluate

    Returns:
        List of dicts with results for each test example
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load the dataset
    positive_prompts = dataset.positive_prompts
    negative_prompts = dataset.negative_prompts
    ground_truth_labels = dataset.ground_truth_labels
    dataset_name = dataset.dataset_name

    # Validate inputs
    assert len(positive_prompts) == len(negative_prompts) == len(ground_truth_labels)
    assert all(label in [0, 1] for label in ground_truth_labels)
    assert 0 < test_size < 1, "test_size must be between 0 and 1"

    # Split into train and test
    n_examples = len(positive_prompts)
    indices = list(range(n_examples))
    random.shuffle(indices)

    n_train = int(n_examples * (1 - test_size))
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    print(f"Split: {len(train_indices)} train, {len(test_indices)} test examples")

    # Always generate test prompts and labels
    test_labels = []
    test_prompts = []

    for i in test_indices:
        # Randomly choose to use positive or negative claim
        if random.random() < 0.5:
            # Use positive claim
            test_prompts.append(positive_prompts[i])
            # Label: 1 if ground truth says positive is correct, 0 if negative is correct
            test_labels.append(ground_truth_labels[i])
        else:
            # Use negative claim
            test_prompts.append(negative_prompts[i])
            # Label: 0 if ground truth says positive is correct, 1 if negative is correct
            test_labels.append(1 - ground_truth_labels[i])

    # Check if all the results are already saved
    all_cached = check_all_saved(dataset_name, model_name, layer_idx, run_methods)

    if all_cached:
        print("All results are already saved, skipping model computation")
        # Set these to None since we won't need them
        model = None
        tokenizer = None
        positive_token_id = None
        negative_token_id = None
        train_pos_activations = None
        train_neg_activations = None
        train_labels = None
        test_activations = None
        test_logits = None
        test_final_tokens = None
    else:
        print("Some results missing, computing activations...")

        # Load model
        print(f"Loading model: {model_name}")
        if model_name not in LOADED_MODELS:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            LOADED_MODELS[model_name] = model
        else:
            model = LOADED_MODELS[model_name]
        if model_name not in LOADED_TOKENIZERS:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            LOADED_TOKENIZERS[model_name] = tokenizer
        else:
            tokenizer = LOADED_TOKENIZERS[model_name]

        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Automatically detect the common ending tokens
        positive_token_id, negative_token_id = get_common_last_tokens(
            tokenizer, positive_prompts, negative_prompts
        )

        # ✅ COMPUTE ALL ACTIVATIONS ONCE
        print("Computing activations for all prompts...")

        # Get activations for positive and negative prompts
        pos_activations, pos_logits, pos_final_tokens = get_final_acts_and_logits(
            model, tokenizer, positive_prompts, layer_idx, batch_size
        )
        neg_activations, neg_logits, neg_final_tokens = get_final_acts_and_logits(
            model, tokenizer, negative_prompts, layer_idx, batch_size
        )

        # Remove the mean of the pos and neg activations
        pos_activations = pos_activations - pos_activations.mean(dim=0, keepdim=True)
        neg_activations = neg_activations - neg_activations.mean(dim=0, keepdim=True)

        # After mean centering
        pos_norm = pos_activations.norm(dim=1).mean() * np.sqrt(
            pos_activations.shape[1]
        )
        neg_norm = neg_activations.norm(dim=1).mean() * np.sqrt(
            neg_activations.shape[1]
        )
        pos_activations = pos_activations / pos_norm
        neg_activations = neg_activations / neg_norm

        # Split activations by train/test indices
        train_pos_activations = pos_activations[train_indices]
        train_neg_activations = neg_activations[train_indices]
        train_labels = [ground_truth_labels[i] for i in train_indices]

        # For test: use the same logic to pick activations that match our prompt selection
        test_activations = []
        test_logits = []
        test_final_tokens = []

        # Reset random state to reproduce the same choices
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Skip to the same point in random sequence
        n_examples = len(positive_prompts)
        indices = list(range(n_examples))
        random.shuffle(indices)
        n_train = int(n_examples * (1 - test_size))
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

        for i in test_indices:
            # Use the same random choice logic
            if random.random() < 0.5:
                # Used positive claim
                test_activations.append(pos_activations[i])
                test_logits.append(pos_logits[i])
                test_final_tokens.append(pos_final_tokens[i])
            else:
                # Used negative claim
                test_activations.append(neg_activations[i])
                test_logits.append(neg_logits[i])
                test_final_tokens.append(neg_final_tokens[i])

        # Convert to tensors
        test_activations = torch.stack(test_activations)
        test_logits = torch.stack(test_logits)
        test_final_tokens = torch.stack(test_final_tokens)

    # Initialize method results storage
    method_results = {method: [] for method in run_methods}

    # Run each method
    for method in run_methods:
        print(f"Running method: {method}")

        file_path = (
            get_project_root()
            / "results"
            / dataset_name
            / method
            / f"{model_name}_{layer_idx}"
            / f"{dataset_name}.json"
        )

        if os.path.exists(file_path):
            test_scores = _load_results(file_path)
            method_results[method] = test_scores
            continue

        # If we get here and everything was cached, something went wrong
        if all_cached:
            raise RuntimeError(
                f"Expected cached results for {method} but file {file_path} not found"
            )

        if method == "supervised":
            test_scores = _run_supervised_method(
                train_pos_activations,
                train_neg_activations,
                train_labels,
                test_activations,
            )
        elif method == "model_logits":
            test_scores = _run_model_logits_method(
                test_logits,
                test_final_tokens,
                positive_token_id,
                negative_token_id,
            )
        elif method == "ccs":
            test_scores = _run_ccs_method(
                train_pos_activations,
                train_neg_activations,
                train_labels,
                test_activations,
            )
        elif method == "fabiens_method":
            test_scores = _run_fabien_method(
                train_pos_activations,
                train_neg_activations,
                train_labels,
                test_activations,
            )
        elif method == "fabiens_method_exponential":
            test_scores = _run_fabien_method_exponential(
                train_pos_activations,
                train_neg_activations,
                train_labels,
                test_activations,
            )
        elif method == "pca":
            test_scores = _run_pca_method(
                train_pos_activations,
                train_neg_activations,
                train_labels,
                test_activations,
            )
        elif method == "random":
            test_scores = _random_probe_method(
                train_pos_activations,
                train_neg_activations,
                train_labels,
                test_activations,
            )
        else:
            raise NotImplementedError(f"Method {method} not implemented")

        _save_results(test_scores, file_path)
        method_results[method] = test_scores

    # Save all test prompts to a file
    file_path = get_project_root() / "results" / dataset_name / "prompts.json"
    _save_results(test_prompts, file_path)

    # Build results list for test examples
    results = []
    for i in range(len(test_indices)):
        # Extract base prompt
        base_prompt = test_prompts[i]

        # Build result dict
        result = {"prompt": base_prompt, "truth_label": test_labels[i], "methods": {}}

        # Add scores from each method
        for method in run_methods:
            if i < len(method_results[method]):
                result["methods"][method] = method_results[method][i]
            else:
                result["methods"][method] = None

        results.append(result)

    return results


def check_all_saved(
    dataset_name: str, model_name: str, layer_idx: int, run_methods: List[str]
) -> bool:
    """
    Check if all results for the given configuration are already saved to disk.

    Args:
        dataset_name: Name of the dataset
        model_name: HuggingFace model name
        layer_idx: Layer index used for extraction
        run_methods: List of methods to check

    Returns:
        True if all results exist and are loadable, False otherwise
    """
    for method in run_methods:
        file_path = (
            get_project_root()
            / "results"
            / dataset_name
            / method
            / f"{model_name}_{layer_idx}"
            / f"{dataset_name}.json"
        )

        if not os.path.exists(file_path):
            return False

        # Also verify the file is actually loadable
        try:
            _load_results(file_path)
        except Exception:
            return False

    return True


def get_common_last_tokens(
    tokenizer, positive_prompts: List[str], negative_prompts: List[str]
) -> Tuple[int, int]:
    """
    Automatically detect the common last tokens from positive and negative prompt sets.

    Args:
        tokenizer: HuggingFace tokenizer
        positive_prompts: List of positive prompts
        negative_prompts: List of negative prompts

    Returns:
        Tuple of (positive_token_id, negative_token_id)
    """
    # Get last tokens from a sample of prompts to find the common pattern
    pos_last_tokens = []
    neg_last_tokens = []

    # Sample prompts to determine the pattern (use more samples for reliability)
    sample_size = min(50, len(positive_prompts))
    for i in range(sample_size):
        pos_tokens = tokenizer.encode(positive_prompts[i], add_special_tokens=False)
        neg_tokens = tokenizer.encode(negative_prompts[i], add_special_tokens=False)

        if len(pos_tokens) > 0:
            pos_last_tokens.append(pos_tokens[-1])
        if len(neg_tokens) > 0:
            neg_last_tokens.append(neg_tokens[-1])

    # Find the most common last token for each set
    if not pos_last_tokens or not neg_last_tokens:
        raise ValueError("Could not extract tokens from prompts")

    pos_token_id = Counter(pos_last_tokens).most_common(1)[0][0]
    neg_token_id = Counter(neg_last_tokens).most_common(1)[0][0]

    # Debug: print what tokens we found
    pos_token_text = tokenizer.decode([pos_token_id])
    neg_token_text = tokenizer.decode([neg_token_id])
    print(f"Detected positive token: '{pos_token_text}' (ID: {pos_token_id})")
    print(f"Detected negative token: '{neg_token_text}' (ID: {neg_token_id})")

    # Validate that we found different tokens
    if pos_token_id == neg_token_id:
        print("Warning: Positive and negative prompts end with the same token!")
        print("This may indicate an issue with the dataset formatting.")

    return pos_token_id, neg_token_id


def get_final_acts_and_logits(
    model, tokenizer, prompts: List[str], layer_idx: int = -1, batch_size: int = 32
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract final token activations, penultimate token logits, and final token IDs.

    For prompts ending with specific tokens, we want:
    - Activations from the final token position
    - Logits from the penultimate position (to predict the final token)
    - The actual final token IDs (to get their probabilities from the logits)

    Returns:
        - activations: Hidden states at final token position (n_prompts, hidden_dim)
        - logits: Full logits at penultimate token position (n_prompts, vocab_size)
        - final_tokens: Token IDs at final position (n_prompts,)
    """
    all_activations = []
    all_logits = []
    all_final_tokens = []

    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
            batch_prompts = prompts[i : i + batch_size]

            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            outputs = model(**inputs, output_hidden_states=True)

            # Find last non-padded token for each sequence (with right padding)
            attention_mask = inputs["attention_mask"]
            last_indices = attention_mask.sum(dim=1) - 1  # Last actual token position
            penultimate_indices = last_indices - 1  # Penultimate token position

            batch_indices = torch.arange(len(batch_prompts))

            # Extract final token activations
            final_activations = outputs.hidden_states[layer_idx][
                batch_indices, last_indices
            ]

            # Extract full logits at penultimate token position
            penultimate_logits = outputs.logits[batch_indices, penultimate_indices]

            # Extract the actual final token IDs
            input_ids = inputs["input_ids"]
            final_token_ids = input_ids[batch_indices, last_indices]

            all_activations.append(final_activations.cpu())
            all_logits.append(penultimate_logits.cpu())
            all_final_tokens.append(final_token_ids.cpu())

    return (
        torch.cat(all_activations, dim=0),
        torch.cat(all_logits, dim=0),
        torch.cat(all_final_tokens, dim=0),
    )


def _run_supervised_method(
    train_pos_activations: torch.Tensor,
    train_neg_activations: torch.Tensor,
    train_labels: List[int],
    test_activations: torch.Tensor,
) -> List[float]:
    """Run supervised probe method using pre-computed activations."""
    train_acts = torch.cat([train_pos_activations, train_neg_activations], dim=0).to(
        torch.float32
    )
    train_labels = torch.tensor(
        train_labels + [1 - label for label in train_labels]
    ).to(torch.int64)

    probe = LinearProbe(train_acts.shape[-1], 2).cuda()
    probe = train_supervised_probe(
        probe=probe,
        X=train_acts.cuda(),
        y=train_labels.cuda(),
        num_epochs=1000,
        lr=5e-3,
        batch_size=32,
        weight_decay=0.01,
    )

    test_scores = probe(test_activations.cuda().to(torch.float32))
    return torch.softmax(test_scores, dim=-1)[:, 1].cpu().tolist()


def _run_model_logits_method(
    test_logits: torch.Tensor,
    test_final_tokens: torch.Tensor,
    positive_token_id: int,
    negative_token_id: int,
) -> List[float]:
    """
    Compute relative probability: P(actual_token) / (P(positive_token) + P(negative_token))

    This gives the "share" of probability the actual ending token gets
    out of the two possible ending tokens (positive vs negative).

    Args:
        test_logits: Logits at penultimate position for each test example
        test_final_tokens: Actual final token IDs for each test example
        positive_token_id: Token ID that appears at end of positive prompts
        negative_token_id: Token ID that appears at end of negative prompts

    Returns:
        List of relative probability scores (0 to 1)
    """
    # Convert to log probabilities for numerical stability
    log_probs = torch.log_softmax(test_logits, dim=-1)

    # Get log probabilities for positive and negative tokens
    pos_log_probs = log_probs[:, positive_token_id]
    neg_log_probs = log_probs[:, negative_token_id]

    # Get log probability of actual final token for each example
    batch_indices = torch.arange(len(test_final_tokens))
    actual_token_log_probs = log_probs[batch_indices, test_final_tokens]

    # Compute log(P(positive) + P(negative)) using logsumexp for numerical stability
    pos_neg_log_probs = torch.stack([pos_log_probs, neg_log_probs], dim=1)
    log_sum_pos_neg = torch.logsumexp(pos_neg_log_probs, dim=1)

    # Compute relative probability: P(actual) / (P(positive) + P(negative))
    relative_log_probs = actual_token_log_probs - log_sum_pos_neg
    relative_probs = torch.exp(relative_log_probs)

    return relative_probs.tolist()


def _run_ccs_method(
    train_pos_activations: torch.Tensor,
    train_neg_activations: torch.Tensor,
    train_labels: List[
        int
    ],  # Note: we don't actually need this since CCS is unsupervised
    test_activations: torch.Tensor,
) -> List[float]:
    """Run CCS method using pre-computed activations."""
    probe = LinearProbe(train_pos_activations.shape[-1], 1).cuda()
    probe = train_ccs_probe(
        probe=probe,
        X_pos=train_pos_activations.to(torch.float32),
        X_neg=train_neg_activations.to(torch.float32),
        num_epochs=1000,
        lr=0.01,
        batch_size=32,
        verbose=True,
    )

    test_scores = probe(test_activations.cuda().to(torch.float32))
    return torch.sigmoid(test_scores)[:, 0].cpu().tolist()


def _run_fabien_method(
    train_pos_activations: torch.Tensor,
    train_neg_activations: torch.Tensor,
    train_labels: List[
        int
    ],  # Note: This parameter isn't used since Fabien discovers labels
    test_activations: torch.Tensor,
) -> List[float]:
    probe = LinearProbe(train_pos_activations.shape[-1], 1).cuda()
    probe = train_fabien_probe(
        probe=probe,
        X_pos=train_pos_activations.cuda().to(torch.float32),
        X_neg=train_neg_activations.cuda().to(torch.float32),
        n_iterations=20,
        n_relabelings=10,
    )
    test_scores = probe(test_activations.cuda().to(torch.float32))
    return torch.sigmoid(test_scores)[:, 0].cpu().tolist()

def _run_fabien_method_exponential(
    train_pos_activations: torch.Tensor,
    train_neg_activations: torch.Tensor,
    train_labels: List[
        int
    ],  # Note: This parameter isn't used since Fabien discovers labels
    test_activations: torch.Tensor,
) -> List[float]:
    probe = LinearProbe(train_pos_activations.shape[-1], 1).cuda()
    probe = train_fabien_probe_exponential(
        probe=probe,
        X_pos=train_pos_activations.cuda().to(torch.float32),
        X_neg=train_neg_activations.cuda().to(torch.float32),
        max_iterations=20,
        n_relabelings=10,
    )
    test_scores = probe(test_activations.cuda().to(torch.float32))
    return torch.sigmoid(test_scores)[:, 0].cpu().tolist()


def _random_probe_method(
    train_pos_activations: torch.Tensor,
    train_neg_activations: torch.Tensor,
    train_labels: List[int],
    test_activations: torch.Tensor,
) -> List[float]:
    """Run random probe method using pre-computed activations."""
    probe = LinearProbe(train_pos_activations.shape[-1], 1).cuda()
    test_scores = probe(test_activations.cuda().to(torch.float32))
    return torch.sigmoid(test_scores)[:, 0].cpu().tolist()


def _run_pca_method(
    train_pos_activations: torch.Tensor,
    train_neg_activations: torch.Tensor,
    train_labels: List[int],
    test_activations: torch.Tensor,
) -> List[float]:
    """
    Run PCA method to find the principal component that best separates
    positive and negative activations, then project test activations onto it.

    The idea is that the first principal component of the combined data
    should capture the main direction of variation, which hopefully
    corresponds to the true/false distinction.

    Args:
        train_pos_activations: Training activations from positive prompts
        train_neg_activations: Training activations from negative prompts
        train_labels: Ground truth labels (not used in this unsupervised method)
        test_activations: Test activations to score

    Returns:
        List of normalized scores between 0 and 1
    """
    # Convert tensors to numpy arrays for sklearn
    train_pos_np = train_pos_activations.cpu().numpy()
    train_neg_np = train_neg_activations.cpu().numpy()
    test_np = test_activations.cpu().numpy()

    # Combine training activations
    combined_activations = np.concatenate([train_pos_np, train_neg_np], axis=0)

    # Fit PCA to find the principal component
    pca = PCA(n_components=1)
    pca.fit(combined_activations)

    # Project test activations onto the first principal component
    test_projections = pca.transform(test_np).flatten()

    # Project training activations to determine which direction is positive
    train_pos_projections = pca.transform(train_pos_np).flatten()
    train_neg_projections = pca.transform(train_neg_np).flatten()

    # Determine if we need to flip the direction
    # If positive activations have lower mean projection than negative,
    # we should flip the scores
    pos_mean = np.mean(train_pos_projections)
    neg_mean = np.mean(train_neg_projections)

    if pos_mean < neg_mean:
        test_projections = -test_projections

    # Normalize scores to [0, 1] range using min-max normalization
    # We'll use the training data range to determine normalization
    all_train_projections = np.concatenate(
        [train_pos_projections, train_neg_projections]
    )
    if pos_mean < neg_mean:
        all_train_projections = -all_train_projections

    min_val = np.min(all_train_projections)
    max_val = np.max(all_train_projections)

    # Avoid division by zero
    if max_val == min_val:
        normalized_scores = np.full_like(test_projections, 0.5)
    else:
        normalized_scores = (test_projections - min_val) / (max_val - min_val)
        # Clip to [0, 1] range in case test data is outside training range
        normalized_scores = np.clip(normalized_scores, 0.0, 1.0)

    return normalized_scores.tolist()


def _save_results(results: List[float], filename: str):
    """Save results to a JSON file.

    Args:
        results: List of float scores from experiment
        filename: Path to save the results file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save results to JSON file
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)


def _load_results(filename: str) -> List[float]:
    """Load results from a JSON file.

    Args:
        filename: Path to the results file

    Returns:
        List of float scores from experiment
    """
    if not os.path.exists(filename):
        return []

    with open(filename, "r") as f:
        return json.load(f)

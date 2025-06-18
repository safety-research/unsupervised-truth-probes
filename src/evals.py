import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset

IMPLEMENTED_DATASETS = ["imdb", "ag_news"]


@dataclass
class CCSDataset:
    """Dataset for CCS (Contrast-Consistent Search) experiments.

    Attributes:
        positive_prompts: List of prompts ending with "I think this Claim is True"
        negative_prompts: List of prompts ending with "I think this Claim is False"
        ground_truth_labels: List of labels where 1 indicates positive prompt is correct, 0 indicates negative prompt is correct
        dataset_name: Name of the source dataset
    """

    positive_prompts: List[str]
    negative_prompts: List[str]
    ground_truth_labels: List[int]
    dataset_name: str

    def __post_init__(self):
        """Validate the dataset after initialization."""
        # Check that all lists have the same length
        assert (
            len(self.positive_prompts)
            == len(self.negative_prompts)
            == len(self.ground_truth_labels)
        ), "All lists must have the same length"

        # Check that ground truth labels are binary
        assert all(
            label in [0, 1] for label in self.ground_truth_labels
        ), "Ground truth labels must be binary (0 or 1)"

        # Check that prompts are non-empty
        assert all(
            len(p) > 0 for p in self.positive_prompts
        ), "Positive prompts cannot be empty"
        assert all(
            len(p) > 0 for p in self.negative_prompts
        ), "Negative prompts cannot be empty"

    def __len__(self):
        return len(self.positive_prompts)

    def __getitem__(self, idx):
        return (
            self.positive_prompts[idx],
            self.negative_prompts[idx],
            self.ground_truth_labels[idx],
        )


def load_dataset_for_ccs(
    dataset_name: str,
    dataset_config: str | None = None,
    split: str = "train",
    max_examples: int = 1000,
    seed: int = 42,
    prefix: str = "",
    prompt_idx: int = 0,
) -> CCSDataset:
    """
    Load and format datasets following the paper's approach.

    Args:
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset config
        split: Dataset split to use
        max_examples: Maximum examples (paper uses 1000, except COPA which has 500)
        seed: Random seed
        prefix: Optional prefix (e.g., misleading prefix from Figure 5)
        prompt_idx: Which prompt template to use (paper uses 8-13 per dataset)

    Returns:
        CCSDataset containing positive_prompts, negative_prompts, and ground_truth_labels
    """
    random.seed(seed)
    np.random.seed(seed)

    if dataset_name not in IMPLEMENTED_DATASETS:
        try:
            return load_dataset_from_repeng(dataset_name, max_examples, seed)
        except:
            raise NotImplementedError(
                f"Dataset {dataset_name} not implemented in repeng"
            )

    # Load dataset
    if dataset_config:
        dataset = load_dataset(
            dataset_name, dataset_config, split=split, trust_remote_code=True
        )
    else:
        dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)

    # Limit examples (paper uses 1000 for most, 500 for COPA)
    if dataset_name == "copa" and max_examples > 500:
        max_examples = 500

    if len(dataset) > max_examples:
        indices = np.random.choice(len(dataset), max_examples, replace=False)
        dataset = dataset.select(indices)

    # Format based on dataset
    if dataset_name == "ag_news":
        positive_prompts, negative_prompts, ground_truth_labels = format_ag_news(
            dataset, prefix, prompt_idx
        )
    if dataset_name == "imdb":
        positive_prompts, negative_prompts, ground_truth_labels = format_imdb(
            dataset, prefix, prompt_idx
        )

    return CCSDataset(
        positive_prompts=positive_prompts,
        negative_prompts=negative_prompts,
        ground_truth_labels=ground_truth_labels,
        dataset_name=dataset_name,
    )


def load_dataset_from_repeng(
    dataset_name: str, max_examples: Optional[int] = None, seed: int = 42
) -> CCSDataset:
    """
    Load dataset using repeng and convert to CCSDataset format.

    Args:
        dataset_name: Name of the dataset (e.g., "imdb")
        max_examples: Maximum number of examples to include (optional)
        seed: Random seed for reproducibility

    Returns:
        CCSDataset in the old format
    """
    try:
        from repeng.datasets.elk.utils.fns import get_dataset
    except ImportError:
        print("repeng library not found. Please install it first.")
        raise ImportError()

    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Load dataset using repeng
    dataset = get_dataset(dataset_name)

    # Group by group_id to pair positive and negative examples
    grouped_data = {}
    for key, binary_row in dataset.items():
        group_id = binary_row.group_id
        if group_id not in grouped_data:
            grouped_data[group_id] = {}

        # Determine if this is the positive or negative example
        # Based on your sample data, answer_type '0' seems to be negative, '1' positive
        if binary_row.answer_type == "0":
            grouped_data[group_id]["negative"] = binary_row
        elif binary_row.answer_type == "1":
            grouped_data[group_id]["positive"] = binary_row

    # Extract paired examples
    positive_prompts = []
    negative_prompts = []
    ground_truth_labels = []

    # Get list of group_ids and optionally limit examples
    group_ids = list(grouped_data.keys())
    if max_examples is not None and len(group_ids) > max_examples:
        group_ids = random.sample(group_ids, max_examples)

    for group_id in group_ids:
        group = grouped_data[group_id]

        # Skip if we don't have both positive and negative examples
        if "positive" not in group or "negative" not in group:
            continue

        pos_row = group["positive"]
        neg_row = group["negative"]

        # Extract the prompts
        positive_prompts.append(pos_row.text)
        negative_prompts.append(neg_row.text)

        # Determine ground truth based on which one has label=True
        # From your sample data, it looks like:
        # - When answer_type='0' (negative), label=True means negative is correct (ground_truth=0)
        # - When answer_type='1' (positive), label=True means positive is correct (ground_truth=1)
        if pos_row.label:  # positive example is correct
            ground_truth_labels.append(1)
        else:  # negative example is correct
            ground_truth_labels.append(0)

    return CCSDataset(
        positive_prompts=positive_prompts,
        negative_prompts=negative_prompts,
        ground_truth_labels=ground_truth_labels,
        dataset_name=dataset_name,
    )


def format_ag_news(dataset, prefix, prompt_idx):
    """Format AG News following paper's approach."""
    positive_prompts = []
    negative_prompts = []
    ground_truth_labels = []

    label_names = ["World politics", "Sports", "Business", "Science and technology"]

    # Paper uses 8 prompts for AG News
    templates = [
        "Consider the following example: '{text}' Choice 1: {label0}. Choice 2: {label1}. Between choice 1 and choice 2, the topic of this example is",
        "Consider the following example: '{text}' Choice 1: {label0}. Choice 2: {label1}. Between choice 1 and choice 2, what is the topic of this example?",
        "Consider the following example: '{text}' Which is the topic of this example, choice 1: {label0}, or choice 2: {label1}?",
        "{text} Choice 1: {label0}. Choice 2: {label1}. Between choice 1 and choice 2, the topic of this example is",
        "{text} Choice 1: {label0}. Choice 2: {label1}. Between choice 1 and choice 2, what is the topic of this example?",
        "{text} Which is the topic of this example, choice 1: {label0}, or choice 2: {label1}?",
        "{text} What label best describes this news article, choice 1: {label0}, or choice 2: {label1}?",
        "{text} Which section of a newspaper would this article likely appear in, choice 1: {label0}, or choice 2: {label1}?",
    ]

    template = templates[prompt_idx % len(templates)]

    for example in dataset:
        text = example["text"]
        true_label = label_names[example["label"]]

        # Randomly select one incorrect label
        incorrect_labels = [l for l in label_names if l != true_label]
        false_label = random.choice(incorrect_labels)

        # Randomly assign to label0 and label1
        if random.random() < 0.5:
            label0, label1 = true_label, false_label
            correct_choice = "choice 1"
        else:
            label0, label1 = false_label, true_label
            correct_choice = "choice 2"

        # Create base question
        question = template.format(text=text, label0=label0, label1=label1)

        # Create contrast pairs
        pos_prompt = f"{prefix}{question} choice 1"
        neg_prompt = f"{prefix}{question} choice 2"

        # Ground truth: 1 if choice 1 is correct, 0 if choice 2 is correct
        ground_truth = 1 if correct_choice == "choice 1" else 0

        positive_prompts.append(pos_prompt)
        negative_prompts.append(neg_prompt)
        ground_truth_labels.append(ground_truth)

    return positive_prompts, negative_prompts, ground_truth_labels


def format_imdb(dataset, prefix, prompt_idx):
    """Format IMDB following paper's approach."""
    positive_prompts = []
    negative_prompts = []
    ground_truth_labels = []

    # Paper uses 13 prompts for IMDB
    templates = [
        "Consider the following example: '{text}' Between negative and positive, the sentiment of this example is",
        "Consider the following example: '{text}' Between negative and positive, which is the sentiment of this example?",
        # Add more templates as specified in the paper
    ]

    template = templates[prompt_idx % len(templates)]

    for example in dataset:
        text = example["text"]
        # IMDB uses label=1 for positive, 0 for negative
        true_label = "positive" if example["label"] == 1 else "negative"

        # Create base question
        question = template.format(text=text)

        # Create contrast pairs
        pos_prompt = f"{prefix}{question} positive"
        neg_prompt = f"{prefix}{question} negative"

        # Ground truth: 1 if positive is correct, 0 if negative is correct
        ground_truth = 1 if true_label == "positive" else 0

        positive_prompts.append(pos_prompt)
        negative_prompts.append(neg_prompt)
        ground_truth_labels.append(ground_truth)

    return positive_prompts, negative_prompts, ground_truth_labels

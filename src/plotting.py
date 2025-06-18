import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score
import pandas as pd


def compute_auroc(results):
    """Compute AUROC for each method from results."""
    methods = list(results[0]["methods"].keys())
    auroc_scores = {}

    for method in methods:
        y_true = [r["truth_label"] for r in results if r["methods"][method] is not None]
        y_scores = [
            r["methods"][method] for r in results if r["methods"][method] is not None
        ]

        if len(set(y_true)) > 1:  # Need both classes
            auroc_scores[method] = max(
                roc_auc_score(y_true, y_scores),
                roc_auc_score(y_true, [-x for x in y_scores]),
            )
        else:
            auroc_scores[method] = None

    return auroc_scores


def plot_auroc_comparison(all_results_dict, figsize=(16, 8), style="modern"):
    """
    Create aesthetic bar plots comparing AUROC scores across datasets and methods.

    Parameters:
    all_results_dict: Dictionary with dataset names as keys and results as values
    figsize: Figure size tuple
    style: 'modern', 'classic', or 'minimal'
    """

    # Collect all AUROC scores
    plot_data = []
    for dataset_name, results in all_results_dict.items():
        auroc_scores = compute_auroc(results)
        for method, score in auroc_scores.items():
            if score is not None:
                plot_data.append(
                    {"Dataset": dataset_name, "Method": method, "AUROC": score}
                )

    df = pd.DataFrame(plot_data)

    if df.empty:
        print("No valid AUROC scores to plot.")
        return

    # Set style
    if style == "modern":
        plt.style.use("default")
        sns.set_palette("husl")
    elif style == "classic":
        plt.style.use("classic")
        sns.set_palette("deep")
    else:  # minimal
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_palette("muted")

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Get unique methods and datasets
    methods = df["Method"].unique()
    datasets = df["Dataset"].unique()

    # Create color palette and patterns
    colors = sns.color_palette("husl", len(methods))
    patterns = ["", "///", "...", "+++", "xxx", "ooo", "***"][: len(methods)]

    # Set up bar positions with wider bars and spacing between datasets
    x = np.arange(len(datasets)) * 1.5  # Add space between datasets
    width = 1.0 / len(methods)  # Make bars wider

    # Plot bars for each method
    for i, method in enumerate(methods):
        method_data = df[df["Method"] == method]
        scores = []

        for dataset in datasets:
            dataset_score = method_data[method_data["Dataset"] == dataset]["AUROC"]
            if len(dataset_score) > 0:
                scores.append(dataset_score.iloc[0])
            else:
                scores.append(0)  # or np.nan if you prefer gaps

        bars = ax.bar(
            x + i * width,
            scores,
            width,
            label=method,
            color=colors[i],
            alpha=0.8,
            hatch=patterns[i],
            edgecolor="white",
            linewidth=1.2,
        )

        # Remove value labels on bars (commented out)
        # for j, bar in enumerate(bars):
        #     if scores[j] > 0:  # Only label non-zero bars
        #         height = bar.get_height()
        #         ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
        #                f'{scores[j]:.3f}', ha='center', va='bottom',
        #                fontsize=9, fontweight='bold')

    # Customize the plot
    ax.set_xlabel("Datasets", fontsize=16, fontweight="bold")
    ax.set_ylabel("AUROC Score", fontsize=16, fontweight="bold")
    ax.set_title(
        "AUROC Performance Comparison Across Datasets and Methods",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )

    # Set x-axis
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=14)

    # Set y-axis
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="y", labelsize=12)

    # Customize legend - make it bigger
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=14,
        title="Methods",
        title_fontsize=16,
    )

    # Adjust layout
    plt.tight_layout()

    # Add subtle background color
    ax.set_facecolor("#fafafa")

    return fig, ax


def plot_auroc_heatmap(all_results_dict, figsize=(12, 8)):
    """
    Create a heatmap visualization of AUROC scores.
    """
    # Collect data for heatmap
    heatmap_data = {}
    datasets = list(all_results_dict.keys())
    all_methods = set()

    for dataset_name, results in all_results_dict.items():
        auroc_scores = compute_auroc(results)
        heatmap_data[dataset_name] = auroc_scores
        all_methods.update(auroc_scores.keys())

    # Create DataFrame for heatmap
    methods = sorted(list(all_methods))
    heatmap_matrix = []

    for method in methods:
        row = []
        for dataset in datasets:
            score = heatmap_data[dataset].get(method)
            row.append(score if score is not None else np.nan)
        heatmap_matrix.append(row)

    df_heatmap = pd.DataFrame(heatmap_matrix, index=methods, columns=datasets)

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)

    # Custom colormap
    cmap = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)

    sns.heatmap(
        df_heatmap,
        annot=True,
        fmt=".3f",
        cmap="RdYlBu_r",
        center=0.7,
        vmin=0.5,
        vmax=1.0,
        square=False,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )

    plt.title(
        "AUROC Heatmap: Methods vs Datasets", fontsize=16, fontweight="bold", pad=20
    )
    plt.xlabel("Datasets", fontsize=14, fontweight="bold")
    plt.ylabel("Methods", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    return fig, ax

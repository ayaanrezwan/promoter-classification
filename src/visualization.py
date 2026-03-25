"""
visualization.py — Figure Generation for Promoter Classification Project

This module creates all plots needed for the paper and GitHub README:
1. ROC curves (all three models overlaid)
2. Confusion matrix heatmaps
3. Feature importance bar charts (LR coefficients + RF Gini)
4. K-mer frequency comparison between classes
5. Model comparison summary bar chart
6. GC content distribution plot

All figures are saved to results/figures/ as both PNG (for the paper)
and SVG (for the website/README — scales without pixelation).

Author: Ayaonic
Project: Human Promoter Classification
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os

# Use a non-interactive backend so plots save without displaying
# (prevents issues when running on remote servers or in scripts)
matplotlib.use('Agg')

# Set global style — this applies to ALL plots created after this line
# 'seaborn-v0_8-whitegrid' gives clean backgrounds with subtle gridlines
# If this style isn't available, fall back to a safe default
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    plt.style.use('default')

# Color palette — consistent across all figures for visual coherence
# Using colorblind-friendly colors (important for accessibility)
COLORS = {
    'Logistic Regression': '#2196F3',  # Blue
    'SVM': '#FF5722',                   # Deep orange
    'Random Forest': '#4CAF50',         # Green
    'promoter': '#2196F3',              # Blue
    'non_promoter': '#FF5722',          # Deep orange
    'highlight': '#FFC107',             # Amber (for emphasis)
}

FIGURE_DIR = 'results/figures'


def setup_figure_dir():
    """Create the output directory if it doesn't exist."""
    os.makedirs(FIGURE_DIR, exist_ok=True)


# =============================================================================
# FIGURE 1: ROC CURVES
# =============================================================================

def plot_roc_curves(roc_data, test_results, save=True):
    """
    Plot ROC curves for all three models on a single figure.

    The ROC curve shows the tradeoff between True Positive Rate (sensitivity)
    and False Positive Rate (1 - specificity) across all classification
    thresholds. A curve that hugs the top-left corner is ideal.

    The diagonal dashed line represents random guessing (AUC = 0.5).
    The further a curve is above this line, the better the model.
    """
    setup_figure_dir()

    fig, ax = plt.subplots(figsize=(8, 7))

    for name, (fpr, tpr, _) in roc_data.items():
        auc_score = test_results[name]['roc_auc']
        color = COLORS.get(name, '#333333')
        ax.plot(fpr, tpr, color=color, linewidth=2.5,
                label=f'{name} (AUC = {auc_score:.4f})')

    # Random guessing baseline
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5,
            label='Random Guessing (AUC = 0.5)')

    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curves — Model Comparison', fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    # Add subtle grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        fig.savefig(os.path.join(FIGURE_DIR, 'roc_curves.png'),
                    dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(FIGURE_DIR, 'roc_curves.svg'),
                    bbox_inches='tight')
        print("Saved: roc_curves.png / .svg")

    plt.close(fig)
    return fig


# =============================================================================
# FIGURE 2: CONFUSION MATRICES
# =============================================================================

def plot_confusion_matrices(test_results, save=True):
    """
    Plot confusion matrices for all three models side by side.

    Confusion matrices show the raw counts of correct and incorrect
    predictions. The diagonal (top-left to bottom-right) contains
    correct predictions; off-diagonal cells are errors.

    We use a heatmap with annotations showing both the count and
    the percentage of each class.
    """
    setup_figure_dir()

    model_names = list(test_results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, name in enumerate(model_names):
        ax = axes[idx]
        cm = test_results[name]['confusion_matrix']

        # Calculate percentages for annotation
        # Normalize by row (actual class) to get per-class accuracy
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        # Create annotation strings with both count and percentage
        annotations = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotations[i, j] = f'{cm[i, j]}\n({cm_pct[i, j]:.1f}%)'

        # Plot heatmap
        sns.heatmap(
            cm, annot=annotations, fmt='', cmap='Blues',
            xticklabels=['Non-promoter', 'Promoter'],
            yticklabels=['Non-promoter', 'Promoter'],
            ax=ax, cbar=False,
            annot_kws={'size': 12, 'fontweight': 'bold'}
        )

        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title(f'{name}', fontsize=13, fontweight='bold')

    plt.suptitle('Confusion Matrices — Test Set Performance',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save:
        fig.savefig(os.path.join(FIGURE_DIR, 'confusion_matrices.png'),
                    dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(FIGURE_DIR, 'confusion_matrices.svg'),
                    bbox_inches='tight')
        print("Saved: confusion_matrices.png / .svg")

    plt.close(fig)
    return fig


# =============================================================================
# FIGURE 3: FEATURE IMPORTANCE — LOGISTIC REGRESSION COEFFICIENTS
# =============================================================================

def plot_lr_coefficients(importance_dict, top_n=20, save=True):
    """
    Plot logistic regression coefficients as a horizontal bar chart.

    Positive coefficients (right) = k-mer pushes prediction toward promoter.
    Negative coefficients (left) = k-mer pushes toward non-promoter.

    This is one of the most interpretable visualizations because you can
    directly read off which k-mers favor which class and by how much.
    """
    setup_figure_dir()

    if 'Logistic Regression' not in importance_dict:
        print("Logistic Regression not found in importance dict")
        return None

    lr_importance = importance_dict['Logistic Regression']

    # Take top N by absolute value
    top_kmers = lr_importance[:top_n]

    # Reverse so the most important is at the top of the bar chart
    kmers = [k for k, _ in top_kmers][::-1]
    coefficients = [v for _, v in top_kmers][::-1]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Color bars by direction
    bar_colors = [COLORS['promoter'] if c > 0 else COLORS['non_promoter']
                  for c in coefficients]

    bars = ax.barh(range(len(kmers)), coefficients, color=bar_colors,
                   edgecolor='white', linewidth=0.5, height=0.7)

    ax.set_yticks(range(len(kmers)))
    ax.set_yticklabels(kmers, fontsize=11, fontfamily='monospace')
    ax.set_xlabel('Coefficient Value', fontsize=13)
    ax.set_title(f'Logistic Regression — Top {top_n} K-mer Coefficients',
                 fontsize=15, fontweight='bold')

    # Add a vertical line at x=0
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['promoter'], label='Toward Promoter (+)'),
        Patch(facecolor=COLORS['non_promoter'], label='Toward Non-Promoter (−)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()

    if save:
        fig.savefig(os.path.join(FIGURE_DIR, 'lr_coefficients.png'),
                    dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(FIGURE_DIR, 'lr_coefficients.svg'),
                    bbox_inches='tight')
        print("Saved: lr_coefficients.png / .svg")

    plt.close(fig)
    return fig


# =============================================================================
# FIGURE 4: FEATURE IMPORTANCE — RANDOM FOREST GINI
# =============================================================================

def plot_rf_importance(importance_dict, top_n=20, save=True):
    """
    Plot Random Forest Gini importance as a horizontal bar chart.

    Unlike LR coefficients, Gini importance has no sign — it only tells
    you HOW important a feature is, not in which direction it pushes
    the prediction. All values are positive and sum to 1.0.
    """
    setup_figure_dir()

    if 'Random Forest' not in importance_dict:
        print("Random Forest not found in importance dict")
        return None

    rf_importance = importance_dict['Random Forest']
    top_kmers = rf_importance[:top_n]

    kmers = [k for k, _ in top_kmers][::-1]
    importances = [v for _, v in top_kmers][::-1]

    fig, ax = plt.subplots(figsize=(10, 8))

    bars = ax.barh(range(len(kmers)), importances,
                   color=COLORS['Random Forest'],
                   edgecolor='white', linewidth=0.5, height=0.7)

    ax.set_yticks(range(len(kmers)))
    ax.set_yticklabels(kmers, fontsize=11, fontfamily='monospace')
    ax.set_xlabel('Gini Importance', fontsize=13)
    ax.set_title(f'Random Forest — Top {top_n} K-mer Importances',
                 fontsize=15, fontweight='bold')

    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()

    if save:
        fig.savefig(os.path.join(FIGURE_DIR, 'rf_importance.png'),
                    dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(FIGURE_DIR, 'rf_importance.svg'),
                    bbox_inches='tight')
        print("Saved: rf_importance.png / .svg")

    plt.close(fig)
    return fig


# =============================================================================
# FIGURE 5: K-MER FREQUENCY COMPARISON BETWEEN CLASSES
# =============================================================================

def plot_kmer_frequency_comparison(X, labels, vocabulary, top_n=20, save=True):
    """
    Compare the mean k-mer frequencies between promoters and non-promoters
    for the most differentially frequent k-mers.

    This is a direct visualization of the raw signal that the models are
    learning from. Wider gaps between the two bars = more discriminating k-mer.
    """
    setup_figure_dir()

    promoter_mask = labels == 1
    non_promoter_mask = labels == 0

    mean_prom = X[promoter_mask].mean(axis=0)
    mean_non = X[non_promoter_mask].mean(axis=0)
    diffs = np.abs(mean_prom - mean_non)

    # Get top N most different k-mers
    top_indices = np.argsort(diffs)[::-1][:top_n]

    kmers = [vocabulary[i] for i in top_indices][::-1]
    prom_vals = [mean_prom[i] for i in top_indices][::-1]
    non_vals = [mean_non[i] for i in top_indices][::-1]

    fig, ax = plt.subplots(figsize=(10, 8))

    y_pos = np.arange(len(kmers))
    bar_height = 0.35

    bars1 = ax.barh(y_pos - bar_height/2, prom_vals, bar_height,
                    label='Promoter', color=COLORS['promoter'],
                    edgecolor='white', linewidth=0.5)
    bars2 = ax.barh(y_pos + bar_height/2, non_vals, bar_height,
                    label='Non-promoter', color=COLORS['non_promoter'],
                    edgecolor='white', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(kmers, fontsize=11, fontfamily='monospace')
    ax.set_xlabel('Mean Frequency', fontsize=13)
    ax.set_title(f'Top {top_n} Most Differentially Frequent K-mers',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()

    if save:
        fig.savefig(os.path.join(FIGURE_DIR, 'kmer_frequency_comparison.png'),
                    dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(FIGURE_DIR, 'kmer_frequency_comparison.svg'),
                    bbox_inches='tight')
        print("Saved: kmer_frequency_comparison.png / .svg")

    plt.close(fig)
    return fig


# =============================================================================
# FIGURE 6: MODEL COMPARISON BAR CHART
# =============================================================================

def plot_model_comparison(test_results, save=True):
    """
    Grouped bar chart comparing all metrics across models.
    This is the summary figure — a visual version of the comparison table.
    """
    setup_figure_dir()

    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    model_names = list(test_results.keys())

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(metrics))
    width = 0.25  # Width of each bar (3 models, so 0.25 each with gaps)

    for i, name in enumerate(model_names):
        values = [test_results[name][m] for m in metrics]
        color = COLORS.get(name, '#333333')
        bars = ax.bar(x + i * width, values, width, label=name,
                      color=color, edgecolor='white', linewidth=0.5)

        # Add value labels on top of each bar
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8,
                    fontweight='bold')

    ax.set_ylabel('Score', fontsize=13)
    ax.set_title('Model Performance Comparison — Test Set',
                 fontsize=15, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.legend(fontsize=11)

    # Set y-axis to start near the lowest value for better visual comparison
    all_values = [test_results[n][m] for n in model_names for m in metrics]
    y_min = min(all_values) - 0.02
    ax.set_ylim([y_min, 1.005])

    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    if save:
        fig.savefig(os.path.join(FIGURE_DIR, 'model_comparison.png'),
                    dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(FIGURE_DIR, 'model_comparison.svg'),
                    bbox_inches='tight')
        print("Saved: model_comparison.png / .svg")

    plt.close(fig)
    return fig


# =============================================================================
# FIGURE 7: GC CONTENT DISTRIBUTION
# =============================================================================

def plot_gc_distribution(sequences, labels, save=True):
    """
    Plot overlapping histograms of GC content for promoters vs non-promoters.

    This figure validates our GC-matching strategy. If the two distributions
    overlap heavily, the model can't rely on GC content alone and must learn
    actual sequence patterns.
    """
    setup_figure_dir()

    def gc_content(seq):
        return (seq.count('G') + seq.count('C')) / len(seq)

    gc_promoter = [gc_content(sequences[i])
                   for i in range(len(sequences)) if labels[i] == 1]
    gc_non_promoter = [gc_content(sequences[i])
                       for i in range(len(sequences)) if labels[i] == 0]

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.hist(gc_promoter, bins=50, alpha=0.6, color=COLORS['promoter'],
            label=f'Promoter (mean={np.mean(gc_promoter):.3f})',
            density=True, edgecolor='white', linewidth=0.5)
    ax.hist(gc_non_promoter, bins=50, alpha=0.6, color=COLORS['non_promoter'],
            label=f'Non-promoter (mean={np.mean(gc_non_promoter):.3f})',
            density=True, edgecolor='white', linewidth=0.5)

    ax.set_xlabel('GC Content', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title('GC Content Distribution — Promoter vs Non-Promoter',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        fig.savefig(os.path.join(FIGURE_DIR, 'gc_distribution.png'),
                    dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(FIGURE_DIR, 'gc_distribution.svg'),
                    bbox_inches='tight')
        print("Saved: gc_distribution.png / .svg")

    plt.close(fig)
    return fig


# =============================================================================
# MAIN: GENERATE ALL FIGURES
# =============================================================================

def generate_all_figures(results, sequences=None, labels=None, X=None,
                         vocabulary=None):
    """
    Generate all figures for the project.

    Parameters
    ----------
    results : dict from models.run_full_pipeline()
        Must contain: 'roc_data', 'test_results', 'importance_dict'
    sequences : list of str (optional, for GC distribution plot)
    labels : numpy array (optional, for GC distribution and kmer comparison)
    X : numpy array (optional, for kmer frequency comparison)
    vocabulary : list of str (optional, for kmer frequency comparison)
    """
    print("=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)

    # Figure 1: ROC curves
    print("\n[1/7] ROC curves...")
    plot_roc_curves(results['roc_data'], results['test_results'])

    # Figure 2: Confusion matrices
    print("[2/7] Confusion matrices...")
    plot_confusion_matrices(results['test_results'])

    # Figure 3: LR coefficients
    print("[3/7] Logistic regression coefficients...")
    plot_lr_coefficients(results['importance_dict'])

    # Figure 4: RF importance
    print("[4/7] Random forest importance...")
    plot_rf_importance(results['importance_dict'])

    # Figure 5: K-mer frequency comparison
    if X is not None and labels is not None and vocabulary is not None:
        print("[5/7] K-mer frequency comparison...")
        plot_kmer_frequency_comparison(X, labels, vocabulary)
    else:
        print("[5/7] Skipped (need X, labels, vocabulary)")

    # Figure 6: Model comparison
    print("[6/7] Model comparison bar chart...")
    plot_model_comparison(results['test_results'])

    # Figure 7: GC distribution
    if sequences is not None and labels is not None:
        print("[7/7] GC content distribution...")
        plot_gc_distribution(sequences, labels)
    else:
        print("[7/7] Skipped (need sequences and labels)")

    print("\n" + "=" * 60)
    print(f"ALL FIGURES SAVED TO {FIGURE_DIR}/")
    print("=" * 60)

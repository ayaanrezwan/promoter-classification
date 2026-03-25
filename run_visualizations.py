"""
run_visualizations.py — Generate all figures for the project.

Usage:
    python run_visualizations.py

This script loads the results from run_models.py and generates
all 7 figures for the paper and GitHub README.

Run this AFTER run_models.py has completed successfully.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import build_dataset
from src.encoding import encode_sequences, load_processed_data
from src.models import (
    prepare_data, get_models, tune_models,
    evaluate_on_test_set, extract_feature_importance
)
from src.visualization import generate_all_figures


def main():
    processed_dir = 'data/processed'
    fasta_path = 'data/raw/human_promoters.fasta'

    # Load processed data
    print("Loading data...")
    X, labels, vocabulary = load_processed_data(processed_dir)

    # We need the raw sequences for the GC distribution plot
    print("Loading raw sequences for GC plot...")
    seqs, raw_labels, headers = build_dataset(fasta_path, seq_length=300)

    # We need to re-run the model pipeline to get the results dict
    # (In a production project, you'd pickle the results object.
    #  For now, we re-run — it's fast since data is already processed.)
    print("\nRe-running model pipeline for results...")
    print("(This is needed to get ROC data and importance dicts)\n")

    # Split data with the SAME random state as before
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Tune models (this takes a few minutes)
    best_models = tune_models(X_train, y_train)

    # Evaluate
    test_results, roc_data = evaluate_on_test_set(best_models, X_test, y_test)

    # Feature importance
    importance_dict = extract_feature_importance(best_models, vocabulary)

    # Build results dict
    results = {
        'test_results': test_results,
        'roc_data': roc_data,
        'importance_dict': importance_dict,
    }

    # Generate all figures
    generate_all_figures(
        results,
        sequences=seqs,
        labels=labels,
        X=X,
        vocabulary=vocabulary
    )

    print("\nDone! Check results/figures/ for all plots.")


if __name__ == '__main__':
    main()

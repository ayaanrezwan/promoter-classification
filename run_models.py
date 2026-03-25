"""
run_models.py — Execute the full model training and evaluation pipeline.

Usage:
    python run_models.py

This script:
1. Loads the processed data (or builds it from scratch)
2. Runs cross-validation with default hyperparameters
3. Tunes hyperparameters via GridSearchCV
4. Evaluates on the held-out test set
5. Extracts feature importance from all three models
6. Saves results to results/metrics/

Expected runtime: 5-15 minutes depending on your machine
(most time is spent on SVM and Random Forest grid search)
"""

import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import build_dataset
from src.encoding import encode_sequences, save_processed_data, load_processed_data
from src.models import run_full_pipeline


def main():
    processed_dir = 'data/processed'
    fasta_path = 'data/raw/human_promoters.fasta'

    # Check if we already have processed data
    if os.path.exists(os.path.join(processed_dir, 'X.npy')):
        print("Loading pre-processed data...")
        X, labels, vocabulary = load_processed_data(processed_dir)
    else:
        print("Building dataset from scratch...")
        seqs, labels, headers = build_dataset(fasta_path, seq_length=300)
        X, vocabulary, kmer_idx = encode_sequences(seqs, k=3)
        save_processed_data(X, labels, vocabulary, processed_dir)

    # Run the full pipeline
    print("\n" + "=" * 60)
    print("STARTING MODEL TRAINING PIPELINE")
    print("=" * 60)

    results = run_full_pipeline(X, labels, vocabulary)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("\nResults saved to results/metrics/")
    print("Next step: run visualization.py to generate figures")


if __name__ == '__main__':
    main()

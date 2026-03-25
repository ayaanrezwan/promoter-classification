"""
data_loader.py — Sequence Acquisition and Processing Pipeline

This module handles:
1. Downloading human promoter sequences from EPDnew (Eukaryotic Promoter Database)
2. Generating negative (non-promoter) sequences
3. Parsing FASTA format files
4. Splitting data into train/test sets

Author: Ayaonic
Project: Human Promoter Classification
"""

import os
import random
import numpy as np
from collections import Counter


# =============================================================================
# SECTION 1: FASTA PARSING
# =============================================================================

def parse_fasta(filepath):
    """
    Parse a FASTA file and return a list of (header, sequence) tuples.

    FASTA format looks like this:
        >sequence_id some description text
        ATCGATCGATCGATCG
        ATCGATCGATCGATCG
        >next_sequence_id
        GGCCAATTGGCCAATT

    The '>' marks the start of a new sequence. Everything after '>' on that
    line is the header. All subsequent lines (until the next '>') are the
    sequence, which can be split across multiple lines.

    Parameters
    ----------
    filepath : str
        Path to a .fasta or .fa file.

    Returns
    -------
    list of tuples
        Each tuple is (header_string, sequence_string).
        Sequences are returned as uppercase with no whitespace.
    """
    sequences = []
    current_header = None
    current_seq_parts = []  # We accumulate lines here because sequences span multiple lines

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()  # Remove leading/trailing whitespace and newline chars

            if not line:
                continue  # Skip empty lines

            if line.startswith('>'):
                # We've hit a new sequence header.
                # First, save the previous sequence if there was one.
                if current_header is not None:
                    full_sequence = ''.join(current_seq_parts).upper()
                    sequences.append((current_header, full_sequence))

                # Start tracking the new sequence
                current_header = line[1:]  # Everything after the '>'
                current_seq_parts = []     # Reset the accumulator
            else:
                # This line is part of the current sequence
                current_seq_parts.append(line)

    # Don't forget the last sequence in the file!
    # This is a classic off-by-one pattern: the loop only saves a sequence
    # when it encounters the NEXT header, so the final sequence never
    # triggers that save.
    if current_header is not None:
        full_sequence = ''.join(current_seq_parts).upper()
        sequences.append((current_header, full_sequence))

    return sequences


# =============================================================================
# SECTION 2: SEQUENCE VALIDATION AND CLEANING
# =============================================================================

def clean_sequences(sequences, expected_length=None, valid_bases=set('ATCG')):
    """
    Filter out sequences that contain ambiguous bases or have unexpected lengths.

    Why this matters:
    -----------------
    Real genomic data is messy. You'll encounter:
    - 'N' bases: positions where the sequencer couldn't determine the nucleotide
    - IUPAC ambiguity codes: 'R' (A or G), 'Y' (C or T), 'S' (G or C), etc.
    - Sequences of different lengths due to annotation inconsistencies

    For k-mer encoding, we need clean sequences with only A, T, C, G because:
    1. Our k-mer vocabulary is built from a 4-letter alphabet
    2. Ambiguous bases would create k-mers not in our vocabulary
    3. Variable-length sequences would produce different numbers of k-mers,
       but after normalization to frequencies this is actually okay — however,
       keeping fixed length is cleaner for this project.

    Parameters
    ----------
    sequences : list of (header, sequence) tuples
    expected_length : int or None
        If specified, only keep sequences of exactly this length.
    valid_bases : set
        Set of allowed characters. Default is {'A', 'T', 'C', 'G'}.

    Returns
    -------
    list of (header, sequence) tuples — filtered
    """
    cleaned = []
    removed_count = 0

    for header, seq in sequences:
        # Check 1: Are all bases in our valid set?
        if not set(seq).issubset(valid_bases):
            removed_count += 1
            continue

        # Check 2: Is the sequence the expected length?
        if expected_length is not None and len(seq) != expected_length:
            removed_count += 1
            continue

        cleaned.append((header, seq))

    print(f"Cleaning: kept {len(cleaned)}/{len(sequences)} sequences "
          f"(removed {removed_count})")

    return cleaned


# =============================================================================
# SECTION 3: NEGATIVE SET GENERATION
# =============================================================================

def generate_negative_sequences(n_sequences, seq_length, gc_content_target=None,
                                 promoter_sequences=None, random_seed=42):
    """
    Generate non-promoter (negative) sequences.

    Strategy:
    ---------
    We generate random DNA sequences, but with a critical nuance: we MATCH
    the GC content distribution of the promoter (positive) sequences.

    Why GC matching matters:
    If promoters have ~60% GC content and our random negatives have ~50%
    (which is what you'd get from a uniform random generator), then the model
    could achieve high accuracy by simply learning "high GC = promoter."
    That's not learning promoter biology — it's learning a statistical artifact
    of how we constructed the dataset. This is a form of DATA LEAKAGE.

    By matching GC content, we force the model to learn actual SEQUENCE PATTERNS
    (motifs, k-mer combinations) rather than just nucleotide composition.

    Parameters
    ----------
    n_sequences : int
        How many negative sequences to generate.
    seq_length : int
        Length of each sequence.
    gc_content_target : float or None
        If specified, generate sequences with approximately this GC%.
        If None and promoter_sequences provided, match the promoter distribution.
        If both None, use 0.5 (uniform).
    promoter_sequences : list of str or None
        Actual promoter sequences to compute GC distribution from.
    random_seed : int
        For reproducibility. Always set this.

    Returns
    -------
    list of (header, sequence) tuples
    """
    rng = random.Random(random_seed)  # Local RNG instance so we don't affect global state

    # Determine target GC content
    if gc_content_target is not None:
        gc_target = gc_content_target
    elif promoter_sequences is not None:
        # Calculate the mean GC content of the promoter set
        gc_values = [_gc_content(seq) for seq in promoter_sequences]
        gc_target = np.mean(gc_values)
        print(f"Promoter mean GC content: {gc_target:.3f}")
        print(f"Promoter GC std dev: {np.std(gc_values):.3f}")
    else:
        gc_target = 0.5

    print(f"Generating {n_sequences} negative sequences with target GC = {gc_target:.3f}")

    negatives = []
    for i in range(n_sequences):
        seq = _generate_gc_matched_sequence(seq_length, gc_target, rng)
        header = f"negative_random_{i:05d} | synthetic | gc_target={gc_target:.3f}"
        negatives.append((header, seq))

    return negatives


def _gc_content(sequence):
    """
    Calculate the GC content of a DNA sequence.

    GC content = (count of G + count of C) / total length

    This is one of the most fundamental sequence statistics in genomics.
    Different genomic regions have characteristic GC contents:
    - Human genome average: ~41%
    - CpG islands (often at promoters): >50%
    - AT-rich regions (heterochromatin): <35%
    """
    gc_count = sequence.count('G') + sequence.count('C')
    return gc_count / len(sequence)


def _generate_gc_matched_sequence(length, gc_target, rng):
    """
    Generate a single random sequence with approximately the target GC content.

    Method: For each position, we draw from {G, C} with probability gc_target,
    and from {A, T} with probability (1 - gc_target). Within each pair,
    the choice is uniform (50/50 between G and C, 50/50 between A and T).
    """
    sequence = []
    for _ in range(length):
        if rng.random() < gc_target:
            # This position is G or C
            sequence.append(rng.choice(['G', 'C']))
        else:
            # This position is A or T
            sequence.append(rng.choice(['A', 'T']))
    return ''.join(sequence)


# =============================================================================
# SECTION 4: DATASET CONSTRUCTION
# =============================================================================

def build_dataset(positive_fasta_path, n_negatives=None, seq_length=300,
                  random_seed=42):
    """
    Build the complete labeled dataset from a promoter FASTA file.

    This is the main entry point that chains together:
    1. Parse the FASTA file
    2. Clean the sequences
    3. Generate GC-matched negatives
    4. Combine and label everything

    Parameters
    ----------
    positive_fasta_path : str
        Path to FASTA file containing promoter sequences.
    n_negatives : int or None
        Number of negatives to generate. If None, matches the number of positives
        (balanced dataset). This is important — imbalanced classes affect model
        evaluation metrics significantly.
    seq_length : int
        Expected sequence length. Sequences not matching this are removed.
    random_seed : int
        For reproducibility.

    Returns
    -------
    sequences : list of str
        All DNA sequences (positives + negatives).
    labels : numpy array
        1 for promoter, 0 for non-promoter.
    headers : list of str
        Sequence identifiers.
    """
    # Step 1: Load and clean promoter sequences
    print("=" * 60)
    print("LOADING POSITIVE (PROMOTER) SEQUENCES")
    print("=" * 60)
    raw_positives = parse_fasta(positive_fasta_path)
    print(f"Raw sequences loaded: {len(raw_positives)}")

    positives = clean_sequences(raw_positives, expected_length=seq_length)

    # Step 2: Generate negatives
    print("\n" + "=" * 60)
    print("GENERATING NEGATIVE (NON-PROMOTER) SEQUENCES")
    print("=" * 60)
    if n_negatives is None:
        n_negatives = len(positives)  # Balanced classes
        print(f"Matching positive count: {n_negatives} negatives")

    promoter_seqs = [seq for _, seq in positives]
    negatives = generate_negative_sequences(
        n_sequences=n_negatives,
        seq_length=seq_length,
        promoter_sequences=promoter_seqs,
        random_seed=random_seed
    )

    # Step 3: Combine and label
    print("\n" + "=" * 60)
    print("BUILDING FINAL DATASET")
    print("=" * 60)
    all_headers = [h for h, _ in positives] + [h for h, _ in negatives]
    all_sequences = [s for _, s in positives] + [s for _, s in negatives]
    labels = np.array([1] * len(positives) + [0] * len(negatives))

    print(f"Total sequences: {len(all_sequences)}")
    print(f"  Promoters (1):     {np.sum(labels == 1)}")
    print(f"  Non-promoters (0): {np.sum(labels == 0)}")
    print(f"  Class balance:     {np.mean(labels):.3f} "
          f"(0.5 = perfectly balanced)")

    return all_sequences, labels, all_headers


# =============================================================================
# SECTION 5: DATA SUMMARY STATISTICS
# =============================================================================

def sequence_summary(sequences, labels):
    """
    Print summary statistics for the dataset.
    Useful for EDA (Exploratory Data Analysis) and sanity checking.
    """
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    for label_val, label_name in [(1, "Promoters"), (0, "Non-promoters")]:
        mask = labels == label_val
        subset = [sequences[i] for i in range(len(sequences)) if mask[i]]

        lengths = [len(s) for s in subset]
        gc_values = [_gc_content(s) for s in subset]

        # Nucleotide frequencies
        all_bases = ''.join(subset)
        base_counts = Counter(all_bases)
        total_bases = len(all_bases)

        print(f"\n--- {label_name} (n={len(subset)}) ---")
        print(f"  Length:  min={min(lengths)}, max={max(lengths)}, "
              f"mean={np.mean(lengths):.1f}")
        print(f"  GC%:    min={min(gc_values):.3f}, max={max(gc_values):.3f}, "
              f"mean={np.mean(gc_values):.3f} ± {np.std(gc_values):.3f}")
        print(f"  Bases:  ", end="")
        for base in 'ATCG':
            pct = base_counts.get(base, 0) / total_bases * 100
            print(f"{base}={pct:.1f}%  ", end="")
        print()

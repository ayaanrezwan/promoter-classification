"""
encoding.py — K-mer Feature Encoding Pipeline

This module converts raw DNA sequences into numerical feature vectors
that machine learning models can process.

The core idea:
    DNA sequence (string) → k-mer frequency vector (numpy array)

    "ATCGATCG" with k=3 → count all 64 possible 3-mers → normalize to frequencies

This is a "bag of words" approach borrowed from NLP (Natural Language Processing).
Just like you can represent a document by counting word frequencies, you can
represent a DNA sequence by counting k-mer frequencies. The model then learns
which k-mer frequency patterns distinguish promoters from non-promoters.

Author: Ayaonic
Project: Human Promoter Classification
"""

import numpy as np
from itertools import product
import os 


# =============================================================================
# SECTION 1: VOCABULARY GENERATION
# =============================================================================

def generate_kmer_vocabulary(k):
    """
    Generate all possible k-mers for a DNA alphabet {A, T, C, G}.

    For k=3, this produces 4^3 = 64 k-mers:
    ['AAA', 'AAT', 'AAC', 'AAG', 'ATA', 'ATT', ..., 'GGG']

    We need a FIXED vocabulary so that every sequence gets encoded into
    a vector of the same length, with each position always corresponding
    to the same k-mer. This is critical — if the mapping between position
    and k-mer isn't consistent, the features are meaningless.

    Parameters
    ----------
    k : int
        Length of each k-mer.

    Returns
    -------
    vocabulary : list of str
        All possible k-mers, sorted lexicographically.
    kmer_to_index : dict
        Maps each k-mer string to its position in the feature vector.
        Example: {'AAA': 0, 'AAC': 1, 'AAG': 2, 'AAT': 3, ...}

    How itertools.product works:
    ----------------------------
    product('ATCG', repeat=3) generates the Cartesian product:
    ('A','A','A'), ('A','A','T'), ('A','A','C'), ('A','A','G'),
    ('A','T','A'), ('A','T','T'), ...

    It's equivalent to nested for loops:
    for b1 in 'ATCG':
        for b2 in 'ATCG':
            for b3 in 'ATCG':
                yield (b1, b2, b3)

    But product() generalizes to any k without writing k nested loops.
    """
    bases = ['A', 'T', 'C', 'G']

    # Generate all combinations and join each tuple into a string
    vocabulary = sorted([''.join(combo) for combo in product(bases, repeat=k)])

    # Create the lookup dictionary: k-mer → index
    kmer_to_index = {kmer: idx for idx, kmer in enumerate(vocabulary)}

    print(f"Generated {k}-mer vocabulary: {len(vocabulary)} features")
    print(f"  First 5: {vocabulary[:5]}")
    print(f"  Last 5:  {vocabulary[-5:]}")

    return vocabulary, kmer_to_index


# =============================================================================
# SECTION 2: SINGLE SEQUENCE ENCODING
# =============================================================================

def count_kmers(sequence, k, kmer_to_index):
    """
    Count the occurrences of each k-mer in a single DNA sequence.

    The sliding window approach:
    ----------------------------
    For sequence "ATCGATCG" with k=3:

        Position 0: ATC
        Position 1: TCG
        Position 2: CGA
        Position 3: GAT
        Position 4: ATC  (same as position 0!)
        Position 5: TCG  (same as position 1!)

    Total windows = len(sequence) - k + 1 = 8 - 3 + 1 = 6

    We slide a window of size k across the sequence, one base at a time,
    and increment the count for whatever k-mer we see at each position.

    Parameters
    ----------
    sequence : str
        A DNA sequence (must contain only A, T, C, G).
    k : int
        K-mer length.
    kmer_to_index : dict
        Maps k-mer strings to indices in the count vector.

    Returns
    -------
    counts : numpy array of shape (4^k,)
        Raw counts of each k-mer.
    """
    n_kmers = len(kmer_to_index)  # 4^k
    counts = np.zeros(n_kmers, dtype=np.float64)

    # Slide the window across the sequence
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]  # Extract the k-mer at this position

        # Look up its index in our vocabulary
        if kmer in kmer_to_index:
            counts[kmer_to_index[kmer]] += 1
        # If the k-mer contains characters outside ATCG (shouldn't happen
        # after cleaning, but defensive programming is good practice),
        # we simply skip it.

    return counts


def encode_sequence(sequence, k, kmer_to_index, normalize=True):
    """
    Encode a single DNA sequence as a k-mer frequency vector.

    Why normalize?
    --------------
    Raw counts depend on sequence length:
    - A 300bp sequence has 298 3-mers
    - A 500bp sequence has 498 3-mers

    If we used raw counts, longer sequences would have higher values everywhere,
    and the model might learn "higher counts = class X" when really it's just
    "longer sequence = class X."

    Normalization converts counts to FREQUENCIES (proportions), which are
    comparable across sequences of different lengths.

    The normalized vector sums to 1.0, making it a discrete probability
    distribution over k-mers. You can interpret each value as:
    "what fraction of all k-mers in this sequence are this particular k-mer?"

    Parameters
    ----------
    sequence : str
    k : int
    kmer_to_index : dict
    normalize : bool
        If True, divide by total k-mer count to get frequencies.

    Returns
    -------
    features : numpy array of shape (4^k,)
    """
    counts = count_kmers(sequence, k, kmer_to_index)

    if normalize:
        total = counts.sum()
        if total > 0:
            counts = counts / total
        # If total is 0 (empty sequence), we return all zeros.
        # This shouldn't happen with proper input, but again: defensive programming.

    return counts


# =============================================================================
# SECTION 3: BATCH ENCODING (THE MAIN FUNCTION YOU'LL CALL)
# =============================================================================

def encode_sequences(sequences, k=3, normalize=True):
    """
    Encode a list of DNA sequences into a feature matrix.

    This is the main entry point for encoding. It takes raw sequences and
    returns a matrix ready for scikit-learn.

    The output matrix X has shape (n_samples, n_features):
    - Each ROW is one sequence
    - Each COLUMN is one k-mer
    - Each VALUE is the frequency of that k-mer in that sequence

    Example with k=3 (64 features) and 1000 sequences:
        X.shape = (1000, 64)

    This matrix is what gets fed directly into our models:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    Parameters
    ----------
    sequences : list of str
        DNA sequences (already cleaned — only ATCG characters).
    k : int
        K-mer size. Default 3.
    normalize : bool
        Whether to normalize counts to frequencies.

    Returns
    -------
    X : numpy array of shape (n_sequences, 4^k)
        Feature matrix.
    vocabulary : list of str
        K-mer names corresponding to each column.
    kmer_to_index : dict
        K-mer to column index mapping.
    """
    # Step 1: Build the vocabulary
    vocabulary, kmer_to_index = generate_kmer_vocabulary(k)
    n_features = len(vocabulary)

    # Step 2: Pre-allocate the feature matrix
    # Why pre-allocate instead of appending rows? Performance.
    # np.vstack() or list appending copies the entire array each time → O(n²)
    # Pre-allocation fills in place → O(n)
    # For 1000 sequences this doesn't matter, but it's a good habit.
    X = np.zeros((len(sequences), n_features), dtype=np.float64)

    # Step 3: Encode each sequence
    for i, seq in enumerate(sequences):
        X[i, :] = encode_sequence(seq, k, kmer_to_index, normalize=normalize)

        # Progress indicator for large datasets
        if (i + 1) % 1000 == 0:
            print(f"  Encoded {i+1}/{len(sequences)} sequences...")

    print(f"\nEncoding complete:")
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  (rows = sequences, columns = {k}-mer features)")

    # Sanity check: if normalized, each row should sum to ~1.0
    if normalize:
        row_sums = X.sum(axis=1)
        print(f"  Row sums (should be ~1.0): "
              f"min={row_sums.min():.6f}, max={row_sums.max():.6f}")

    return X, vocabulary, kmer_to_index


# =============================================================================
# SECTION 4: FEATURE ANALYSIS HELPERS
# =============================================================================

def kmer_frequency_comparison(X, labels, vocabulary):
    """
    Compare k-mer frequencies between promoters and non-promoters.

    This is the foundation of feature importance analysis. Before we even
    train a model, we can see which k-mers have different frequency
    distributions between the two classes.

    A k-mer with VERY different mean frequencies between classes is likely
    to be important for classification. This is essentially a univariate
    feature analysis.

    Parameters
    ----------
    X : numpy array of shape (n_samples, n_features)
    labels : numpy array of shape (n_samples,)
    vocabulary : list of str

    Returns
    -------
    comparison : list of dicts, sorted by absolute frequency difference
        Each dict: {'kmer', 'mean_promoter', 'mean_non_promoter', 'diff', 'abs_diff'}
    """
    promoter_mask = labels == 1
    non_promoter_mask = labels == 0

    # Calculate mean frequency of each k-mer for each class
    mean_promoter = X[promoter_mask].mean(axis=0)
    mean_non_promoter = X[non_promoter_mask].mean(axis=0)

    comparison = []
    for i, kmer in enumerate(vocabulary):
        diff = mean_promoter[i] - mean_non_promoter[i]
        comparison.append({
            'kmer': kmer,
            'mean_promoter': mean_promoter[i],
            'mean_non_promoter': mean_non_promoter[i],
            'diff': diff,
            'abs_diff': abs(diff)
        })

    # Sort by absolute difference (most distinguishing k-mers first)
    comparison.sort(key=lambda x: x['abs_diff'], reverse=True)

    # Print top 10
    print("\nTop 10 most distinguishing k-mers:")
    print(f"  {'K-mer':<8} {'Promoter':>10} {'Non-promoter':>13} {'Diff':>8}")
    print("  " + "-" * 43)
    for entry in comparison[:10]:
        print(f"  {entry['kmer']:<8} {entry['mean_promoter']:>10.5f} "
              f"{entry['mean_non_promoter']:>13.5f} {entry['diff']:>+8.5f}")

    return comparison


# =============================================================================
# SECTION 5: SAVING AND LOADING PROCESSED DATA
# =============================================================================

def save_processed_data(X, labels, vocabulary, output_dir='data/processed'):
    """
    Save the encoded feature matrix and labels for later use.

    We save as .npy (NumPy binary format) because:
    1. It's much faster to load than re-encoding from raw sequences
    2. It preserves exact floating-point values (no CSV rounding issues)
    3. It's compact

    The vocabulary is saved as a text file so it's human-readable.
    """
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, 'X.npy'), X)
    np.save(os.path.join(output_dir, 'labels.npy'), labels)

    vocab_path = os.path.join(output_dir, 'vocabulary.txt')
    with open(vocab_path, 'w') as f:
        f.write('\n'.join(vocabulary))

    print(f"\nSaved processed data to {output_dir}/")
    print(f"  X.npy:          {X.shape}")
    print(f"  labels.npy:     {labels.shape}")
    print(f"  vocabulary.txt: {len(vocabulary)} k-mers")


def load_processed_data(data_dir='data/processed'):
    """Load previously saved processed data."""
    X = np.load(os.path.join(data_dir, 'X.npy'))
    labels = np.load(os.path.join(data_dir, 'labels.npy'))

    vocab_path = os.path.join(data_dir, 'vocabulary.txt')
    with open(vocab_path, 'r') as f:
        vocabulary = [line.strip() for line in f.readlines()]

    print(f"Loaded: X={X.shape}, labels={labels.shape}, "
          f"vocab={len(vocabulary)} k-mers")

    return X, labels, vocabulary

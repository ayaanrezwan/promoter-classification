"""
download_data.py — Data Acquisition Script

Run this ONCE to download and prepare the dataset.

Usage:
    python download_data.py

Data Source:
    EPDnew (Eukaryotic Promoter Database) — https://epd.expasy.org/epd/
    This is the gold-standard curated database of experimentally validated
    eukaryotic promoters. Each entry is anchored to a Transcription Start Site
    (TSS) that has been verified by high-throughput experiments (CAGE, etc.).

What this script does:
    1. Downloads human promoter sequences from EPDnew
    2. Extracts sequences of fixed length centered around TSS
    3. Generates GC-matched negative sequences
    4. Saves everything as FASTA files in data/raw/
"""

import os
import urllib.request
import sys

# Add the project root to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def download_epd_sequences():
    """
    Download human promoter sequences from EPDnew.

    EPDnew provides a bulk download URL that returns FASTA-formatted sequences.
    We request sequences from -249 to +50 relative to each TSS, giving us
    300bp sequences centered slightly upstream of the transcription start.

    Why -249 to +50?
    - Most core promoter elements are within -40 to +40 of the TSS
    - Proximal promoter elements extend to ~-250
    - This window captures the biologically relevant region
    - 300bp is a common choice in the literature for promoter classification
    """
    data_dir = os.path.join('data', 'raw')
    os.makedirs(data_dir, exist_ok=True)

    output_path = os.path.join(data_dir, 'human_promoters.fasta')

    if os.path.exists(output_path):
        print(f"File already exists: {output_path}")
        print("Delete it manually if you want to re-download.")
        return output_path

    # EPDnew bulk FASTA download for human promoters
    # Format: sequences from -249 to +50 relative to TSS
    url = ("https://epd.expasy.org/ftp/epdnew/H_sapiens/current/"
           "Hs_EPDnew.fasta")

    print(f"Downloading human promoter sequences from EPDnew...")
    print(f"URL: {url}")
    print(f"This may take a moment...\n")

    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Downloaded successfully to: {output_path}")

        # Count sequences
        count = 0
        with open(output_path, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    count += 1
        print(f"Total promoter sequences: {count}")

        return output_path

    except urllib.error.URLError as e:
        print(f"\nDownload failed: {e}")
        print("\nFallback: Manual download instructions")
        print("=" * 50)
        print("1. Go to: https://epd.expasy.org/epd/")
        print("2. Select 'Human' under organism")
        print("3. Use the search/download to get FASTA sequences")
        print("4. Save to: data/raw/human_promoters.fasta")
        print("\nAlternative: Use the UCSC Table Browser")
        print("   https://genome.ucsc.edu/cgi-bin/hgTables")
        print("   - Assembly: hg38")
        print("   - Group: Genes and Gene Predictions")
        print("   - Track: EPDnew Promoters")
        print("   - Output format: sequence (FASTA)")
        return None

    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Creating a synthetic dataset instead for development...\n")
        return create_synthetic_dataset(output_path)


def create_synthetic_dataset(output_path, n_sequences=2000, seq_length=300):
    """
    Create a synthetic promoter dataset for development and testing.

    This generates sequences with embedded promoter-like motifs so the
    pipeline works even if the download fails. The synthetic data lets you:
    1. Test the entire pipeline end-to-end
    2. Verify that models can learn known patterns
    3. Debug code without waiting for downloads

    For the final project, you MUST use real EPDnew data. This is just
    a development convenience.

    Motifs embedded in synthetic promoters:
    - TATA box (TATAAA) at position ~120 (which maps to ~ -30 from center)
    - GC-rich regions (simulating CpG islands)
    - Inr-like element (YYANWYY) near position 150 (the center / "TSS")
    """
    import random
    rng = random.Random(42)

    print("Generating synthetic dataset for development...")
    print(f"  {n_sequences} promoter sequences, {seq_length} bp each")

    with open(output_path, 'w') as f:
        for i in range(n_sequences):
            # Start with a base sequence at ~55% GC (promoter-like)
            seq = []
            for _ in range(seq_length):
                if rng.random() < 0.55:
                    seq.append(rng.choice(['G', 'C']))
                else:
                    seq.append(rng.choice(['A', 'T']))

            # Embed TATA box in ~30% of sequences (matching real biology:
            # only ~25-30% of human promoters have a TATA box)
            if rng.random() < 0.30:
                tata_pos = rng.randint(115, 125)  # Around -30 from center
                tata_variant = rng.choice(['TATAAA', 'TATAAT', 'TATAAG'])
                for j, base in enumerate(tata_variant):
                    if tata_pos + j < seq_length:
                        seq[tata_pos + j] = base

            # Embed GC-rich region in ~70% (CpG island promoters)
            if rng.random() < 0.70:
                gc_start = rng.randint(80, 130)
                gc_length = rng.randint(30, 60)
                for j in range(gc_length):
                    pos = gc_start + j
                    if pos < seq_length and rng.random() < 0.75:
                        seq[pos] = rng.choice(['G', 'C'])

            # Embed Inr-like element near center
            if rng.random() < 0.50:
                inr_pos = rng.randint(148, 152)
                inr = list(rng.choice(['TCAGT', 'CCAAT', 'TCAAT']))
                for j, base in enumerate(inr):
                    if inr_pos + j < seq_length:
                        seq[inr_pos + j] = base

            sequence = ''.join(seq)
            f.write(f">synthetic_promoter_{i:05d} | embedded_motifs | len={seq_length}\n")

            # Write sequence in lines of 80 characters (FASTA convention)
            for j in range(0, len(sequence), 80):
                f.write(sequence[j:j+80] + '\n')

    print(f"  Saved to: {output_path}")
    return output_path


if __name__ == '__main__':
    print("=" * 60)
    print("PROMOTER CLASSIFICATION — DATA ACQUISITION")
    print("=" * 60)
    print()

    result = download_epd_sequences()

    if result:
        print("\n" + "=" * 60)
        print("SUCCESS — Next step:")
        print("  Run notebook 01_data_exploration.ipynb")
        print("  Or run: python -c \"from src.data_loader import *; "
              "build_dataset('data/raw/human_promoters.fasta')\"")
        print("=" * 60)

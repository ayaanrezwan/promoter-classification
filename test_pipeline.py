from src.data_loader import build_dataset, sequence_summary
from src.encoding import encode_sequences

# Step 1: Build the dataset
seqs, labels, headers = build_dataset('data/raw/human_promoters.fasta', seq_length=300)

# Step 2: Print summary stats
sequence_summary(seqs, labels)

# Step 3: Encode with 3-mers
X, vocab, kmer_idx = encode_sequences(seqs, k=3)

print(f"\nFeature matrix ready: {X.shape}")
print("Pipeline working correctly!")
#!/usr/bin/env python3
"""
Create a large test FASTA file to test batch processing
"""

import random

def generate_dna_sequence(length):
    """Generate a random DNA sequence"""
    bases = ['A', 'T', 'G', 'C']
    return ''.join(random.choice(bases) for _ in range(length))

def create_large_fasta(filename, num_sequences=150):
    """Create a large FASTA file with many sequences"""
    
    sequence_types = [
        'Marine_Deep_Sea',
        'Novel_Taxa',
        'Eukaryotic_18S',
        'Environmental_Sample',
        'Unknown_Species',
        'Bacterial_Sample',
        'Cnidaria_Sample',
        'Mollusca_Sample',
        'Arthropoda_Sample'
    ]
    
    with open(filename, 'w') as f:
        for i in range(1, num_sequences + 1):
            seq_type = random.choice(sequence_types)
            seq_length = random.randint(100, 300)
            sequence = generate_dna_sequence(seq_length)
            
            f.write(f">{seq_type}_{i:03d}\n")
            f.write(f"{sequence}\n")
    
    print(f"Created {filename} with {num_sequences} sequences")

if __name__ == "__main__":
    create_large_fasta("large_test_dataset.fasta", 150)
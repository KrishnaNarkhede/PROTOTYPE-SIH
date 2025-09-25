#!/usr/bin/env python3
"""
eDNA Processing Pipeline for High-Throughput Analysis
===================================================

Implements:
- FASTQ file processing
- Quality filtering and denoising
- Bulk sequence handling
- eDNA-specific optimizations
"""

import os
import gzip
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

class eDNAProcessor:
    """High-throughput eDNA sequence processor"""
    
    def __init__(self):
        self.quality_threshold = 20
        self.min_length = 50
        self.max_length = 500
        self.processed_count = 0
        
    def process_fastq_file(self, fastq_path):
        """Process FASTQ file with quality filtering"""
        sequences = []
        
        try:
            if fastq_path.endswith('.gz'):
                file_handle = gzip.open(fastq_path, 'rt')
            else:
                file_handle = open(fastq_path, 'r')
            
            with file_handle as f:
                while True:
                    # Read FASTQ record (4 lines)
                    header = f.readline().strip()
                    if not header:
                        break
                    
                    sequence = f.readline().strip()
                    plus = f.readline().strip()
                    quality = f.readline().strip()
                    
                    if self._passes_quality_filter(sequence, quality):
                        sequences.append({
                            'id': header[1:],  # Remove @
                            'sequence': sequence.upper(),
                            'quality': quality,
                            'length': len(sequence)
                        })
                        
                        self.processed_count += 1
            
            return self._denoise_sequences(sequences)
            
        except Exception as e:
            print(f"Error processing FASTQ: {e}")
            return []
    
    def _passes_quality_filter(self, sequence, quality_string):
        """Check if sequence passes quality filters"""
        # Length filter
        if len(sequence) < self.min_length or len(sequence) > self.max_length:
            return False
        
        # Quality score filter (Phred scores)
        if len(quality_string) != len(sequence):
            return False
        
        avg_quality = sum(ord(q) - 33 for q in quality_string) / len(quality_string)
        if avg_quality < self.quality_threshold:
            return False
        
        # Ambiguous base filter
        n_count = sequence.count('N')
        if n_count / len(sequence) > 0.1:  # Max 10% N's
            return False
        
        return True
    
    def _denoise_sequences(self, sequences):
        """Remove noise and artifacts from sequences"""
        if len(sequences) < 2:
            return sequences
        
        # Group similar sequences for denoising
        sequence_groups = defaultdict(list)
        
        for seq in sequences:
            # Use first 20 bases as grouping key
            key = seq['sequence'][:20]
            sequence_groups[key].append(seq)
        
        denoised = []
        for group in sequence_groups.values():
            if len(group) == 1:
                denoised.extend(group)
            else:
                # Keep the highest quality sequence from each group
                best_seq = max(group, key=lambda x: self._calculate_avg_quality(x['quality']))
                denoised.append(best_seq)
        
        return denoised
    
    def _calculate_avg_quality(self, quality_string):
        """Calculate average quality score"""
        return sum(ord(q) - 33 for q in quality_string) / len(quality_string)
    
    def process_bulk_sequences(self, file_paths):
        """Process multiple files in bulk"""
        all_sequences = []
        processing_stats = {
            'files_processed': 0,
            'total_sequences': 0,
            'filtered_sequences': 0,
            'quality_stats': {}
        }
        
        for file_path in file_paths:
            if file_path.endswith(('.fastq', '.fq', '.fastq.gz', '.fq.gz')):
                sequences = self.process_fastq_file(file_path)
            else:
                sequences = self._process_fasta_file(file_path)
            
            all_sequences.extend(sequences)
            processing_stats['files_processed'] += 1
            processing_stats['total_sequences'] += len(sequences)
        
        # Additional bulk processing
        processed_sequences = self._bulk_quality_control(all_sequences)
        processing_stats['filtered_sequences'] = len(processed_sequences)
        processing_stats['quality_stats'] = self._generate_quality_stats(processed_sequences)
        
        return processed_sequences, processing_stats
    
    def _process_fasta_file(self, fasta_path):
        """Process FASTA file for bulk analysis"""
        sequences = []
        current_id = None
        current_seq = []
        
        try:
            with open(fasta_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('>'):
                        if current_id:
                            seq_str = ''.join(current_seq).upper()
                            if self.min_length <= len(seq_str) <= self.max_length:
                                sequences.append({
                                    'id': current_id,
                                    'sequence': seq_str,
                                    'length': len(seq_str),
                                    'quality': 'I' * len(seq_str)  # Mock quality
                                })
                        current_id = line[1:]
                        current_seq = []
                    elif current_id and line:
                        current_seq.append(line)
                
                # Process last sequence
                if current_id:
                    seq_str = ''.join(current_seq).upper()
                    if self.min_length <= len(seq_str) <= self.max_length:
                        sequences.append({
                            'id': current_id,
                            'sequence': seq_str,
                            'length': len(seq_str),
                            'quality': 'I' * len(seq_str)
                        })
        
        except Exception as e:
            print(f"Error processing FASTA: {e}")
        
        return sequences
    
    def _bulk_quality_control(self, sequences):
        """Apply bulk quality control measures"""
        if not sequences:
            return sequences
        
        # Remove duplicates
        unique_sequences = {}
        for seq in sequences:
            seq_key = seq['sequence']
            if seq_key not in unique_sequences:
                unique_sequences[seq_key] = seq
            else:
                # Keep the one with better quality
                if hasattr(seq, 'quality') and hasattr(unique_sequences[seq_key], 'quality'):
                    if self._calculate_avg_quality(seq['quality']) > self._calculate_avg_quality(unique_sequences[seq_key]['quality']):
                        unique_sequences[seq_key] = seq
        
        filtered_sequences = list(unique_sequences.values())
        
        # Remove chimeric sequences (simplified detection)
        non_chimeric = []
        for seq in filtered_sequences:
            if not self._is_likely_chimeric(seq['sequence']):
                non_chimeric.append(seq)
        
        return non_chimeric
    
    def _is_likely_chimeric(self, sequence):
        """Simple chimera detection"""
        # Check for abrupt GC content changes
        if len(sequence) < 100:
            return False
        
        mid_point = len(sequence) // 2
        first_half = sequence[:mid_point]
        second_half = sequence[mid_point:]
        
        gc1 = (first_half.count('G') + first_half.count('C')) / len(first_half)
        gc2 = (second_half.count('G') + second_half.count('C')) / len(second_half)
        
        # If GC content differs by more than 30%, might be chimeric
        return abs(gc1 - gc2) > 0.3
    
    def _generate_quality_stats(self, sequences):
        """Generate quality statistics for processed sequences"""
        if not sequences:
            return {}
        
        lengths = [seq['length'] for seq in sequences]
        
        stats = {
            'sequence_count': len(sequences),
            'length_distribution': {
                'min': min(lengths),
                'max': max(lengths),
                'mean': sum(lengths) / len(lengths),
                'median': sorted(lengths)[len(lengths)//2]
            }
        }
        
        # Quality scores if available
        if all('quality' in seq for seq in sequences):
            quality_scores = [self._calculate_avg_quality(seq['quality']) for seq in sequences]
            stats['quality_distribution'] = {
                'min': min(quality_scores),
                'max': max(quality_scores),
                'mean': sum(quality_scores) / len(quality_scores)
            }
        
        # Base composition
        all_bases = ''.join(seq['sequence'] for seq in sequences)
        base_counts = Counter(all_bases)
        total_bases = len(all_bases)
        
        stats['base_composition'] = {
            base: count / total_bases * 100 
            for base, count in base_counts.items()
        }
        
        return stats
    
    def cluster_otus(self, sequences, similarity_threshold=0.97):
        """Cluster sequences into OTUs (Operational Taxonomic Units)"""
        if len(sequences) < 2:
            return sequences, [0] * len(sequences)
        
        # Simple clustering based on sequence similarity
        clusters = []
        cluster_labels = []
        
        for i, seq in enumerate(sequences):
            assigned = False
            
            for cluster_id, cluster in enumerate(clusters):
                # Check similarity with cluster representative
                representative = sequences[cluster[0]]
                similarity = self._calculate_similarity(seq['sequence'], representative['sequence'])
                
                if similarity >= similarity_threshold:
                    cluster.append(i)
                    cluster_labels.append(cluster_id)
                    assigned = True
                    break
            
            if not assigned:
                clusters.append([i])
                cluster_labels.append(len(clusters) - 1)
        
        # Create OTU representatives
        otu_sequences = []
        for cluster_id, cluster in enumerate(clusters):
            # Use the longest sequence as representative
            representative_idx = max(cluster, key=lambda x: sequences[x]['length'])
            representative = sequences[representative_idx].copy()
            representative['id'] = f"OTU_{cluster_id:04d}"
            representative['cluster_size'] = len(cluster)
            representative['member_sequences'] = [sequences[idx]['id'] for idx in cluster]
            otu_sequences.append(representative)
        
        return otu_sequences, cluster_labels
    
    def _calculate_similarity(self, seq1, seq2):
        """Calculate sequence similarity (simplified)"""
        if len(seq1) != len(seq2):
            # Align to shorter sequence
            min_len = min(len(seq1), len(seq2))
            seq1 = seq1[:min_len]
            seq2 = seq2[:min_len]
        
        if len(seq1) == 0:
            return 0
        
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
        return matches / len(seq1)
    
    def extract_marker_genes(self, sequences, marker_type='18S'):
        """Extract specific marker gene sequences"""
        marker_patterns = {
            '18S': ['GTCAGAGGTGAAATTCTTGGATTT', 'TGATCCTTCTGCAGGTTCACCTAC'],
            'COI': ['GGTCAACAAATCATAAAGATATTGG', 'TAAACTTCAGGGTGACCAAAAAATCA'],
            'ITS': ['CTTGGTCATTTAGAGGAAGTAA', 'GCTGCGTTCTTCATCGATGC']
        }
        
        patterns = marker_patterns.get(marker_type, [])
        marker_sequences = []
        
        for seq in sequences:
            sequence = seq['sequence']
            
            # Check for marker patterns
            has_marker = any(pattern in sequence for pattern in patterns)
            
            if has_marker or marker_type.lower() in seq['id'].lower():
                marker_seq = seq.copy()
                marker_seq['marker_type'] = marker_type
                marker_seq['marker_confidence'] = 0.8 if has_marker else 0.6
                marker_sequences.append(marker_seq)
        
        return marker_sequences

class HighThroughputProcessor:
    """Specialized processor for high-throughput eDNA analysis"""
    
    def __init__(self):
        self.processor = eDNAProcessor()
        self.batch_size = 1000
        
    def process_large_dataset(self, file_paths, output_dir=None):
        """Process large eDNA datasets efficiently"""
        results = {
            'total_files': len(file_paths),
            'processed_sequences': 0,
            'otus_identified': 0,
            'processing_time': 0,
            'quality_summary': {}
        }
        
        all_sequences = []
        
        # Process files in batches
        for i in range(0, len(file_paths), self.batch_size):
            batch_files = file_paths[i:i + self.batch_size]
            
            batch_sequences, batch_stats = self.processor.process_bulk_sequences(batch_files)
            all_sequences.extend(batch_sequences)
            
            results['processed_sequences'] += batch_stats['filtered_sequences']
        
        # Cluster into OTUs
        otu_sequences, cluster_labels = self.processor.cluster_otus(all_sequences)
        results['otus_identified'] = len(otu_sequences)
        
        # Extract marker genes
        marker_18s = self.processor.extract_marker_genes(otu_sequences, '18S')
        marker_coi = self.processor.extract_marker_genes(otu_sequences, 'COI')
        
        results['marker_genes'] = {
            '18S_sequences': len(marker_18s),
            'COI_sequences': len(marker_coi)
        }
        
        # Generate quality summary
        results['quality_summary'] = self.processor._generate_quality_stats(otu_sequences)
        
        return {
            'otu_sequences': otu_sequences,
            'marker_18s': marker_18s,
            'marker_coi': marker_coi,
            'processing_results': results
        }
#!/usr/bin/env python3
"""
Database-Independent Classification Module
=========================================

Implements:
- Sequence similarity clustering without reference databases
- Feature extraction from sequences
- Non-database classification methods
- Self-organizing classification system
"""

import math
import random
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

class IndependentClassifier:
    """Database-independent sequence classifier"""
    
    def __init__(self):
        self.similarity_threshold = 0.85
        self.feature_cache = {}
        self.classification_tree = {}
        
    def extract_intrinsic_features(self, sequences):
        """Extract features without external database reference"""
        features = []
        
        for seq in sequences:
            seq_str = seq['sequence']
            seq_id = seq['id']
            
            if seq_id in self.feature_cache:
                features.append(self.feature_cache[seq_id])
                continue
            
            # Compositional features
            length = len(seq_str)
            gc_content = (seq_str.count('G') + seq_str.count('C')) / length if length > 0 else 0
            at_content = (seq_str.count('A') + seq_str.count('T')) / length if length > 0 else 0
            
            # Dinucleotide frequencies
            dinucs = ['AA', 'AT', 'AG', 'AC', 'TA', 'TT', 'TG', 'TC', 
                     'GA', 'GT', 'GG', 'GC', 'CA', 'CT', 'CG', 'CC']
            dinuc_freqs = {}
            
            for i in range(len(seq_str) - 1):
                dinuc = seq_str[i:i+2]
                dinuc_freqs[dinuc] = dinuc_freqs.get(dinuc, 0) + 1
            
            total_dinucs = sum(dinuc_freqs.values())
            dinuc_features = [dinuc_freqs.get(d, 0) / total_dinucs if total_dinucs > 0 else 0 
                            for d in dinucs]
            
            # Codon usage (if length is multiple of 3)
            codon_bias = self._calculate_codon_bias(seq_str)
            
            # Sequence complexity
            complexity = self._calculate_complexity(seq_str)
            
            # Repetitive elements
            repeat_content = self._calculate_repeat_content(seq_str)
            
            # Combine all features
            feature_vector = [
                length, gc_content, at_content, complexity, repeat_content, codon_bias
            ] + dinuc_features
            
            features.append(feature_vector)
            self.feature_cache[seq_id] = feature_vector
        
        return features
    
    def _calculate_codon_bias(self, sequence):
        """Calculate codon usage bias"""
        if len(sequence) < 3:
            return 0
        
        codons = {}
        for i in range(0, len(sequence) - 2, 3):
            codon = sequence[i:i+3]
            if 'N' not in codon:
                codons[codon] = codons.get(codon, 0) + 1
        
        if not codons:
            return 0
        
        # Calculate effective number of codons (simplified)
        total_codons = sum(codons.values())
        frequencies = [count / total_codons for count in codons.values()]
        
        # Shannon entropy as codon bias measure
        entropy = -sum(f * math.log(f) for f in frequencies if f > 0)
        max_entropy = math.log(len(codons)) if len(codons) > 1 else 1
        
        return entropy / max_entropy if max_entropy > 0 else 0
    
    def _calculate_complexity(self, sequence):
        """Calculate sequence complexity using linguistic complexity"""
        if len(sequence) < 4:
            return 0
        
        # Count unique 4-mers
        kmers = set()
        for i in range(len(sequence) - 3):
            kmers.add(sequence[i:i+4])
        
        # Complexity as ratio of unique k-mers to possible k-mers
        max_possible = min(4**4, len(sequence) - 3)
        return len(kmers) / max_possible if max_possible > 0 else 0
    
    def _calculate_repeat_content(self, sequence):
        """Calculate repetitive content"""
        if len(sequence) < 6:
            return 0
        
        repeat_bases = 0
        window_size = 6
        
        for i in range(len(sequence) - window_size):
            window = sequence[i:i+window_size]
            
            # Check for tandem repeats
            for j in range(i + window_size, len(sequence) - window_size):
                if sequence[j:j+window_size] == window:
                    repeat_bases += window_size
                    break
        
        return repeat_bases / len(sequence)
    
    def hierarchical_clustering(self, sequences):
        """Perform hierarchical clustering without reference database"""
        features = self.extract_intrinsic_features(sequences)
        
        if len(sequences) < 2:
            return [0] * len(sequences), self._build_single_cluster_tree(sequences)
        
        # Distance matrix calculation
        distance_matrix = self._calculate_distance_matrix(features)
        
        # Hierarchical clustering
        clusters = [[i] for i in range(len(sequences))]
        cluster_distances = {}
        
        # Build dendrogram
        merge_history = []
        
        while len(clusters) > 1:
            # Find closest clusters
            min_distance = float('inf')
            merge_i, merge_j = 0, 1
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    distance = self._cluster_distance(clusters[i], clusters[j], distance_matrix)
                    if distance < min_distance:
                        min_distance = distance
                        merge_i, merge_j = i, j
            
            # Merge clusters
            new_cluster = clusters[merge_i] + clusters[merge_j]
            merge_history.append({
                'clusters': [clusters[merge_i], clusters[merge_j]],
                'distance': min_distance,
                'size': len(new_cluster)
            })
            
            # Remove old clusters and add new one
            clusters = [c for i, c in enumerate(clusters) if i not in [merge_i, merge_j]]
            clusters.append(new_cluster)
        
        # Generate cluster labels at optimal cut
        optimal_clusters = self._find_optimal_clusters(merge_history, len(sequences))
        cluster_labels = self._assign_cluster_labels(optimal_clusters, len(sequences))
        
        # Build classification tree
        classification_tree = self._build_classification_tree(merge_history, sequences)
        
        return cluster_labels, classification_tree
    
    def _calculate_distance_matrix(self, features):
        """Calculate pairwise distance matrix"""
        n = len(features)
        matrix = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                # Euclidean distance
                distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(features[i], features[j])))
                matrix[i][j] = matrix[j][i] = distance
        
        return matrix
    
    def _cluster_distance(self, cluster1, cluster2, distance_matrix):
        """Calculate distance between clusters (average linkage)"""
        distances = []
        for i in cluster1:
            for j in cluster2:
                distances.append(distance_matrix[i][j])
        
        return sum(distances) / len(distances) if distances else 0
    
    def _find_optimal_clusters(self, merge_history, n_sequences):
        """Find optimal number of clusters using elbow method"""
        if not merge_history:
            return [[i] for i in range(n_sequences)]
        
        # Calculate within-cluster sum of squares for different k values
        wcss_values = []
        
        for k in range(1, min(10, n_sequences + 1)):
            # Reconstruct clusters at this level
            clusters = self._get_clusters_at_level(merge_history, k)
            wcss = self._calculate_wcss(clusters, merge_history)
            wcss_values.append(wcss)
        
        # Find elbow point (simplified)
        if len(wcss_values) < 3:
            optimal_k = 2
        else:
            # Find point with maximum curvature
            differences = [wcss_values[i] - wcss_values[i+1] for i in range(len(wcss_values)-1)]
            optimal_k = differences.index(max(differences)) + 2
        
        return self._get_clusters_at_level(merge_history, optimal_k)
    
    def _get_clusters_at_level(self, merge_history, k):
        """Get clusters at specific level k"""
        # Start with individual sequences
        clusters = [[i] for i in range(len(merge_history) + 1)]
        
        # Apply merges until we have k clusters
        merges_to_apply = len(merge_history) - k + 1
        
        for i in range(min(merges_to_apply, len(merge_history))):
            merge = merge_history[i]
            # Find and merge the clusters
            cluster1_idx = None
            cluster2_idx = None
            
            for j, cluster in enumerate(clusters):
                if set(cluster) == set(merge['clusters'][0]):
                    cluster1_idx = j
                elif set(cluster) == set(merge['clusters'][1]):
                    cluster2_idx = j
            
            if cluster1_idx is not None and cluster2_idx is not None:
                # Merge clusters
                new_cluster = clusters[cluster1_idx] + clusters[cluster2_idx]
                clusters = [c for j, c in enumerate(clusters) if j not in [cluster1_idx, cluster2_idx]]
                clusters.append(new_cluster)
        
        return clusters
    
    def _calculate_wcss(self, clusters, merge_history):
        """Calculate within-cluster sum of squares"""
        # Simplified WCSS calculation
        total_wcss = 0
        
        for cluster in clusters:
            if len(cluster) > 1:
                # Calculate cluster variance (simplified)
                cluster_variance = len(cluster) * 0.1  # Mock calculation
                total_wcss += cluster_variance
        
        return total_wcss
    
    def _assign_cluster_labels(self, clusters, n_sequences):
        """Assign cluster labels to sequences"""
        labels = [0] * n_sequences
        
        for cluster_id, cluster in enumerate(clusters):
            for seq_idx in cluster:
                labels[seq_idx] = cluster_id
        
        return labels
    
    def _build_classification_tree(self, merge_history, sequences):
        """Build hierarchical classification tree"""
        tree = {
            'type': 'hierarchical_classification',
            'levels': len(merge_history),
            'total_sequences': len(sequences),
            'merge_history': merge_history
        }
        
        return tree
    
    def _build_single_cluster_tree(self, sequences):
        """Build tree for single cluster"""
        return {
            'type': 'single_cluster',
            'levels': 1,
            'total_sequences': len(sequences),
            'merge_history': []
        }
    
    def classify_without_database(self, sequences):
        """Complete database-independent classification"""
        # Hierarchical clustering
        cluster_labels, classification_tree = self.hierarchical_clustering(sequences)
        
        # Feature-based classification
        features = self.extract_intrinsic_features(sequences)
        
        # Generate classifications
        classifications = []
        
        for i, seq in enumerate(sequences):
            cluster_id = cluster_labels[i]
            feature_vector = features[i]
            
            # Predict taxonomic level based on features
            taxonomic_prediction = self._predict_taxonomic_level(feature_vector)
            
            # Calculate confidence based on cluster cohesion
            confidence = self._calculate_classification_confidence(i, cluster_labels, features)
            
            # Determine if novel
            novelty_score = self._assess_novelty(i, cluster_labels, features)
            
            classification = {
                'sequence_id': seq['id'],
                'cluster_id': cluster_id,
                'predicted_taxonomic_level': taxonomic_prediction,
                'confidence': confidence,
                'novelty_score': novelty_score,
                'is_novel': novelty_score > 0.7,
                'classification_method': 'database_independent',
                'feature_signature': feature_vector[:6]  # First 6 features as signature
            }
            
            classifications.append(classification)
        
        return {
            'classifications': classifications,
            'classification_tree': classification_tree,
            'cluster_summary': self._generate_cluster_summary(cluster_labels, classifications),
            'novel_taxa_count': sum(1 for c in classifications if c['is_novel'])
        }
    
    def _predict_taxonomic_level(self, feature_vector):
        """Predict taxonomic level based on intrinsic features"""
        length, gc_content, at_content, complexity, repeat_content, codon_bias = feature_vector[:6]
        
        # Rule-based taxonomic prediction
        if complexity > 0.8 and codon_bias > 0.7:
            return 'eukaryotic'
        elif gc_content > 0.6:
            return 'bacterial'
        elif repeat_content > 0.3:
            return 'viral'
        elif length > 1000:
            return 'genomic'
        else:
            return 'environmental'
    
    def _calculate_classification_confidence(self, seq_idx, cluster_labels, features):
        """Calculate confidence based on cluster cohesion"""
        seq_cluster = cluster_labels[seq_idx]
        seq_features = features[seq_idx]
        
        # Find other sequences in same cluster
        cluster_members = [i for i, label in enumerate(cluster_labels) if label == seq_cluster and i != seq_idx]
        
        if not cluster_members:
            return 0.5  # Singleton cluster
        
        # Calculate average distance to cluster members
        distances = []
        for member_idx in cluster_members:
            member_features = features[member_idx]
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(seq_features, member_features)))
            distances.append(distance)
        
        avg_distance = sum(distances) / len(distances)
        
        # Convert distance to confidence (inverse relationship)
        confidence = max(0.1, 1.0 - (avg_distance / 10.0))
        return min(0.95, confidence)
    
    def _assess_novelty(self, seq_idx, cluster_labels, features):
        """Assess novelty of sequence"""
        seq_cluster = cluster_labels[seq_idx]
        seq_features = features[seq_idx]
        
        # Count cluster size
        cluster_size = sum(1 for label in cluster_labels if label == seq_cluster)
        
        # Singleton clusters are highly novel
        if cluster_size == 1:
            return 0.9
        
        # Small clusters are moderately novel
        if cluster_size <= 3:
            return 0.7
        
        # Calculate distance to cluster centroid
        cluster_members = [i for i, label in enumerate(cluster_labels) if label == seq_cluster]
        cluster_features = [features[i] for i in cluster_members]
        
        # Calculate centroid
        if NUMPY_AVAILABLE:
            centroid = np.mean(cluster_features, axis=0)
            distance_to_centroid = np.sqrt(np.sum((seq_features - centroid) ** 2))
        else:
            centroid = [sum(f[j] for f in cluster_features) / len(cluster_features) 
                       for j in range(len(seq_features))]
            distance_to_centroid = math.sqrt(sum((a - b) ** 2 for a, b in zip(seq_features, centroid)))
        
        # Normalize distance to novelty score
        novelty_score = min(1.0, distance_to_centroid / 5.0)
        
        return novelty_score
    
    def _generate_cluster_summary(self, cluster_labels, classifications):
        """Generate summary of clustering results"""
        cluster_counts = Counter(cluster_labels)
        
        summary = {
            'total_clusters': len(cluster_counts),
            'largest_cluster_size': max(cluster_counts.values()),
            'smallest_cluster_size': min(cluster_counts.values()),
            'singleton_clusters': sum(1 for count in cluster_counts.values() if count == 1),
            'cluster_distribution': dict(cluster_counts)
        }
        
        # Taxonomic distribution
        taxonomic_counts = Counter(c['predicted_taxonomic_level'] for c in classifications)
        summary['taxonomic_distribution'] = dict(taxonomic_counts)
        
        return summary

class SelfOrganizingClassifier:
    """Self-organizing classification system"""
    
    def __init__(self):
        self.independent_classifier = IndependentClassifier()
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.8
        
    def adaptive_classification(self, sequences):
        """Perform adaptive classification that improves over time"""
        # Initial classification
        initial_results = self.independent_classifier.classify_without_database(sequences)
        
        # Adaptive refinement
        refined_results = self._refine_classification(sequences, initial_results)
        
        # Self-validation
        validation_results = self._self_validate(sequences, refined_results)
        
        return {
            'initial_classification': initial_results,
            'refined_classification': refined_results,
            'validation_results': validation_results,
            'final_classifications': refined_results['classifications']
        }
    
    def _refine_classification(self, sequences, initial_results):
        """Refine classification based on internal consistency"""
        classifications = initial_results['classifications'].copy()
        
        # Identify low-confidence classifications
        low_confidence = [c for c in classifications if c['confidence'] < 0.6]
        
        # Attempt to reclassify low-confidence sequences
        for classification in low_confidence:
            seq_idx = next(i for i, seq in enumerate(sequences) if seq['id'] == classification['sequence_id'])
            
            # Find most similar high-confidence sequence
            best_match = self._find_best_match(seq_idx, sequences, classifications)
            
            if best_match and best_match['confidence'] > 0.8:
                # Update classification based on best match
                classification['cluster_id'] = best_match['cluster_id']
                classification['predicted_taxonomic_level'] = best_match['predicted_taxonomic_level']
                classification['confidence'] = min(0.8, best_match['confidence'] * 0.9)
                classification['classification_method'] = 'adaptive_refinement'
        
        return {
            'classifications': classifications,
            'refinement_applied': len(low_confidence),
            'classification_tree': initial_results['classification_tree']
        }
    
    def _find_best_match(self, seq_idx, sequences, classifications):
        """Find best matching sequence for reclassification"""
        target_seq = sequences[seq_idx]
        target_features = self.independent_classifier.extract_intrinsic_features([target_seq])[0]
        
        best_match = None
        best_similarity = 0
        
        for i, seq in enumerate(sequences):
            if i == seq_idx:
                continue
            
            classification = classifications[i]
            if classification['confidence'] < 0.8:
                continue
            
            seq_features = self.independent_classifier.extract_intrinsic_features([seq])[0]
            
            # Calculate similarity
            similarity = self._calculate_feature_similarity(target_features, seq_features)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = classification
        
        return best_match if best_similarity > 0.7 else None
    
    def _calculate_feature_similarity(self, features1, features2):
        """Calculate similarity between feature vectors"""
        if len(features1) != len(features2):
            return 0
        
        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(features1, features2))
        magnitude1 = math.sqrt(sum(a * a for a in features1))
        magnitude2 = math.sqrt(sum(b * b for b in features2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _self_validate(self, sequences, refined_results):
        """Perform self-validation of classification results"""
        classifications = refined_results['classifications']
        
        # Cross-validation within clusters
        cluster_consistency = self._validate_cluster_consistency(sequences, classifications)
        
        # Taxonomic consistency
        taxonomic_consistency = self._validate_taxonomic_consistency(classifications)
        
        # Overall quality metrics
        avg_confidence = sum(c['confidence'] for c in classifications) / len(classifications)
        novel_taxa_ratio = sum(1 for c in classifications if c['is_novel']) / len(classifications)
        
        return {
            'cluster_consistency': cluster_consistency,
            'taxonomic_consistency': taxonomic_consistency,
            'average_confidence': avg_confidence,
            'novel_taxa_ratio': novel_taxa_ratio,
            'validation_score': (cluster_consistency + taxonomic_consistency + avg_confidence) / 3
        }
    
    def _validate_cluster_consistency(self, sequences, classifications):
        """Validate consistency within clusters"""
        cluster_groups = defaultdict(list)
        
        for classification in classifications:
            cluster_groups[classification['cluster_id']].append(classification)
        
        consistency_scores = []
        
        for cluster_id, cluster_classifications in cluster_groups.items():
            if len(cluster_classifications) < 2:
                consistency_scores.append(1.0)
                continue
            
            # Check taxonomic consistency within cluster
            taxonomic_levels = [c['predicted_taxonomic_level'] for c in cluster_classifications]
            most_common = max(set(taxonomic_levels), key=taxonomic_levels.count)
            consistency = taxonomic_levels.count(most_common) / len(taxonomic_levels)
            consistency_scores.append(consistency)
        
        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0
    
    def _validate_taxonomic_consistency(self, classifications):
        """Validate taxonomic assignment consistency"""
        taxonomic_groups = defaultdict(list)
        
        for classification in classifications:
            taxonomic_groups[classification['predicted_taxonomic_level']].append(classification)
        
        # Check if similar sequences have similar taxonomic assignments
        consistency_score = 0.8  # Base score
        
        # Penalize if novel taxa are too frequent (might indicate over-splitting)
        novel_ratio = sum(1 for c in classifications if c['is_novel']) / len(classifications)
        if novel_ratio > 0.5:
            consistency_score *= 0.8
        
        return consistency_score
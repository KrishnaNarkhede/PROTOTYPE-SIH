#!/usr/bin/env python3
"""
Deep-Sea Specific Optimization Module
====================================

Implements:
- Marine organism detection patterns
- Deep-sea habitat classification
- Environmental context analysis
- Specialized eukaryotic identification
"""

import math
import random
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class DeepSeaOptimizer:
    """Specialized optimizer for deep-sea eDNA analysis"""
    
    def __init__(self):
        self.marine_signatures = self._load_marine_signatures()
        self.depth_indicators = self._load_depth_indicators()
        self.eukaryotic_markers = self._load_eukaryotic_markers()
        
    def _load_marine_signatures(self):
        """Load marine-specific sequence signatures"""
        return {
            'deep_sea_patterns': {
                'hydrothermal_vent': ['TAAGGCCT', 'CCTTAAGC', 'GGCCTTAA'],
                'abyssal_plain': ['ATCCGGTA', 'TACCGGAT', 'CCGGATAC'],
                'seamount': ['GCCTTAGG', 'CCTTAGGC', 'TTAGGCCT'],
                'cold_seep': ['TTGGCCAA', 'GGCCAATT', 'CCAATTGG']
            },
            'marine_taxa_indicators': {
                'cnidaria': ['AATGCGTT', 'CGTTAATG', 'TTAATGCG'],
                'mollusca': ['CCGGTAAC', 'GGTAACCG', 'TAACCGGT'],
                'arthropoda': ['TTCCGGAA', 'CCGGAATT', 'GGAATTCC'],
                'echinodermata': ['GGTTCCAA', 'TTCCAAGG', 'CCAAGGTT'],
                'chordata': ['AACCGGTT', 'CCGGTTAA', 'GGTTAACC']
            },
            'pressure_adaptation': {
                'high_pressure': ['CCGGCCGG', 'GGCCGGCC'],
                'osmotic_regulation': ['AATTGGCC', 'TTGGCCAA'],
                'cold_adaptation': ['GGCCAATT', 'CCAATTGG']
            }
        }
    
    def _load_depth_indicators(self):
        """Load depth-specific indicators"""
        return {
            'surface': {'gc_range': (0.35, 0.45), 'complexity_min': 0.6},
            'mesopelagic': {'gc_range': (0.40, 0.50), 'complexity_min': 0.65},
            'bathypelagic': {'gc_range': (0.45, 0.55), 'complexity_min': 0.70},
            'abyssopelagic': {'gc_range': (0.50, 0.60), 'complexity_min': 0.75},
            'hadalpelagic': {'gc_range': (0.55, 0.65), 'complexity_min': 0.80}
        }
    
    def _load_eukaryotic_markers(self):
        """Load eukaryotic-specific markers"""
        return {
            '18S_rRNA': {
                'conserved_regions': [
                    'GTCAGAGGTGAAATTCTTGGATTT',
                    'TGATCCTTCTGCAGGTTCACCTAC',
                    'CCTGCGGAAGGATCATTAACAGAA'
                ],
                'variable_regions': ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9']
            },
            'COI': {
                'conserved_regions': [
                    'GGTCAACAAATCATAAAGATATTGG',
                    'TAAACTTCAGGGTGACCAAAAAATCA',
                    'GGWACWGGWTGAACWGTWTAYCCYCC'
                ],
                'barcode_region': 'COI_5P'
            },
            'ITS': {
                'conserved_regions': [
                    'CTTGGTCATTTAGAGGAAGTAA',
                    'GCTGCGTTCTTCATCGATGC',
                    'TCCTCCGCTTATTGATATGC'
                ]
            }
        }
    
    def analyze_marine_context(self, sequences):
        """Analyze sequences for marine environmental context"""
        marine_analysis = {
            'habitat_predictions': [],
            'depth_estimates': [],
            'marine_taxa_detected': [],
            'environmental_indicators': {}
        }
        
        for seq in sequences:
            seq_str = seq['sequence']
            seq_id = seq['id']
            
            # Habitat prediction
            habitat = self._predict_habitat(seq_str)
            
            # Depth estimation
            depth_zone = self._estimate_depth_zone(seq_str)
            
            # Marine taxa detection
            taxa_prediction = self._detect_marine_taxa(seq_str)
            
            # Environmental stress indicators
            stress_indicators = self._detect_stress_indicators(seq_str)
            
            marine_analysis['habitat_predictions'].append({
                'sequence_id': seq_id,
                'predicted_habitat': habitat['habitat'],
                'confidence': habitat['confidence'],
                'supporting_patterns': habitat['patterns']
            })
            
            marine_analysis['depth_estimates'].append({
                'sequence_id': seq_id,
                'depth_zone': depth_zone['zone'],
                'depth_confidence': depth_zone['confidence'],
                'depth_indicators': depth_zone['indicators']
            })
            
            marine_analysis['marine_taxa_detected'].append({
                'sequence_id': seq_id,
                'predicted_taxa': taxa_prediction['taxa'],
                'taxa_confidence': taxa_prediction['confidence'],
                'taxonomic_markers': taxa_prediction['markers']
            })
        
        # Environmental summary
        marine_analysis['environmental_indicators'] = self._summarize_environmental_context(
            marine_analysis['habitat_predictions'],
            marine_analysis['depth_estimates']
        )
        
        return marine_analysis
    
    def _predict_habitat(self, sequence):
        """Predict marine habitat from sequence characteristics"""
        habitat_scores = {}
        detected_patterns = []
        
        # Check for habitat-specific patterns
        for habitat, patterns in self.marine_signatures['deep_sea_patterns'].items():
            score = 0
            habitat_patterns = []
            
            for pattern in patterns:
                if pattern in sequence:
                    score += 1
                    habitat_patterns.append(pattern)
            
            if score > 0:
                habitat_scores[habitat] = score / len(patterns)
                detected_patterns.extend(habitat_patterns)
        
        # GC content analysis for habitat prediction
        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
        
        # Adjust scores based on GC content
        if gc_content > 0.55:  # High GC typical of deep-sea organisms
            habitat_scores['abyssal_plain'] = habitat_scores.get('abyssal_plain', 0) + 0.3
            habitat_scores['hydrothermal_vent'] = habitat_scores.get('hydrothermal_vent', 0) + 0.2
        
        # Determine best habitat prediction
        if habitat_scores:
            best_habitat = max(habitat_scores, key=habitat_scores.get)
            confidence = habitat_scores[best_habitat]
        else:
            best_habitat = 'unknown_marine'
            confidence = 0.1
        
        return {
            'habitat': best_habitat,
            'confidence': min(0.95, confidence),
            'patterns': detected_patterns,
            'all_scores': habitat_scores
        }
    
    def _estimate_depth_zone(self, sequence):
        """Estimate depth zone based on sequence characteristics"""
        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
        complexity = self._calculate_sequence_complexity(sequence)
        
        depth_scores = {}
        indicators = []
        
        # Check each depth zone
        for zone, criteria in self.depth_indicators.items():
            score = 0
            zone_indicators = []
            
            # GC content check
            gc_min, gc_max = criteria['gc_range']
            if gc_min <= gc_content <= gc_max:
                score += 0.5
                zone_indicators.append(f'gc_content_{gc_content:.3f}')
            
            # Complexity check
            if complexity >= criteria['complexity_min']:
                score += 0.3
                zone_indicators.append(f'complexity_{complexity:.3f}')
            
            # Pressure adaptation patterns
            pressure_patterns = self.marine_signatures['pressure_adaptation']
            for adaptation_type, patterns in pressure_patterns.items():
                for pattern in patterns:
                    if pattern in sequence:
                        score += 0.2
                        zone_indicators.append(f'{adaptation_type}_pattern')
                        break
            
            if score > 0:
                depth_scores[zone] = score
                indicators.extend(zone_indicators)
        
        # Determine best depth prediction
        if depth_scores:
            best_zone = max(depth_scores, key=depth_scores.get)
            confidence = min(0.95, depth_scores[best_zone])
        else:
            best_zone = 'unknown_depth'
            confidence = 0.1
        
        return {
            'zone': best_zone,
            'confidence': confidence,
            'indicators': indicators,
            'all_scores': depth_scores
        }
    
    def _detect_marine_taxa(self, sequence):
        """Detect marine taxa from sequence patterns"""
        taxa_scores = {}
        detected_markers = []
        
        # Check for taxa-specific indicators
        for taxa, patterns in self.marine_signatures['marine_taxa_indicators'].items():
            score = 0
            taxa_markers = []
            
            for pattern in patterns:
                if pattern in sequence:
                    score += 1
                    taxa_markers.append(pattern)
            
            if score > 0:
                taxa_scores[taxa] = score / len(patterns)
                detected_markers.extend(taxa_markers)
        
        # Additional analysis based on sequence characteristics
        length = len(sequence)
        gc_content = (sequence.count('G') + sequence.count('C')) / length
        
        # Eukaryotic vs prokaryotic indicators
        if gc_content < 0.45 and length > 200:
            taxa_scores['eukaryotic'] = taxa_scores.get('eukaryotic', 0) + 0.3
        elif gc_content > 0.6 and length < 150:
            taxa_scores['prokaryotic'] = taxa_scores.get('prokaryotic', 0) + 0.3
        
        # Determine best taxa prediction
        if taxa_scores:
            best_taxa = max(taxa_scores, key=taxa_scores.get)
            confidence = min(0.95, taxa_scores[best_taxa])
        else:
            best_taxa = 'unknown_taxa'
            confidence = 0.1
        
        return {
            'taxa': best_taxa,
            'confidence': confidence,
            'markers': detected_markers,
            'all_scores': taxa_scores
        }
    
    def _detect_stress_indicators(self, sequence):
        """Detect environmental stress adaptation indicators"""
        stress_indicators = {
            'pressure_adaptation': 0,
            'temperature_adaptation': 0,
            'salinity_adaptation': 0,
            'nutrient_limitation': 0
        }
        
        # Pressure adaptation (high GC content, specific patterns)
        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
        if gc_content > 0.55:
            stress_indicators['pressure_adaptation'] += 0.4
        
        # Temperature adaptation (codon usage bias)
        codon_bias = self._calculate_codon_bias(sequence)
        if codon_bias > 0.7:
            stress_indicators['temperature_adaptation'] += 0.3
        
        # Check for specific adaptation patterns
        adaptation_patterns = {
            'pressure': ['CCGGCCGG', 'GGCCGGCC'],
            'cold': ['GGCCAATT', 'CCAATTGG'],
            'osmotic': ['AATTGGCC', 'TTGGCCAA']
        }
        
        for adaptation_type, patterns in adaptation_patterns.items():
            for pattern in patterns:
                if pattern in sequence:
                    if adaptation_type == 'pressure':
                        stress_indicators['pressure_adaptation'] += 0.2
                    elif adaptation_type == 'cold':
                        stress_indicators['temperature_adaptation'] += 0.2
                    elif adaptation_type == 'osmotic':
                        stress_indicators['salinity_adaptation'] += 0.2
        
        return stress_indicators
    
    def _calculate_sequence_complexity(self, sequence):
        """Calculate sequence complexity"""
        if len(sequence) < 4:
            return 0
        
        # Count unique 4-mers
        kmers = set()
        for i in range(len(sequence) - 3):
            kmers.add(sequence[i:i+4])
        
        max_possible = min(4**4, len(sequence) - 3)
        return len(kmers) / max_possible if max_possible > 0 else 0
    
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
        
        total_codons = sum(codons.values())
        frequencies = [count / total_codons for count in codons.values()]
        
        # Shannon entropy as codon bias measure
        entropy = -sum(f * math.log(f) for f in frequencies if f > 0)
        max_entropy = math.log(len(codons)) if len(codons) > 1 else 1
        
        return entropy / max_entropy if max_entropy > 0 else 0
    
    def _summarize_environmental_context(self, habitat_predictions, depth_estimates):
        """Summarize environmental context from all sequences"""
        # Habitat distribution
        habitats = [h['predicted_habitat'] for h in habitat_predictions]
        habitat_distribution = Counter(habitats)
        
        # Depth distribution
        depths = [d['depth_zone'] for d in depth_estimates]
        depth_distribution = Counter(depths)
        
        # Confidence statistics
        habitat_confidences = [h['confidence'] for h in habitat_predictions]
        depth_confidences = [d['depth_confidence'] for d in depth_estimates]
        
        return {
            'habitat_distribution': dict(habitat_distribution),
            'depth_distribution': dict(depth_distribution),
            'average_habitat_confidence': sum(habitat_confidences) / len(habitat_confidences),
            'average_depth_confidence': sum(depth_confidences) / len(depth_confidences),
            'dominant_habitat': max(habitat_distribution, key=habitat_distribution.get) if habitat_distribution else 'unknown',
            'dominant_depth': max(depth_distribution, key=depth_distribution.get) if depth_distribution else 'unknown'
        }
    
    def identify_eukaryotic_sequences(self, sequences):
        """Identify eukaryotic sequences using marker genes"""
        eukaryotic_sequences = []
        
        for seq in sequences:
            seq_str = seq['sequence']
            seq_id = seq['id']
            
            eukaryotic_score = 0
            detected_markers = []
            marker_details = {}
            
            # Check for 18S rRNA markers
            for region in self.eukaryotic_markers['18S_rRNA']['conserved_regions']:
                if region in seq_str:
                    eukaryotic_score += 0.4
                    detected_markers.append('18S_rRNA')
                    marker_details['18S_rRNA'] = True
                    break
            
            # Check for COI markers
            for region in self.eukaryotic_markers['COI']['conserved_regions']:
                if region in seq_str:
                    eukaryotic_score += 0.4
                    detected_markers.append('COI')
                    marker_details['COI'] = True
                    break
            
            # Check for ITS markers
            for region in self.eukaryotic_markers['ITS']['conserved_regions']:
                if region in seq_str:
                    eukaryotic_score += 0.3
                    detected_markers.append('ITS')
                    marker_details['ITS'] = True
                    break
            
            # Additional eukaryotic indicators
            gc_content = (seq_str.count('G') + seq_str.count('C')) / len(seq_str)
            length = len(seq_str)
            
            # Eukaryotic sequences often have moderate GC content and longer lengths
            if 0.35 <= gc_content <= 0.55 and length > 150:
                eukaryotic_score += 0.2
            
            # Intron indicators (simplified)
            if 'GT' in seq_str and 'AG' in seq_str:  # Splice sites
                eukaryotic_score += 0.1
            
            # If score is high enough, classify as eukaryotic
            if eukaryotic_score >= 0.3:
                eukaryotic_sequences.append({
                    'sequence_id': seq_id,
                    'eukaryotic_score': min(0.95, eukaryotic_score),
                    'detected_markers': detected_markers,
                    'marker_details': marker_details,
                    'sequence_length': length,
                    'gc_content': gc_content
                })
        
        return eukaryotic_sequences
    
    def optimize_for_deep_sea(self, sequences):
        """Complete deep-sea optimization analysis"""
        # Marine context analysis
        marine_context = self.analyze_marine_context(sequences)
        
        # Eukaryotic identification
        eukaryotic_seqs = self.identify_eukaryotic_sequences(sequences)
        
        # Deep-sea specific classification
        deep_sea_classifications = []
        
        for i, seq in enumerate(sequences):
            seq_id = seq['id']
            
            # Get marine context for this sequence
            habitat_pred = next((h for h in marine_context['habitat_predictions'] if h['sequence_id'] == seq_id), None)
            depth_pred = next((d for d in marine_context['depth_estimates'] if d['sequence_id'] == seq_id), None)
            taxa_pred = next((t for t in marine_context['marine_taxa_detected'] if t['sequence_id'] == seq_id), None)
            
            # Check if eukaryotic
            euk_info = next((e for e in eukaryotic_seqs if e['sequence_id'] == seq_id), None)
            
            # Combine information
            classification = {
                'sequence_id': seq_id,
                'is_marine': habitat_pred['confidence'] > 0.3 if habitat_pred else False,
                'is_deep_sea': (depth_pred['depth_zone'] in ['bathypelagic', 'abyssopelagic', 'hadalpelagic'] 
                               if depth_pred else False),
                'is_eukaryotic': euk_info is not None,
                'habitat': habitat_pred['predicted_habitat'] if habitat_pred else 'unknown',
                'depth_zone': depth_pred['depth_zone'] if depth_pred else 'unknown',
                'predicted_taxa': taxa_pred['predicted_taxa'] if taxa_pred else 'unknown',
                'marine_confidence': habitat_pred['confidence'] if habitat_pred else 0,
                'depth_confidence': depth_pred['depth_confidence'] if depth_pred else 0,
                'eukaryotic_score': euk_info['eukaryotic_score'] if euk_info else 0,
                'optimization_applied': True
            }
            
            # Calculate overall deep-sea relevance score
            relevance_score = 0
            if classification['is_marine']:
                relevance_score += 0.3
            if classification['is_deep_sea']:
                relevance_score += 0.4
            if classification['is_eukaryotic']:
                relevance_score += 0.3
            
            classification['deep_sea_relevance'] = min(0.95, relevance_score)
            
            deep_sea_classifications.append(classification)
        
        return {
            'deep_sea_classifications': deep_sea_classifications,
            'marine_context_analysis': marine_context,
            'eukaryotic_sequences': eukaryotic_seqs,
            'optimization_summary': {
                'total_sequences': len(sequences),
                'marine_sequences': sum(1 for c in deep_sea_classifications if c['is_marine']),
                'deep_sea_sequences': sum(1 for c in deep_sea_classifications if c['is_deep_sea']),
                'eukaryotic_sequences': len(eukaryotic_seqs),
                'high_relevance_sequences': sum(1 for c in deep_sea_classifications if c['deep_sea_relevance'] > 0.7)
            }
        }
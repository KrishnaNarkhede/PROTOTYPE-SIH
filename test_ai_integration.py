#!/usr/bin/env python3
"""
Test AI Integration for Deep-Sea eDNA Analysis
==============================================

Tests all AI modules working together for comprehensive analysis
"""

import sys
import os
from biomapper_prototype import BioMapperPrototype

def create_test_sequences():
    """Create test sequences for deep-sea eDNA analysis"""
    return [
        {
            'id': 'DeepSea_Novel_001',
            'sequence': 'ATCGATCGATCGATCGAAGGCCTTAGGCCTTAAGCCCGGCCGGCCAATTGGCCTTGGCCAATTGGCC',
            'length': 64
        },
        {
            'id': 'Marine_Cnidaria_002', 
            'sequence': 'AATGCGTTCGTTAATGCGTTAATGCGGGCCAATTGGCCTTGGCCAATTGGCCTTGGCCAATTGGCC',
            'length': 64
        },
        {
            'id': 'Eukaryotic_18S_003',
            'sequence': 'GTCAGAGGTGAAATTCTTGGATTTCCGGCCGGCCAATTGGCCTTGGCCAATTGGCCTTGGCCAATT',
            'length': 64
        },
        {
            'id': 'Unknown_Deep_004',
            'sequence': 'TTGGCCAATTGGCCTTGGCCAATTGGCCTTGGCCAATTGGCCTTGGCCAATTGGCCTTGGCCAATT',
            'length': 64
        },
        {
            'id': 'Environmental_OTU_005',
            'sequence': 'CCGGATACCGGATACCGGATACCGGATACCGGATACCGGATACCGGATACCGGATACCGGATAC',
            'length': 62
        }
    ]

def test_ai_modules():
    """Test all AI modules integration"""
    print("🧪 Testing AI Integration for Deep-Sea eDNA Analysis")
    print("=" * 60)
    
    # Initialize BioMapper with AI capabilities
    biomapper = BioMapperPrototype()
    
    # Create test sequences
    test_sequences = create_test_sequences()
    
    print(f"📊 Testing with {len(test_sequences)} sequences...")
    
    # Run comprehensive analysis
    try:
        results = biomapper.analyze_fasta_sequences(test_sequences)
        
        if results['status'] == 'success':
            print("✅ Analysis completed successfully!")
            
            # Test AI-specific results
            if 'ai_classification' in results:
                ai_results = results['ai_classification']
                print(f"🤖 AI Classification: {ai_results['novel_taxa_count']} novel taxa detected")
                print(f"📈 Clustering Quality: {ai_results['clustering_quality']:.3f}")
            
            if 'deepsea_analysis' in results:
                deepsea = results['deepsea_analysis']['optimization_summary']
                print(f"🌊 Marine Sequences: {deepsea['marine_sequences']}")
                print(f"🏔️ Deep-Sea Sequences: {deepsea['deep_sea_sequences']}")
                print(f"🦠 Eukaryotic Sequences: {deepsea['eukaryotic_sequences']}")
            
            if 'independent_classification' in results:
                independent = results['independent_classification']
                print(f"🔬 Independent Classification: {independent['novel_taxa_count']} novel taxa")
            
            if 'edna_processing' in results:
                edna = results['edna_processing']
                print(f"🧬 eDNA Processing: {edna['otu_count']} OTUs identified")
                print(f"📍 Marker Genes: 18S={edna['marker_genes']['18S_count']}, COI={edna['marker_genes']['COI_count']}")
            
            print("\n📋 Summary Report:")
            summary = biomapper.generate_summary_report(results)
            print(summary)
            
            return True
            
        else:
            print(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {str(e)}")
        return False

def test_individual_modules():
    """Test individual AI modules"""
    print("\n🔧 Testing Individual AI Modules")
    print("-" * 40)
    
    try:
        from ai_classifier import AIClassifier, DeepSeaAI
        from edna_processor import eDNAProcessor
        from independent_classifier import IndependentClassifier
        from deepsea_optimizer import DeepSeaOptimizer
        
        test_sequences = create_test_sequences()
        
        # Test AI Classifier
        print("Testing AI Classifier...")
        ai_classifier = AIClassifier()
        ai_results = ai_classifier.predict_novel_taxa(test_sequences)
        print(f"✅ AI Classifier: {len(ai_results['predictions'])} predictions made")
        
        # Test Deep-Sea AI
        print("Testing Deep-Sea AI...")
        deepsea_ai = DeepSeaAI()
        marine_results = deepsea_ai.classify_marine_sequences(test_sequences)
        print(f"✅ Deep-Sea AI: {len(marine_results['predictions'])} marine classifications")
        
        # Test eDNA Processor
        print("Testing eDNA Processor...")
        edna_processor = eDNAProcessor()
        otu_sequences, labels = edna_processor.cluster_otus(test_sequences)
        print(f"✅ eDNA Processor: {len(otu_sequences)} OTUs clustered")
        
        # Test Independent Classifier
        print("Testing Independent Classifier...")
        independent = IndependentClassifier()
        independent_results = independent.classify_without_database(test_sequences)
        print(f"✅ Independent Classifier: {len(independent_results['classifications'])} classifications")
        
        # Test Deep-Sea Optimizer
        print("Testing Deep-Sea Optimizer...")
        optimizer = DeepSeaOptimizer()
        optimization_results = optimizer.optimize_for_deep_sea(test_sequences)
        print(f"✅ Deep-Sea Optimizer: {len(optimization_results['deep_sea_classifications'])} optimized classifications")
        
        return True
        
    except ImportError as e:
        print(f"⚠️ Some AI modules not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Module test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 BioMapper AI Integration Test Suite")
    print("=" * 50)
    
    # Test integrated system
    integration_success = test_ai_modules()
    
    # Test individual modules
    modules_success = test_individual_modules()
    
    print("\n" + "=" * 50)
    if integration_success and modules_success:
        print("🎉 All tests passed! AI integration successful.")
        print("\n✨ Features now available:")
        print("  • AI-driven classification without database dependency")
        print("  • Unsupervised clustering for novel taxa discovery")
        print("  • Deep-sea specific optimization")
        print("  • High-throughput eDNA processing")
        print("  • Database-independent analysis")
        print("  • Marine organism detection")
        print("  • Eukaryotic marker gene identification")
        return 0
    else:
        print("⚠️ Some tests failed. Check AI module installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
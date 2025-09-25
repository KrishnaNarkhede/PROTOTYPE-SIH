# BioMapper Prototype - AI-Enhanced Deep-Sea eDNA Analysis System

## 🧬 Overview

This is an AI-enhanced standalone prototype of the BioMapper biodiversity analysis platform, specifically optimized for deep-sea environmental DNA (eDNA) analysis. It features advanced machine learning capabilities for novel taxa discovery without database dependency.

## 🚀 NEW AI CAPABILITIES

### ✨ AI-Driven Classification
- **Unsupervised Learning**: K-means and DBSCAN clustering for novel species discovery
- **Deep Learning Models**: Neural networks for sequence classification
- **Database Independence**: Classification without relying on reference databases
- **Novel Taxa Detection**: AI-powered identification of potentially new species

### 🌊 Deep-Sea Optimization
- **Marine Organism Detection**: Specialized patterns for deep-sea life
- **Habitat Classification**: Depth zone prediction (surface to hadal)
- **Eukaryotic Identification**: 18S rRNA and COI marker gene detection
- **Environmental Context**: Pressure, temperature, and salinity adaptation analysis

### 🧬 High-Throughput eDNA Processing
- **FASTQ File Support**: Quality filtering and denoising
- **OTU Clustering**: Operational Taxonomic Unit identification
- **Bulk Processing**: Efficient handling of large datasets
- **Marker Gene Extraction**: Automated 18S, COI, and ITS detection

## ✨ Features

### 🤖 AI-Enhanced Analysis:
- **Machine Learning Classification** - Unsupervised clustering and deep learning
- **Novel Taxa Discovery** - AI-powered detection without database dependency
- **Database-Independent Analysis** - Self-organizing classification system
- **Deep-Sea Optimization** - Marine-specific organism detection
- **High-Throughput Processing** - Bulk eDNA sequence analysis
- **Eukaryotic Identification** - Automated marker gene detection

### 🧬 Core Analysis Functions:
- **DNA Sequence Analysis** - Parse FASTA/FASTQ files with quality control
- **AI Species Classification** - Machine learning-based identification
- **Biodiversity Metrics** - Calculate diversity indices and richness
- **Quality Assessment** - eDNA-specific quality analysis
- **Conservation Assessment** - Evaluate conservation status and generate alerts
- **Phylogenetic Analysis** - Build phylogenetic trees
- **Microbiome Analysis** - Analyze microbial communities
- **Quantum Computing Simulation** - Simulate quantum-enhanced analysis
- **Protein Structure Prediction** - Predict protein structures
- **Comprehensive Reporting** - Generate detailed analysis reports

### 🌊 Deep-Sea Specific Features:
- **Habitat Prediction** - Hydrothermal vents, abyssal plains, seamounts
- **Depth Zone Classification** - Surface to hadalpelagic zone identification
- **Marine Taxa Detection** - Cnidaria, Mollusca, Arthropoda, Echinodermata
- **Pressure Adaptation Analysis** - Environmental stress indicators
- **Conservation Alerts** - Marine biodiversity threat assessment
- **Export Capabilities** - JSON export and summary reports

## 🚀 Quick Start

### 1. Install Dependencies (Optional)
```bash
pip install -r requirements.txt
```

### 2. Run AI-Enhanced Analysis
```bash
# Basic FASTA analysis with AI
python biomapper_prototype.py your_sequences.fasta

# High-throughput eDNA analysis
python biomapper_prototype.py your_edna_data.fastq

# Deep-sea specific analysis
python biomapper_prototype.py deep_sea_samples.fasta
```

### 3. Test AI Integration
```bash
# Test all AI modules
python test_ai_integration.py
```

### 4. Export Results
```bash
python biomapper_prototype.py your_sequences.fasta results.json
```

## 📊 Example Output

The prototype generates comprehensive analysis results including:

```json
{
  "status": "success",
  "analysis_type": "comprehensive_biodiversity_analysis",
  "sequence_statistics": {
    "total_sequences": 20,
    "total_base_pairs": 3902,
    "average_length": 195.1
  },
  "species_classification": [
    {
      "sequence_id": "Panthera_onca_001",
      "predicted_species": "Panthera onca",
      "confidence": 0.95,
      "iucn_status": "Near Threatened"
    }
  ],
  "biodiversity_metrics": {
    "species_richness": 9,
    "shannon_diversity_index": 2.085,
    "simpson_index": 0.865
  },
  "conservation_assessment": {
    "threatened_species_count": 4,
    "conservation_priority": "High",
    "conservation_alerts": ["Conservation Alert: Panthera onca is Near Threatened"]
  }
}
```

## 🔧 Usage Examples

### AI-Enhanced Analysis
```python
from biomapper_prototype import BioMapperPrototype

# Initialize with AI capabilities
biomapper = BioMapperPrototype()

# Analyze FASTA file with AI
results = biomapper.analyze_fasta_file("sequences.fasta")

# Process FASTQ files for eDNA
fastq_results = biomapper.process_fastq_file("edna_data.fastq")

# Export results
biomapper.export_results(results, "ai_analysis_results.json")

# Generate summary with AI insights
summary = biomapper.generate_summary_report(results)
print(summary)
```

### Individual AI Modules
```python
# Use specific AI modules
from ai_classifier import AIClassifier, DeepSeaAI
from edna_processor import eDNAProcessor
from independent_classifier import IndependentClassifier
from deepsea_optimizer import DeepSeaOptimizer

# AI classification
ai_classifier = AIClassifier()
results = ai_classifier.predict_novel_taxa(sequences)

# Deep-sea optimization
optimizer = DeepSeaOptimizer()
marine_analysis = optimizer.optimize_for_deep_sea(sequences)

# Database-independent classification
independent = IndependentClassifier()
db_free_results = independent.classify_without_database(sequences)
```

### Command Line Usage
```bash
# Basic analysis
python biomapper_prototype.py test_biodiversity_sample.fasta

# With custom output file
python biomapper_prototype.py data.fasta my_results.json

# View help
python biomapper_prototype.py
```

## 📋 Analysis Modules

### 🤖 AI-Enhanced Modules

#### 1. AI Classification (`ai_classifier.py`)
- Unsupervised clustering (K-means, DBSCAN)
- Deep learning sequence classification
- Novel taxa detection without databases
- Feature extraction from DNA sequences

#### 2. eDNA Processing (`edna_processor.py`)
- FASTQ file processing with quality filtering
- High-throughput bulk sequence handling
- OTU clustering and denoising
- Marker gene extraction (18S, COI, ITS)

#### 3. Database-Independent Classification (`independent_classifier.py`)
- Hierarchical clustering without reference databases
- Self-organizing classification system
- Intrinsic feature extraction
- Adaptive refinement algorithms

#### 4. Deep-Sea Optimization (`deepsea_optimizer.py`)
- Marine organism detection patterns
- Depth zone classification (surface to hadal)
- Eukaryotic sequence identification
- Environmental stress adaptation analysis

### 🧬 Core Analysis Modules

#### 5. Sequence Statistics
- Total sequences and base pairs
- Length distribution and statistics
- GC content and composition analysis

#### 6. Enhanced Species Classification
- AI-driven species identification
- Marine-specific classification
- Confidence scoring with novelty detection
- IUCN conservation status

#### 7. Biodiversity Metrics
- Species richness and abundance
- Shannon diversity index
- Simpson index and Pielou evenness
- Chao1 estimator

#### 8. Quality Analysis
- eDNA-specific quality assessment
- Nucleotide composition analysis
- Sequence complexity metrics
- Quality score distribution

#### 9. Conservation Assessment
- Marine biodiversity threat evaluation
- Conservation alerts for deep-sea species
- Priority species identification
- Conservation recommendations

#### 10. Phylogenetic Analysis
- Family-level grouping
- Newick tree generation
- Taxonomic relationships

#### 11. Microbiome Analysis
- Phylum distribution
- OTU clustering and analysis
- Diversity metrics
- Functional prediction

#### 12. Quantum Computing Simulation
- Quantum vs classical performance
- Alignment optimization
- Clustering analysis

#### 13. Protein Structure Prediction
- Domain identification
- Structure confidence
- Functional prediction

## 🛠️ Customization

The prototype is designed to be easily customizable:

### Adding New Species
```python
# Add to species_database in __init__ method
"New_species": {
    "common_name": "New Species",
    "scientific_name": "Species novus",
    "family": "FamilyName",
    "conservation_status": "Least Concern"
}
```

### Custom Classification Patterns
```python
# Modify _classify_by_pattern method
patterns = {
    "YOUR_PATTERN": "species_key",
    # Add more patterns
}
```

### Custom Metrics
```python
# Add new metrics to _calculate_biodiversity_metrics
def custom_metric(self, species_list):
    # Your custom calculation
    return result
```

## 📁 File Structure

```
prototype/
├── biomapper_prototype.py      # Main AI-enhanced prototype
├── ai_classifier.py           # AI classification module
├── edna_processor.py          # eDNA processing pipeline
├── independent_classifier.py  # Database-independent analysis
├── deepsea_optimizer.py       # Deep-sea optimization
├── test_ai_integration.py     # Comprehensive test suite
├── requirements.txt           # Python dependencies
├── app.py                     # Web interface
├── templates/                 # Web UI templates
├── static/                    # Web UI assets
└── README.md                  # This file
```

## 🔍 Dependencies

### Required (Built-in)
- `json` - JSON handling
- `sys` - System operations
- `os` - File operations
- `time` - Timing functions
- `math` - Mathematical operations
- `random` - Random number generation
- `datetime` - Date/time handling
- `collections` - Data structures
- `typing` - Type hints
- `warnings` - Warning handling

### AI/ML Dependencies (Enhanced Features)
- `scikit-learn` - Machine learning and clustering
- `tensorflow` - Deep learning models
- `keras` - Neural network API
- `numpy` - Numerical computing
- `pandas` - Data manipulation

### Optional (Enhanced Features)
- `Bio` (BioPython) - Biological sequence analysis
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `plotly` - Interactive plots
- `ete3` - Phylogenetic analysis
- `qiskit` - Quantum computing
- `torch` - Alternative deep learning framework

## 🚨 Error Handling

The prototype includes comprehensive error handling:

- **File Not Found** - Graceful handling of missing files
- **Invalid FASTA** - Fallback parsing for malformed files
- **Missing Dependencies** - Graceful degradation when optional packages are missing
- **Analysis Errors** - Detailed error messages and recovery

## 📈 Performance

- **Processing Speed** - Optimized for fast analysis
- **Memory Usage** - Efficient memory management
- **Scalability** - Handles large FASTA files
- **Parallel Processing** - Multi-threaded analysis where possible

## 🤝 Contributing

To contribute to the prototype:

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request

## 📄 License

This prototype is part of the BioMapper project and follows the same licensing terms.

## 🆘 Support

For support and questions:

- Check the error messages for guidance
- Review the example usage
- Test with the provided sample FASTA file
- Contact the BioMapper team

## 🔄 Version History

- **v2.0.0** - AI-Enhanced Deep-Sea eDNA Analysis
  - AI-driven classification without database dependency
  - Unsupervised learning for novel taxa discovery
  - Deep-sea specific optimization
  - High-throughput eDNA processing
  - Database-independent analysis
  - Marine organism detection
  - Eukaryotic marker gene identification

- **v1.0.0** - Initial prototype release
  - Core analysis functions
  - Standalone operation
  - Comprehensive reporting
  - Export capabilities

---

## 🧪 Testing

Run the comprehensive test suite:
```bash
python test_ai_integration.py
```

This tests:
- AI classification modules
- Deep-sea optimization
- eDNA processing pipeline
- Database-independent analysis
- Integration between all modules

**Note**: This AI-enhanced prototype addresses the specific challenges of deep-sea eDNA analysis including poor database representation, computational efficiency, and novel taxa discovery. It's designed for research and production use in marine biodiversity studies.

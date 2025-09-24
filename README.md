# BioMapper Prototype - Standalone Analysis System

## 🧬 Overview

This is a standalone prototype of the BioMapper biodiversity analysis platform. It contains all the core analysis functions and can run independently without requiring the full project setup.

## ✨ Features

### Core Analysis Functions:
- **DNA Sequence Analysis** - Parse and analyze FASTA files
- **Species Classification** - Identify species from DNA sequences
- **Biodiversity Metrics** - Calculate diversity indices and richness
- **Quality Assessment** - Analyze sequence quality and composition
- **Conservation Assessment** - Evaluate conservation status and generate alerts
- **Phylogenetic Analysis** - Build phylogenetic trees
- **Microbiome Analysis** - Analyze microbial communities
- **Quantum Computing Simulation** - Simulate quantum-enhanced analysis
- **Protein Structure Prediction** - Predict protein structures
- **Comprehensive Reporting** - Generate detailed analysis reports

### Advanced Features:
- **Pattern-based Classification** - Uses DNA sequence patterns for species identification
- **Conservation Alerts** - Automatically flags threatened species
- **Novel Species Detection** - Identifies potentially new species
- **Biodiversity Metrics** - Shannon, Simpson, Chao1 indices
- **Quality Control** - Sequence quality assessment
- **Export Capabilities** - JSON export and summary reports

## 🚀 Quick Start

### 1. Install Dependencies (Optional)
```bash
pip install -r requirements.txt
```

### 2. Run Analysis
```bash
python biomapper_prototype.py your_sequences.fasta
```

### 3. Export Results
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

### Basic Analysis
```python
from biomapper_prototype import BioMapperPrototype

# Initialize
biomapper = BioMapperPrototype()

# Analyze FASTA file
results = biomapper.analyze_fasta_file("sequences.fasta")

# Export results
biomapper.export_results(results, "analysis_results.json")

# Generate summary
summary = biomapper.generate_summary_report(results)
print(summary)
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

### 1. Sequence Statistics
- Total sequences and base pairs
- Length distribution and statistics
- Sequence composition analysis

### 2. Species Classification
- Pattern-based species identification
- Confidence scoring
- Novel species detection
- IUCN conservation status

### 3. Biodiversity Metrics
- Species richness
- Shannon diversity index
- Simpson index
- Pielou evenness
- Chao1 estimator

### 4. Quality Analysis
- GC content analysis
- Nucleotide composition
- Sequence complexity
- Quality score distribution

### 5. Conservation Assessment
- IUCN status evaluation
- Conservation alerts
- Priority species identification
- Conservation recommendations

### 6. Phylogenetic Analysis
- Family-level grouping
- Newick tree generation
- Taxonomic relationships

### 7. Microbiome Analysis
- Phylum distribution
- OTU clustering
- Diversity metrics
- Functional prediction

### 8. Quantum Computing Simulation
- Quantum vs classical performance
- Alignment optimization
- Clustering analysis

### 9. Protein Structure Prediction
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
├── biomapper_prototype.py    # Main prototype file
├── requirements.txt          # Python dependencies
└── README.md                # This file
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

### Optional (Enhanced Features)
- `Bio` (BioPython) - Biological sequence analysis
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `plotly` - Interactive plots
- `scikit-learn` - Machine learning
- `ete3` - Phylogenetic analysis
- `qiskit` - Quantum computing

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

- **v1.0.0** - Initial prototype release
  - Core analysis functions
  - Standalone operation
  - Comprehensive reporting
  - Export capabilities

---

**Note**: This is a prototype version designed for demonstration and testing. For production use, consider the full BioMapper platform with additional features and optimizations.

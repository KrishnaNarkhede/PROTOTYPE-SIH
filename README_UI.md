# BioMapper Web UI

A minimal web interface for the BioMapper biodiversity analysis prototype.

## 🚀 Quick Start

### Windows
```bash
run_ui.bat
```

### Manual Start
```bash
pip install -r requirements_ui.txt
python app.py
```

Then open: http://localhost:5000

## ✨ Features

- **Drag & Drop Upload** - Easy FASTA file upload
- **Real-time Analysis** - Live progress tracking
- **Comprehensive Results** - All 9 analysis modules displayed
- **Export Functionality** - Download results as JSON
- **Responsive Design** - Works on desktop and mobile

## 📊 Analysis Modules

1. **Sequence Statistics** - Basic sequence metrics
2. **Biodiversity Metrics** - Shannon, Simpson, Chao1 indices
3. **Conservation Status** - IUCN status and alerts
4. **Quality Analysis** - GC content and quality scores
5. **Species Classification** - Species identification table
6. **Phylogenetic Analysis** - Tree construction
7. **Microbiome Analysis** - Phylum distribution
8. **Quantum Analysis** - Performance benchmarks
9. **Protein Prediction** - Structure predictions

## 🎯 Usage

1. Open the web interface
2. Upload a FASTA file (drag & drop or click)
3. Wait for analysis to complete
4. View comprehensive results
5. Export results as JSON

## 📁 File Structure

```
prototype/
├── app.py              # Flask web application
├── templates/
│   └── index.html      # Main UI template
├── static/
│   ├── style.css       # UI styling
│   └── script.js       # Frontend functionality
├── requirements_ui.txt # UI dependencies
└── run_ui.bat         # Windows startup script
```

## 🔧 Technical Details

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **File Upload**: Secure file handling with Werkzeug
- **Analysis**: Integrated BioMapper prototype
- **Export**: Client-side JSON download

## 🌐 Browser Support

- Chrome/Edge (recommended)
- Firefox
- Safari
- Mobile browsers

## 📱 Mobile Responsive

The UI automatically adapts to mobile devices with:
- Touch-friendly upload area
- Responsive grid layout
- Optimized text sizes
- Mobile-friendly navigation
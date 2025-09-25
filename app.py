from flask import Flask, render_template, request, jsonify, send_file
import os
import json
from werkzeug.utils import secure_filename
from biomapper_prototype import BioMapperPrototype
import tempfile
import threading
import time

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching

# Initialize BioMapper with AI capabilities
biomapper = BioMapperPrototype()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'error': 'No file selected'}), 400
    
    # Check file extension
    allowed_extensions = ['.fasta', '.fa', '.fas', '.fastq', '.fq', '.gz']
    if file and any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
            
            # Quick file size check
            file_size = os.path.getsize(filepath)
            if file_size > 100 * 1024 * 1024:  # 100MB
                os.remove(filepath)
                return jsonify({'status': 'error', 'error': 'File too large. Maximum size is 100MB.'}), 400
            
            print(f"Processing file: {filename} ({file_size/1024/1024:.1f}MB)")
            
            # Determine file type and analyze accordingly
            try:
                if filename.lower().endswith(('.fastq', '.fq', '.fastq.gz', '.fq.gz')):
                    # Process FASTQ file for eDNA analysis
                    results = biomapper.process_fastq_file(filepath)
                else:
                    # Process FASTA file
                    results = biomapper.analyze_fasta_file(filepath)
            except Exception as analysis_error:
                print(f"Analysis exception: {analysis_error}")
                import traceback
                traceback.print_exc()
                raise analysis_error
            
            # Clean up uploaded file first
            if os.path.exists(filepath):
                os.remove(filepath)
            
            # Ensure results are valid
            if not results:
                return jsonify({'status': 'error', 'error': 'No results generated'}), 500
            
            if results.get('status') == 'error':
                return jsonify(results), 500
            
            # Store results for export
            biomapper.last_results = results
            
            print(f"Analysis completed successfully for {filename}")
            return jsonify(results)
            
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            print(f"Analysis error: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'status': 'error', 'error': f'Analysis failed: {str(e)}'}), 500
    
    return jsonify({'status': 'error', 'error': 'Invalid file format. Please upload FASTA or FASTQ files.'}), 400

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'version': biomapper.version,
        'ai_features': [
            'AI-Driven Classification',
            'Unsupervised Learning',
            'Database-Independent Analysis', 
            'Novel Taxa Detection',
            'Deep-Sea Optimization',
            'High-Throughput eDNA Processing'
        ],
        'supported_formats': ['FASTA', 'FASTQ', 'Compressed (.gz)'],
        'max_file_size': '100MB'
    })

@app.route('/features')
def get_features():
    """Get available AI features and capabilities"""
    try:
        from ai_classifier import AIClassifier
        from edna_processor import eDNAProcessor
        from independent_classifier import IndependentClassifier
        from deepsea_optimizer import DeepSeaOptimizer
        ai_available = True
    except ImportError:
        ai_available = False
    
    return jsonify({
        'ai_modules_available': ai_available,
        'capabilities': {
            'ai_classification': ai_available,
            'deep_sea_analysis': ai_available,
            'novel_taxa_detection': ai_available,
            'edna_processing': ai_available,
            'database_independent': ai_available,
            'unsupervised_learning': ai_available
        },
        'supported_formats': ['FASTA', 'FASTQ', 'Compressed (.gz)'],
        'max_file_size': '100MB'
    })

@app.route('/export/<analysis_id>')
def export_results(analysis_id):
    # For demo, return the last analysis
    if hasattr(biomapper, 'last_results'):
        output_file = biomapper.export_results(biomapper.last_results)
        return send_file(output_file, as_attachment=True)
    return jsonify({'error': 'No analysis results to export'}), 404

if __name__ == '__main__':
    print("🧬 Starting BioMapper AI Web Interface...")
    print(f"🤖 Version: {biomapper.version} - AI Enhanced")
    print("🌊 Features: Deep-Sea eDNA Analysis with AI")
    print("🌐 Access at: http://localhost:5000")
    print("📊 Supported: FASTA, FASTQ, AI Classification, Novel Taxa Detection")
    app.run(debug=True, host='0.0.0.0', port=5000)
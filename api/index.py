from flask import Flask, render_template, request, jsonify
import os
import sys
import json
from werkzeug.utils import secure_filename
import tempfile

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biomapper_prototype import BioMapperPrototype

app = Flask(__name__, 
           template_folder='../templates',
           static_folder='../static')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB for Vercel

# Initialize BioMapper
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
    
    allowed_extensions = ['.fasta', '.fa', '.fas', '.fastq', '.fq']
    if file and any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        filename = secure_filename(file.filename)
        
        try:
            # Read file content directly without saving to disk
            file_content = file.read().decode('utf-8')
            
            # Create temporary file-like object
            import io
            temp_file = io.StringIO(file_content)
            
            # Analyze based on file type
            if filename.lower().endswith(('.fastq', '.fq')):
                results = biomapper.process_fastq_content(file_content)
            else:
                results = biomapper.analyze_fasta_content(file_content)
            
            if not results or results.get('status') == 'error':
                return jsonify({'status': 'error', 'error': 'Analysis failed'}), 500
            
            return jsonify(results)
            
        except Exception as e:
            return jsonify({'status': 'error', 'error': f'Analysis failed: {str(e)}'}), 500
    
    return jsonify({'status': 'error', 'error': 'Invalid file format'}), 400

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'version': biomapper.version,
        'platform': 'Vercel',
        'ai_features': [
            'AI-Driven Classification',
            'Novel Taxa Detection',
            'Deep-Sea Optimization'
        ]
    })

# Vercel serverless function handler
def handler(request):
    return app(request.environ, lambda status, headers: None)

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, render_template, request, jsonify, send_file
import os
import json
from werkzeug.utils import secure_filename
from biomapper_prototype import BioMapperPrototype
import tempfile

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

biomapper = BioMapperPrototype()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename.lower().endswith(('.fasta', '.fa', '.fas')):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            results = biomapper.analyze_fasta_file(filepath)
            os.remove(filepath)  # Clean up
            return jsonify(results)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file format. Please upload a FASTA file.'}), 400

@app.route('/export/<analysis_id>')
def export_results(analysis_id):
    # For demo, return the last analysis
    if hasattr(biomapper, 'last_results'):
        output_file = biomapper.export_results(biomapper.last_results)
        return send_file(output_file, as_attachment=True)
    return jsonify({'error': 'No analysis results to export'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
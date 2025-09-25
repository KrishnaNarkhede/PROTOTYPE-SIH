// Modern AI-Enhanced BioMapper JavaScript
let analysisResults = null;
let selectedFile = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeFileUpload();
    initializeDragAndDrop();
});

// File upload initialization
function initializeFileUpload() {
    const fileInput = document.getElementById('fileInput');
    const fileInfo = document.getElementById('fileInfo');
    
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            selectedFile = file;
            displayFileInfo(file);
            enableAnalysisButton();
        }
    });
}

// Drag and drop functionality
function initializeDragAndDrop() {
    const uploadBox = document.getElementById('uploadBox');
    
    uploadBox.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadBox.classList.add('dragover');
    });
    
    uploadBox.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadBox.classList.remove('dragover');
    });
    
    uploadBox.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadBox.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            if (isValidFile(file)) {
                selectedFile = file;
                document.getElementById('fileInput').files = files;
                displayFileInfo(file);
                enableAnalysisButton();
            } else {
                showError('Please select a valid FASTA or FASTQ file.');
            }
        }
    });
    
    // Click to upload
    uploadBox.addEventListener('click', function() {
        document.getElementById('fileInput').click();
    });
}

// Validate file type
function isValidFile(file) {
    const validExtensions = ['.fasta', '.fa', '.fas', '.fastq', '.fq', '.gz'];
    const fileName = file.name.toLowerCase();
    return validExtensions.some(ext => fileName.endsWith(ext));
}

// Display file information
function displayFileInfo(file) {
    const fileInfo = document.getElementById('fileInfo');
    const fileSize = (file.size / 1024 / 1024).toFixed(2);
    
    fileInfo.innerHTML = `
        <div style="display: flex; align-items: center; gap: 10px;">
            <i class="fas fa-file-alt" style="color: #4CAF50;"></i>
            <div>
                <strong>${file.name}</strong><br>
                <small>Size: ${fileSize} MB | Type: ${getFileType(file.name)}</small>
            </div>
        </div>
    `;
    fileInfo.style.display = 'block';
}

// Get file type description
function getFileType(filename) {
    const ext = filename.toLowerCase();
    if (ext.includes('.fastq') || ext.includes('.fq')) return 'FASTQ (High-throughput sequencing)';
    if (ext.includes('.fasta') || ext.includes('.fa') || ext.includes('.fas')) return 'FASTA (Sequence data)';
    return 'Sequence file';
}

// Enable analysis button
function enableAnalysisButton() {
    const analyzeBtn = document.getElementById('analyzeBtn');
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = '<i class="fas fa-play"></i> Start AI Analysis';
}

// Start analysis
async function startAnalysis() {
    if (!selectedFile) {
        showError('Please select a file first.');
        return;
    }
    
    // Show loading section
    document.querySelector('.upload-section').style.display = 'none';
    document.getElementById('analysisSection').style.display = 'block';
    
    // Simulate analysis progress
    await simulateAnalysisProgress();
    
    // Upload and analyze file
    try {
        const formData = new FormData();
        formData.append('file', selectedFile);
        
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 minute timeout
        
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData,
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const results = await response.json();
        
        // Check if analysis failed
        if (results.status === 'error') {
            throw new Error(results.error || 'Analysis failed');
        }
        
        analysisResults = results;
        
        // Hide loading and show results
        document.getElementById('analysisSection').style.display = 'none';
        displayResults(results);
        
    } catch (error) {
        console.error('Analysis failed:', error);
        if (error.name === 'AbortError') {
            showError('Analysis timed out. Please try with a smaller file or contact support.');
        } else {
            showError(`Analysis failed: ${error.message}`);
        }
        resetToUpload();
    }
}

// Simulate analysis progress
async function simulateAnalysisProgress() {
    const statusText = document.getElementById('statusText');
    const progressFill = document.getElementById('progress');
    
    const steps = [
        { text: 'Initializing AI modules...', progress: 10 },
        { text: 'Loading sequence data...', progress: 25 },
        { text: 'Running AI classification...', progress: 45 },
        { text: 'Analyzing deep-sea patterns...', progress: 65 },
        { text: 'Detecting novel taxa...', progress: 80 },
        { text: 'Generating results...', progress: 95 },
        { text: 'Finalizing analysis...', progress: 100 }
    ];
    
    for (const step of steps) {
        statusText.textContent = step.text;
        progressFill.style.width = step.progress + '%';
        await sleep(800);
    }
}

// Sleep function for delays
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Display analysis results
function displayResults(results) {
    document.getElementById('resultsSection').style.display = 'block';
    document.getElementById('resultsSection').classList.add('fade-in');
    
    // Display AI summary
    displayAISummary(results);
    
    // Display individual result sections
    displaySequenceStats(results.sequence_statistics);
    displayAIClassification(results.ai_classification);
    displayDeepSeaAnalysis(results.deepsea_analysis);
    displayNovelTaxa(results.ai_classification);
    displayeDNAProcessing(results.edna_processing);
    displayBiodiversityMetrics(results.biodiversity_metrics);
    displayConservationStatus(results.conservation_assessment);
    displayQualityAnalysis(results.quality_analysis);
    displaySpeciesClassification(results.species_classification);
    displayEnhancedAnalysis(results.enhanced_analysis);
    displayExecutiveSummary(results.comprehensive_report, results);
}

// Display AI summary
function displayAISummary(results) {
    const aiSummary = document.getElementById('aiSummary');
    const aiResults = results.ai_classification || {};
    const deepSeaResults = results.deepsea_analysis?.optimization_summary || {};
    
    aiSummary.innerHTML = `
        <h3><i class="fas fa-brain"></i> AI Analysis Summary</h3>
        <p>Advanced machine learning analysis completed successfully</p>
        <div class="ai-summary-grid">
            <div class="ai-metric">
                <div class="ai-metric-value">${aiResults.novel_taxa_count || 0}</div>
                <div class="ai-metric-label">Novel Taxa</div>
            </div>
            <div class="ai-metric">
                <div class="ai-metric-value">${deepSeaResults.marine_sequences || 0}</div>
                <div class="ai-metric-label">Marine Sequences</div>
            </div>
            <div class="ai-metric">
                <div class="ai-metric-value">${deepSeaResults.eukaryotic_sequences || 0}</div>
                <div class="ai-metric-label">Eukaryotic Sequences</div>
            </div>
            <div class="ai-metric">
                <div class="ai-metric-value">${(aiResults.clustering_quality * 100).toFixed(1)}%</div>
                <div class="ai-metric-label">AI Confidence</div>
            </div>
        </div>
    `;
}

// Display sequence statistics
function displaySequenceStats(stats) {
    const container = document.getElementById('sequenceStats');
    container.innerHTML = `
        <div class="metric-item">
            <span class="metric-label">Total Sequences</span>
            <span class="metric-value">${stats.total_sequences}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Total Base Pairs</span>
            <span class="metric-value">${stats.total_base_pairs.toLocaleString()}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Average Length</span>
            <span class="metric-value">${stats.average_length.toFixed(1)} bp</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Length Range</span>
            <span class="metric-value">${stats.min_length} - ${stats.max_length} bp</span>
        </div>
    `;
}

// Display AI classification results
function displayAIClassification(aiResults) {
    const container = document.getElementById('aiClassificationResults');
    if (!aiResults) {
        container.innerHTML = '<p>AI classification not available</p>';
        return;
    }
    
    container.innerHTML = `
        <div class="metric-item">
            <span class="metric-label">Novel Taxa Detected</span>
            <span class="metric-value success">${aiResults.novel_taxa_count || 0}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Clustering Quality</span>
            <span class="metric-value">${(aiResults.clustering_quality * 100).toFixed(1)}%</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">AI Method</span>
            <span class="metric-value">Unsupervised Learning</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Database Dependency</span>
            <span class="metric-value success">Independent</span>
        </div>
    `;
}

// Display deep-sea analysis
function displayDeepSeaAnalysis(deepSeaResults) {
    const container = document.getElementById('deepSeaResults');
    if (!deepSeaResults) {
        container.innerHTML = '<p>Deep-sea analysis not available</p>';
        return;
    }
    
    const summary = deepSeaResults.optimization_summary || {};
    container.innerHTML = `
        <div class="metric-item">
            <span class="metric-label">Marine Sequences</span>
            <span class="metric-value success">${summary.marine_sequences || 0}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Deep-Sea Sequences</span>
            <span class="metric-value">${summary.deep_sea_sequences || 0}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">High Relevance</span>
            <span class="metric-value">${summary.high_relevance_sequences || 0}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Optimization</span>
            <span class="metric-value success">Applied</span>
        </div>
    `;
}

// Display novel taxa results
function displayNovelTaxa(aiResults) {
    const container = document.getElementById('novelTaxaResults');
    if (!aiResults) {
        container.innerHTML = '<p>Novel taxa detection not available</p>';
        return;
    }
    
    const novelCount = aiResults.novel_taxa_count || 0;
    const totalPredictions = aiResults.predictions?.length || 0;
    const noveltyRate = totalPredictions > 0 ? (novelCount / totalPredictions * 100).toFixed(1) : 0;
    
    container.innerHTML = `
        <div class="metric-item">
            <span class="metric-label">Novel Taxa Found</span>
            <span class="metric-value ${novelCount > 0 ? 'warning' : 'success'}">${novelCount}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Novelty Rate</span>
            <span class="metric-value">${noveltyRate}%</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Detection Method</span>
            <span class="metric-value">AI Clustering</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Verification</span>
            <span class="metric-value warning">Recommended</span>
        </div>
    `;
}

// Display eDNA processing results
function displayeDNAProcessing(ednaResults) {
    const container = document.getElementById('ednaResults');
    if (!ednaResults) {
        container.innerHTML = '<p>eDNA processing not available</p>';
        return;
    }
    
    container.innerHTML = `
        <div class="metric-item">
            <span class="metric-label">OTUs Identified</span>
            <span class="metric-value success">${ednaResults.otu_count || 0}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">18S Markers</span>
            <span class="metric-value">${ednaResults.marker_genes?.['18S_count'] || 0}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">COI Markers</span>
            <span class="metric-value">${ednaResults.marker_genes?.['COI_count'] || 0}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Quality Control</span>
            <span class="metric-value success">Applied</span>
        </div>
    `;
}

// Display biodiversity metrics
function displayBiodiversityMetrics(metrics) {
    const container = document.getElementById('biodiversityMetrics');
    container.innerHTML = `
        <div class="metric-item">
            <span class="metric-label">Species Richness</span>
            <span class="metric-value">${metrics.species_richness}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Shannon Diversity</span>
            <span class="metric-value">${metrics.shannon_diversity_index.toFixed(3)}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Simpson Index</span>
            <span class="metric-value">${metrics.simpson_index.toFixed(3)}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Dominant Species</span>
            <span class="metric-value">${metrics.dominant_species}</span>
        </div>
    `;
}

// Display conservation status
function displayConservationStatus(conservation) {
    const container = document.getElementById('conservationStatus');
    const priority = conservation.conservation_priority;
    const priorityClass = priority === 'High' ? 'error' : priority === 'Medium' ? 'warning' : 'success';
    
    let alertsHtml = '';
    if (conservation.conservation_alerts && conservation.conservation_alerts.length > 0) {
        alertsHtml = `
            <div style="margin-top: 15px; padding: 10px; background: #fff3cd; border-radius: 5px;">
                <h4 style="margin: 0 0 10px 0; color: #856404;">🚨 Conservation Alerts:</h4>
                <ul style="margin: 0; padding-left: 20px;">
                    ${conservation.conservation_alerts.map(alert => `<li style="color: #856404; margin: 5px 0;">${alert}</li>`).join('')}
                </ul>
            </div>
        `;
    }
    
    let threatenedSpeciesHtml = '';
    if (conservation.threatened_species_details && conservation.threatened_species_details.length > 0) {
        threatenedSpeciesHtml = `
            <div style="margin-top: 15px; padding: 10px; background: #f8d7da; border-radius: 5px;">
                <h4 style="margin: 0 0 10px 0; color: #721c24;">⚠️ Threatened Species Found:</h4>
                ${conservation.threatened_species_details.map(species => 
                    `<div style="margin: 5px 0; color: #721c24;">
                        <strong>${species.asv_id}</strong> - ${species.common_name} (<em>${species.scientific_name}</em>) 
                        <span style="background: #dc3545; color: white; padding: 2px 6px; border-radius: 3px; font-size: 12px;">${species.status}</span>
                    </div>`
                ).join('')}
            </div>
        `;
    }
    
    container.innerHTML = `
        <div class="metric-item">
            <span class="metric-label">Threatened Species</span>
            <span class="metric-value ${conservation.threatened_species_count > 0 ? 'warning' : 'success'}">${conservation.threatened_species_count}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Conservation Priority</span>
            <span class="metric-value ${priorityClass}">${priority}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Alerts Generated</span>
            <span class="metric-value">${conservation.conservation_alerts.length}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Species Assessed</span>
            <span class="metric-value">${conservation.total_species_assessed}</span>
        </div>
        ${threatenedSpeciesHtml}
        ${alertsHtml}
    `;
}

// Display quality analysis
function displayQualityAnalysis(quality) {
    const container = document.getElementById('qualityAnalysis');
    const avgQuality = quality.average_quality_score;
    const qualityClass = avgQuality > 35 ? 'success' : avgQuality > 25 ? 'warning' : 'error';
    
    container.innerHTML = `
        <div class="metric-item">
            <span class="metric-label">GC Content</span>
            <span class="metric-value">${quality.gc_content}%</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Average Quality</span>
            <span class="metric-value ${qualityClass}">${avgQuality.toFixed(1)}/50</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Sequence Complexity</span>
            <span class="metric-value">${quality.sequence_complexity.toFixed(4)}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">High Quality Seqs</span>
            <span class="metric-value success">${quality.quality_distribution.high_quality}</span>
        </div>
    `;
}

// Display species classification
function displaySpeciesClassification(species) {
    const container = document.getElementById('speciesClassification');
    
    if (!species || species.length === 0) {
        container.innerHTML = '<p>No species classification data available</p>';
        return;
    }
    
    let tableHTML = `
        <table class="species-table">
            <thead>
                <tr>
                    <th>ASV ID</th>
                    <th>Common Name</th>
                    <th>Scientific Name</th>
                    <th>Conservation Status</th>
                    <th>Confidence</th>
                    <th>Method</th>
                </tr>
            </thead>
            <tbody>
    `;
    
    species.slice(0, 10).forEach(sp => {
        // Handle enhanced analysis format
        const asvId = sp.ASV_ID || sp.sequence_id || 'Unknown';
        const scientificName = sp.Assigned_Name || sp.predicted_species || 'Unknown';
        const commonName = sp.Common_Name || sp.common_name || 'Unknown';
        const conservationStatus = sp.Conservation_Status || 'Not Evaluated';
        const confidence = sp.Assignment_Confidence || sp.confidence || 0;
        const method = sp.Assignment_Method || sp.classification_method || 'enhanced';
        
        const confidenceClass = confidence > 0.8 ? 'confidence-high' : confidence > 0.5 ? 'confidence-medium' : 'confidence-low';
        const conservationClass = ['Critically Endangered', 'Endangered'].includes(conservationStatus) ? 'confidence-low' :
                                 ['Vulnerable', 'Near Threatened'].includes(conservationStatus) ? 'confidence-medium' : 'confidence-high';
        
        tableHTML += `
            <tr>
                <td>${asvId}</td>
                <td><strong>${commonName}</strong></td>
                <td><em>${scientificName}</em></td>
                <td><span class="confidence-badge ${conservationClass}">${conservationStatus}</span></td>
                <td><span class="confidence-badge ${confidenceClass}">${(confidence * 100).toFixed(1)}%</span></td>
                <td>${method}</td>
            </tr>
        `;
    });
    
    tableHTML += '</tbody></table>';
    
    if (species.length > 10) {
        tableHTML += `<p style="margin-top: 15px; color: #666;"><em>Showing first 10 of ${species.length} ASVs</em></p>`;
    }
    
    container.innerHTML = tableHTML;
}

// Display executive summary
function displayExecutiveSummary(report, results) {
    const container = document.getElementById('executiveSummary');
    if (!report) {
        container.innerHTML = '<p>Executive summary not available</p>';
        return;
    }
    
    const summary = report.executive_summary || {};
    const findings = report.key_findings || [];
    const recommendations = report.recommendations || [];
    
    // Get actual values from results
    const actualNovelCount = results?.ai_classification?.novel_taxa_count || 0;
    const actualKnownCount = results?.ai_classification?.known_species_count || 0;
    const totalSequences = results?.sequence_statistics?.total_sequences || 0;
    const threatenedCount = results?.conservation_assessment?.threatened_species_count || 0;
    
    container.innerHTML = `
        <div class="summary-highlights">
            <div class="highlight-item">
                <div class="highlight-value">${totalSequences}</div>
                <div class="highlight-label">Total Sequences</div>
            </div>
            <div class="highlight-item">
                <div class="highlight-value">${actualKnownCount}</div>
                <div class="highlight-label">Known Species</div>
            </div>
            <div class="highlight-item">
                <div class="highlight-value">${actualNovelCount}</div>
                <div class="highlight-label">Novel Taxa</div>
            </div>
            <div class="highlight-item">
                <div class="highlight-value">${threatenedCount}</div>
                <div class="highlight-label">Threatened Species</div>
            </div>
        </div>
        
        <div style="margin-top: 30px;">
            <h4><i class="fas fa-lightbulb"></i> Key Findings</h4>
            <ul style="margin: 15px 0; padding-left: 20px;">
                <li>Analyzed ${totalSequences} DNA sequences with enhanced AI classification</li>
                <li>Identified ${actualKnownCount} known species from reference databases</li>
                <li>Detected ${actualNovelCount} potentially novel taxa requiring further investigation</li>
                <li>Found ${threatenedCount} threatened species requiring conservation attention</li>
                <li>Achieved ${results?.data_validation?.accuracy_metrics?.average_confidence ? (results.data_validation.accuracy_metrics.average_confidence * 100).toFixed(1) : 0}% average confidence in classifications</li>
            </ul>
            
            <h4><i class="fas fa-tasks"></i> Recommendations</h4>
            <ul style="margin: 15px 0; padding-left: 20px;">
                ${actualNovelCount > 0 ? '<li>Conduct further taxonomic verification for novel taxa candidates</li>' : ''}
                ${threatenedCount > 0 ? '<li>Implement immediate conservation measures for threatened species</li>' : ''}
                <li>Continue regular biodiversity monitoring using eDNA analysis</li>
                <li>Expand reference database with newly identified sequences</li>
            </ul>
        </div>
    `;
}

// Download results as JSON
function downloadResults() {
    if (!analysisResults) {
        showError('No results to download');
        return;
    }
    
    const dataStr = JSON.stringify(analysisResults, null, 2);
    const dataBlob = new Blob([dataStr], {type: 'application/json'});
    
    const link = document.createElement('a');
    link.href = URL.createObjectURL(dataBlob);
    link.download = `biomapper_analysis_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
    link.click();
}

// Go back to file selection
function goBack() {
    // Reset to main page
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('analysisSection').style.display = 'none';
    document.querySelector('.upload-section').style.display = 'block';
    
    // Reset file input
    document.getElementById('fileInput').value = '';
    document.getElementById('fileInfo').innerHTML = '';
    document.getElementById('fileInfo').style.display = 'none';
    
    // Reset progress
    document.getElementById('progress').style.width = '0%';
    document.getElementById('statusText').textContent = 'Initializing AI modules...';
    
    // Clear results
    analysisResults = null;
    selectedFile = null;
    
    // Reset analyze button
    const analyzeBtn = document.getElementById('analyzeBtn');
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<i class="fas fa-play"></i> Start AI Analysis';
}

// Reset to upload state
function resetToUpload() {
    document.querySelector('.upload-section').style.display = 'block';
    document.getElementById('analysisSection').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'none';
}

// Show error message
function showError(message) {
    // Create error notification
    const errorDiv = document.createElement('div');
    errorDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #f44336;
        color: white;
        padding: 15px 20px;
        border-radius: 5px;
        z-index: 1000;
        max-width: 400px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    `;
    errorDiv.innerHTML = `
        <strong>Error:</strong> ${message}
        <button onclick="this.parentElement.remove()" style="
            background: none;
            border: none;
            color: white;
            float: right;
            font-size: 18px;
            cursor: pointer;
            margin-left: 10px;
        ">&times;</button>
    `;
    
    document.body.appendChild(errorDiv);
    
    // Auto remove after 10 seconds
    setTimeout(() => {
        if (errorDiv.parentElement) {
            errorDiv.remove();
        }
    }, 10000);
}

// Display enhanced analysis results
function displayEnhancedAnalysis(enhancedData) {
    if (!enhancedData || enhancedData.length === 0) return;
    
    // Populate enhanced analysis table
    const container = document.getElementById('enhancedAnalysisContent');
    let tableHTML = `
        <table class="species-table">
            <thead>
                <tr>
                    <th>ASV ID</th>
                    <th>Common Name</th>
                    <th>Conservation Status</th>
                    <th>Novelty Flag</th>
                    <th>Ecological Role</th>
                </tr>
            </thead>
            <tbody>
    `;
    
    enhancedData.slice(0, 10).forEach(asv => {
        const qualityClass = asv.Quality_Score > 0.8 ? 'confidence-high' : 
                           asv.Quality_Score > 0.5 ? 'confidence-medium' : 'confidence-low';
        const noveltyClass = asv.Novelty_Flag === 'Candidate_Novel' ? 'confidence-low' :
                           asv.Novelty_Flag === 'Divergent' ? 'confidence-medium' : 'confidence-high';
        const conservationClass = ['Critically Endangered', 'Endangered'].includes(asv.Conservation_Status) ? 'confidence-low' :
                                 ['Vulnerable', 'Near Threatened'].includes(asv.Conservation_Status) ? 'confidence-medium' : 'confidence-high';
        
        tableHTML += `
            <tr>
                <td>${asv.ASV_ID}</td>
                <td><strong>${asv.Common_Name || 'Unknown'}</strong></td>
                <td><span class="confidence-badge ${conservationClass}">${asv.Conservation_Status || 'Not Evaluated'}</span></td>
                <td><span class="confidence-badge ${noveltyClass}">${asv.Novelty_Flag}</span></td>
                <td>${asv.Predicted_Ecological_Role}</td>
            </tr>
        `;
    });
    
    tableHTML += '</tbody></table>';
    
    if (enhancedData.length > 10) {
        tableHTML += `<p style="margin-top: 15px; color: #666;"><em>Showing first 10 of ${enhancedData.length} ASVs</em></p>`;
    }
    
    container.innerHTML = tableHTML;
}

// Show info message
function showInfo(message) {
    alert('Info: ' + message);
}
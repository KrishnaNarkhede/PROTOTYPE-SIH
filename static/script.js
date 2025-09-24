let currentResults = null;

// File upload handling
const fileInput = document.getElementById('fileInput');
const uploadBox = document.getElementById('uploadBox');

fileInput.addEventListener('change', handleFileSelect);
uploadBox.addEventListener('click', () => fileInput.click());
uploadBox.addEventListener('dragover', handleDragOver);
uploadBox.addEventListener('drop', handleDrop);

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        uploadFile(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    uploadBox.classList.add('dragover');
}

function handleDrop(event) {
    event.preventDefault();
    uploadBox.classList.remove('dragover');
    const file = event.dataTransfer.files[0];
    if (file) {
        uploadFile(file);
    }
}

function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    showAnalysisSection();
    
    fetch('/analyze', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert('Error: ' + data.error);
            hideAnalysisSection();
        } else {
            currentResults = data;
            displayResults(data);
        }
    })
    .catch(error => {
        alert('Error: ' + error.message);
        hideAnalysisSection();
    });
}

function showAnalysisSection() {
    document.getElementById('analysisSection').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';
    
    // Animate progress bar
    const progress = document.getElementById('progress');
    let width = 0;
    const interval = setInterval(() => {
        width += Math.random() * 15;
        if (width >= 100) {
            width = 100;
            clearInterval(interval);
        }
        progress.style.width = width + '%';
    }, 200);
}

function hideAnalysisSection() {
    document.getElementById('analysisSection').style.display = 'none';
}

function displayResults(data) {
    hideAnalysisSection();
    document.getElementById('resultsSection').style.display = 'block';
    
    // Sequence Statistics
    const stats = data.sequence_statistics;
    document.getElementById('sequenceStats').innerHTML = `
        <div class="metric">
            <span class="metric-label">Total Sequences:</span>
            <span class="metric-value">${stats.total_sequences}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Total Base Pairs:</span>
            <span class="metric-value">${stats.total_base_pairs.toLocaleString()}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Average Length:</span>
            <span class="metric-value">${stats.average_length.toFixed(1)} bp</span>
        </div>
        <div class="metric">
            <span class="metric-label">Min Length:</span>
            <span class="metric-value">${stats.min_length} bp</span>
        </div>
        <div class="metric">
            <span class="metric-label">Max Length:</span>
            <span class="metric-value">${stats.max_length} bp</span>
        </div>
    `;
    
    // Biodiversity Metrics
    const bio = data.biodiversity_metrics;
    document.getElementById('biodiversityMetrics').innerHTML = `
        <div class="metric">
            <span class="metric-label">Species Richness:</span>
            <span class="metric-value">${bio.species_richness}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Shannon Index:</span>
            <span class="metric-value">${bio.shannon_diversity_index}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Simpson Index:</span>
            <span class="metric-value">${bio.simpson_index}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Pielou Evenness:</span>
            <span class="metric-value">${bio.pielou_evenness}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Chao1 Estimator:</span>
            <span class="metric-value">${bio.chao1_estimator}</span>
        </div>
    `;
    
    // Conservation Status
    const conservation = data.conservation_assessment;
    const alertsHtml = conservation.conservation_alerts.map(alert => 
        `<div class="conservation-alert ${alert.includes('Endangered') ? 'high' : ''}">${alert}</div>`
    ).join('');
    
    document.getElementById('conservationStatus').innerHTML = `
        <div class="metric">
            <span class="metric-label">Threatened Species:</span>
            <span class="metric-value">${conservation.threatened_species_count}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Priority Level:</span>
            <span class="metric-value">${conservation.conservation_priority}</span>
        </div>
        <div style="margin-top: 15px;">
            <strong>Conservation Alerts:</strong>
            ${alertsHtml || '<p>No conservation alerts</p>'}
        </div>
    `;
    
    // Quality Analysis
    const quality = data.quality_analysis;
    document.getElementById('qualityAnalysis').innerHTML = `
        <div class="metric">
            <span class="metric-label">GC Content:</span>
            <span class="metric-value">${quality.gc_content}%</span>
        </div>
        <div class="metric">
            <span class="metric-label">Average Quality:</span>
            <span class="metric-value">${quality.average_quality_score}/50</span>
        </div>
        <div class="metric">
            <span class="metric-label">Sequence Complexity:</span>
            <span class="metric-value">${quality.sequence_complexity}</span>
        </div>
        <div class="metric">
            <span class="metric-label">High Quality Seqs:</span>
            <span class="metric-value">${quality.quality_distribution.high_quality}</span>
        </div>
    `;
    
    // Species Classification
    const species = data.species_classification; // Show all species
    const speciesTableHtml = `
        <div class="table-container">
            <table class="species-table">
                <thead>
                    <tr>
                        <th>Sequence ID</th>
                        <th>Scientific Name</th>
                        <th>Common Name</th>
                        <th>Confidence</th>
                        <th>IUCN Status</th>
                    </tr>
                </thead>
                <tbody>
                    ${species.map(s => `
                        <tr>
                            <td>${s.sequence_id}</td>
                            <td>${s.predicted_species}</td>
                            <td>${s.common_name || 'Unknown'}</td>
                            <td>${(s.confidence * 100).toFixed(1)}%</td>
                            <td>${s.iucn_status}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;
    document.getElementById('speciesClassification').innerHTML = speciesTableHtml;
    
    // Phylogenetic Analysis
    const phylo = data.phylogenetic_analysis;
    const treeHtml = generateTreeVisualization(phylo.family_distribution);
    
    document.getElementById('phylogeneticAnalysis').innerHTML = `
        <div class="phylo-metrics">
            <div class="metric">
                <span class="metric-label">Families:</span>
                <span class="metric-value">${phylo.families_identified}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Total Nodes:</span>
                <span class="metric-value">${phylo.total_nodes}</span>
            </div>
        </div>
        <div class="tree-container">
            <h4>🌳 Family Tree</h4>
            ${treeHtml}
        </div>
        <div class="newick-container">
            <strong>Newick Format:</strong>
            <code>${phylo.newick_tree}</code>
        </div>
    `;
    
    // Microbiome Analysis
    const microbiome = data.microbiome_analysis;
    const phylaHtml = Object.entries(microbiome.phylum_distribution)
        .map(([phylum, percent]) => `
            <div class="metric">
                <span class="metric-label">${phylum}:</span>
                <span class="metric-value">${percent}%</span>
            </div>
        `).join('');
    
    document.getElementById('microbiomeAnalysis').innerHTML = `
        ${phylaHtml}
        <div style="margin-top: 15px;">
            <div class="metric">
                <span class="metric-label">Total OTUs:</span>
                <span class="metric-value">${microbiome.otu_analysis.total_otus}</span>
            </div>
        </div>
    `;
    
    // Quantum Analysis
    const quantum = data.quantum_analysis;
    document.getElementById('quantumAnalysis').innerHTML = `
        <div class="metric">
            <span class="metric-label">Speed Ratio:</span>
            <span class="metric-value">${quantum.quantum_benchmark.speed_ratio}x</span>
        </div>
        <div class="metric">
            <span class="metric-label">Quantum Time:</span>
            <span class="metric-value">${quantum.quantum_benchmark.quantum_time_ms}ms</span>
        </div>
        <div class="metric">
            <span class="metric-label">Classical Time:</span>
            <span class="metric-value">${quantum.quantum_benchmark.classical_time_ms}ms</span>
        </div>
        <div class="metric">
            <span class="metric-label">Alignment Score:</span>
            <span class="metric-value">${quantum.alignment_analysis.average_alignment_score}</span>
        </div>
    `;
    
    // Protein Analysis
    const protein = data.protein_analysis;
    document.getElementById('proteinAnalysis').innerHTML = `
        <div class="metric">
            <span class="metric-label">Structures Predicted:</span>
            <span class="metric-value">${protein.total_structures_predicted}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Average Confidence:</span>
            <span class="metric-value">${(protein.average_confidence * 100).toFixed(1)}%</span>
        </div>
        <div style="margin-top: 10px;">
            ${protein.protein_predictions.slice(0, 3).map(p => `
                <div style="font-size: 0.85rem; margin: 5px 0; padding: 5px; background: #f9f9f9; border-radius: 3px;">
                    <strong>${p.sequence_id}:</strong> ${p.structure_quality} quality (${(p.confidence_score * 100).toFixed(1)}%)
                </div>
            `).join('')}
        </div>
    `;
    
    // Executive Summary
    const summary = data.comprehensive_report.executive_summary;
    const findings = data.comprehensive_report.key_findings;
    
    document.getElementById('executiveSummary').innerHTML = `
        <div class="summary-item">
            <strong>Analysis Overview:</strong> Processed ${summary.total_sequences} sequences, identified ${summary.species_identified} species
        </div>
        <div class="summary-item">
            <strong>Data Quality:</strong> ${summary.data_quality}
        </div>
        <div class="summary-item">
            <strong>Conservation Priority:</strong> ${summary.conservation_priority}
        </div>
        <div class="summary-item">
            <strong>Processing Time:</strong> ${data.processing_time}
        </div>
        <div style="margin-top: 15px;">
            <strong>Key Findings:</strong>
            <ul style="margin-left: 20px; margin-top: 10px;">
                ${findings.map(finding => `<li>${finding}</li>`).join('')}
            </ul>
        </div>
    `;
}

function generateTreeVisualization(familyDistribution) {
    let treeHtml = '<div class="tree-diagram">';
    
    Object.entries(familyDistribution).forEach(([family, species], index) => {
        const colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe'];
        const color = colors[index % colors.length];
        
        treeHtml += `
            <div class="family-branch" style="border-left-color: ${color}">
                <div class="family-name" style="background: ${color}">${family}</div>
                <div class="species-list">
                    ${species.slice(0, 4).map(sp => `
                        <div class="species-node">${sp}</div>
                    `).join('')}
                    ${species.length > 4 ? `<div class="more-species">+${species.length - 4} more</div>` : ''}
                </div>
            </div>
        `;
    });
    
    treeHtml += '</div>';
    return treeHtml;
}

function exportResults() {
    if (currentResults) {
        const dataStr = JSON.stringify(currentResults, null, 2);
        const dataBlob = new Blob([dataStr], {type: 'application/json'});
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `biomapper_analysis_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
        link.click();
        URL.revokeObjectURL(url);
    }
}
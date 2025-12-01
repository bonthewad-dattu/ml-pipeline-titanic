// ML Pipeline Frontend JavaScript

// DOM Content Loaded
document.addEventListener('DOMContentLoaded', function() {
    loadModelInfo();
    initializeEventListeners();
});

// Initialize event listeners
function initializeEventListeners() {
    // Form submission
    document.getElementById('predictionForm').addEventListener('submit', handleFormSubmit);
    
    // Real-time form validation
    const inputs = document.querySelectorAll('input, select');
    inputs.forEach(input => {
        input.addEventListener('input', validateForm);
    });
}

// Load model information
async function loadModelInfo() {
    try {
        const response = await fetch('/model_info');
        const data = await response.json();
        
        const modelInfoDiv = document.getElementById('modelInfo');
        modelInfoDiv.innerHTML = `
            <div class="model-details">
                <p><strong>Model Type:</strong> ${data.model_type}</p>
                <p><strong>Features:</strong> ${data.feature_count}</p>
                <p><strong>Status:</strong> 
                    <span class="status-indicator status-healthy"></span>
                    ${data.loaded_successfully ? 'Loaded' : 'Not Loaded'}
                </p>
                <p><strong>CV Folds:</strong> ${data.config?.cv_folds || 'N/A'}</p>
            </div>
        `;
    } catch (error) {
        console.error('Error loading model info:', error);
        document.getElementById('modelInfo').innerHTML = `
            <div class="alert alert-warning">
                <i class="fas fa-exclamation-triangle"></i> Unable to load model information
            </div>
        `;
    }
}

// Handle form submission
async function handleFormSubmit(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const inputData = Object.fromEntries(formData.entries());
    
    // Convert numeric fields
    inputData.Pclass = parseInt(inputData.Pclass);
    inputData.Age = parseFloat(inputData.Age);
    inputData.SibSp = parseInt(inputData.SibSp);
    inputData.Parch = parseInt(inputData.Parch);
    inputData.Fare = parseFloat(inputData.Fare);
    
    // Show loading state
    showLoading();
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(inputData)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            displayPredictionResult(result, inputData);
            displayFeatureImportance(result.feature_importance);
        } else {
            showError(result.error || 'Prediction failed');
        }
    } catch (error) {
        console.error('Prediction error:', error);
        showError('Network error: Unable to connect to prediction service');
    }
}

// Display prediction result
function displayPredictionResult(result, inputData) {
    const resultCard = document.getElementById('resultCard');
    const resultContent = document.getElementById('resultContent');
    const resultHeader = document.getElementById('resultHeader');
    
    // Set card styling based on prediction
    if (result.prediction === 1) {
        resultCard.className = 'card shadow-lg result-card survived fade-in';
        resultHeader.innerHTML = '<h4><i class="fas fa-check-circle"></i> Survival Predicted!</h4>';
    } else {
        resultCard.className = 'card shadow-lg result-card not-survived fade-in';
        resultHeader.innerHTML = '<h4><i class="fas fa-times-circle"></i> Did Not Survive</h4>';
    }
    
    // Create result content
    resultContent.innerHTML = `
        <div class="prediction-result">
            <div class="survival-status ${result.prediction === 1 ? 'text-success' : 'text-danger'} mb-3">
                <h2 class="display-4 fw-bold">${result.survival_status}</h2>
            </div>
            
            <div class="confidence-meter mb-4">
                <h5>Confidence: ${result.confidence}</h5>
                <div class="progress">
                    <div class="progress-bar ${result.prediction === 1 ? 'bg-success' : 'bg-danger'}" 
                         role="progressbar" 
                         style="width: ${result.confidence}" 
                         aria-valuenow="${parseFloat(result.confidence)}" 
                         aria-valuemin="0" 
                         aria-valuemax="100">
                        ${result.confidence}
                    </div>
                </div>
            </div>
            
            <div class="model-info">
                <p><strong>Model:</strong> ${result.model_used}</p>
                <p><strong>Probability:</strong> ${(result.probability * 100).toFixed(2)}%</p>
                <p><strong>Timestamp:</strong> ${result.timestamp}</p>
            </div>
            
            <div class="passenger-summary mt-3 p-3 bg-light rounded">
                <h6><i class="fas fa-user"></i> Passenger Summary</h6>
                <small>
                    ${inputData.Name ? `<strong>Name:</strong> ${inputData.Name}<br>` : ''}
                    <strong>Class:</strong> ${inputData.Pclass}<br>
                    <strong>Gender:</strong> ${inputData.Sex}<br>
                    <strong>Age:</strong> ${inputData.Age}
                </small>
            </div>
        </div>
    `;
    
    // Show the result card with animation
    resultCard.style.display = 'block';
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Display feature importance
function displayFeatureImportance(featureImportance) {
    const featureDiv = document.getElementById('featureImportance');
    
    if (!featureImportance || Object.keys(featureImportance).length === 0) {
        featureDiv.innerHTML = '<p class="text-muted">Feature importance not available for this model</p>';
        return;
    }
    
    // Sort features by importance
    const sortedFeatures = Object.entries(featureImportance)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 8); // Show top 8 features
    
    let html = '<h6>Top Features Influencing Prediction:</h6>';
    
    sortedFeatures.forEach(([feature, importance]) => {
        const percentage = (importance * 100).toFixed(1);
        const featureName = feature.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase());
        
        html += `
            <div class="feature-item mb-2">
                <div class="d-flex justify-content-between">
                    <small>${featureName}</small>
                    <small>${percentage}%</small>
                </div>
                <div class="feature-bar" style="width: ${percentage}%"></div>
            </div>
        `;
    });
    
    featureDiv.innerHTML = html;
}

// Batch prediction
async function predictBatch() {
    const batchInput = document.getElementById('batchInput');
    const batchResults = document.getElementById('batchResults');
    
    let passengers;
    try {
        passengers = JSON.parse(batchInput.value);
    } catch (error) {
        showError('Invalid JSON format');
        return;
    }
    
    if (!Array.isArray(passengers)) {
        showError('Please provide an array of passenger objects');
        return;
    }
    
    // Show loading
    batchResults.innerHTML = '<div class="text-center"><div class="loading-spinner"></div><p>Processing batch prediction...</p></div>';
    
    try {
        const response = await fetch('/batch_predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ passengers: passengers })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            displayBatchResults(result);
        } else {
            showError(result.error || 'Batch prediction failed');
        }
    } catch (error) {
        console.error('Batch prediction error:', error);
        showError('Network error: Unable to connect to prediction service');
    }
}

// Display batch results
function displayBatchResults(result) {
    const batchResults = document.getElementById('batchResults');
    
    let html = `
        <div class="batch-summary card mb-3">
            <div class="card-body">
                <h5>Batch Prediction Summary</h5>
                <div class="row text-center">
                    <div class="col-md-3">
                        <h3>${result.total_passengers}</h3>
                        <small>Total Passengers</small>
                    </div>
                    <div class="col-md-3">
                        <h3 class="text-success">${result.survived_count}</h3>
                        <small>Survived</small>
                    </div>
                    <div class="col-md-3">
                        <h3 class="text-danger">${result.total_passengers - result.survived_count}</h3>
                        <small>Did Not Survive</small>
                    </div>
                    <div class="col-md-3">
                        <h3>${result.survival_rate}</h3>
                        <small>Survival Rate</small>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="batch-details">
            <h5>Individual Predictions:</h5>
    `;
    
    result.predictions.forEach((prediction, index) => {
        const passenger = prediction.passenger_data;
        html += `
            <div class="batch-result-item ${prediction.prediction === 1 ? 'survived' : 'not-survived'}">
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <strong>Passenger ${index + 1}</strong>
                        <small class="ms-2">${passenger.Sex}, ${passenger.Age} years, Class ${passenger.Pclass}</small>
                    </div>
                    <div>
                        <span class="badge ${prediction.prediction === 1 ? 'bg-success' : 'bg-danger'}">
                            ${prediction.survival_status} (${prediction.confidence})
                        </span>
                    </div>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    batchResults.innerHTML = html;
}

// Load example data for batch prediction
function loadExample() {
    const exampleData = `[
  {"Pclass": 1, "Sex": "female", "Age": 25, "SibSp": 0, "Parch": 0, "Fare": 50.0, "Embarked": "C", "Name": "Emily Johnson"},
  {"Pclass": 3, "Sex": "male", "Age": 30, "SibSp": 0, "Parch": 0, "Fare": 7.25, "Embarked": "S", "Name": "James Smith"},
  {"Pclass": 2, "Sex": "female", "Age": 18, "SibSp": 1, "Parch": 0, "Fare": 25.0, "Embarked": "Q", "Name": "Sarah Wilson"},
  {"Pclass": 1, "Sex": "male", "Age": 45, "SibSp": 1, "Parch": 2, "Fare": 100.0, "Embarked": "C", "Name": "Robert Brown"}
]`;
    
    document.getElementById('batchInput').value = exampleData;
}

// Test API connection
async function testAPI() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        
        if (response.ok) {
            showSuccess('API is healthy and responding!');
        } else {
            showError('API health check failed');
        }
    } catch (error) {
        showError('Unable to connect to API');
    }
}

// Utility functions
function showLoading() {
    // You can implement a more sophisticated loading indicator
    console.log('Loading...');
}

function showError(message) {
    // Create toast notification
    const toast = document.createElement('div');
    toast.className = 'alert alert-danger alert-dismissible fade show position-fixed';
    toast.style.top = '20px';
    toast.style.right = '20px';
    toast.style.zIndex = '9999';
    toast.style.minWidth = '300px';
    toast.innerHTML = `
        <strong><i class="fas fa-exclamation-triangle"></i> Error:</strong> ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(toast);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (toast.parentNode) {
            toast.parentNode.removeChild(toast);
        }
    }, 5000);
}

function showSuccess(message) {
    // Create toast notification
    const toast = document.createElement('div');
    toast.className = 'alert alert-success alert-dismissible fade show position-fixed';
    toast.style.top = '20px';
    toast.style.right = '20px';
    toast.style.zIndex = '9999';
    toast.style.minWidth = '300px';
    toast.innerHTML = `
        <strong><i class="fas fa-check-circle"></i> Success:</strong> ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(toast);
    
    // Auto remove after 3 seconds
    setTimeout(() => {
        if (toast.parentNode) {
            toast.parentNode.removeChild(toast);
        }
    }, 3000);
}

function validateForm() {
    // Basic form validation can be added here
    const form = document.getElementById('predictionForm');
    const submitButton = form.querySelector('button[type="submit"]');
    
    const isValid = form.checkValidity();
    submitButton.disabled = !isValid;
    
    return isValid;
}

// Export functions for global access
window.predictBatch = predictBatch;
window.loadExample = loadExample;
window.testAPI = testAPI;
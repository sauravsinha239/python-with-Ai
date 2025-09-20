"""
Web Application Integration with Machine Learning

This module demonstrates how to integrate ML models into web applications.
"""

from flask import Flask, request, jsonify, render_template_string
import numpy as np
import joblib
import json
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Initialize Flask app
app = Flask(__name__)

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ML Model Web App</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .form-group { margin: 15px 0; }
        label { display: block; margin-bottom: 5px; }
        input { padding: 8px; margin-bottom: 10px; width: 200px; }
        button { padding: 10px 20px; background-color: #007bff; color: white; border: none; cursor: pointer; }
        button:hover { background-color: #0056b3; }
        .result { margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; }
        .error { color: red; }
        .success { color: green; }
    </style>
</head>
<body>
    <h1>Machine Learning Model Prediction</h1>
    <p>This web app demonstrates integration of ML models with Flask.</p>
    
    <h2>Make a Prediction</h2>
    <form id="predictionForm">
        <div class="form-group">
            <label for="feature1">Feature 1:</label>
            <input type="number" step="0.01" id="feature1" name="feature1" value="0.5" required>
        </div>
        <div class="form-group">
            <label for="feature2">Feature 2:</label>
            <input type="number" step="0.01" id="feature2" name="feature2" value="-0.3" required>
        </div>
        <div class="form-group">
            <label for="feature3">Feature 3:</label>
            <input type="number" step="0.01" id="feature3" name="feature3" value="1.2" required>
        </div>
        <div class="form-group">
            <label for="feature4">Feature 4:</label>
            <input type="number" step="0.01" id="feature4" name="feature4" value="-0.8" required>
        </div>
        <button type="submit">Predict</button>
    </form>
    
    <div id="result" class="result" style="display: none;"></div>
    
    <h2>Model Information</h2>
    <p><strong>Model Type:</strong> Random Forest Classifier</p>
    <p><strong>Features:</strong> 4 numerical features</p>
    <p><strong>Output:</strong> Binary classification (0 or 1)</p>
    
    <script>
        document.getElementById('predictionForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const features = [
                parseFloat(formData.get('feature1')),
                parseFloat(formData.get('feature2')),
                parseFloat(formData.get('feature3')),
                parseFloat(formData.get('feature4'))
            ];
            
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({features: features})
            })
            .then(response => response.json())
            .then(data => {
                const resultDiv = document.getElementById('result');
                resultDiv.style.display = 'block';
                
                if (data.error) {
                    resultDiv.innerHTML = '<p class="error">Error: ' + data.error + '</p>';
                } else {
                    resultDiv.innerHTML = `
                        <h3>Prediction Result</h3>
                        <p class="success"><strong>Predicted Class:</strong> ${data.prediction}</p>
                        <p><strong>Probability:</strong> ${(data.probability * 100).toFixed(2)}%</p>
                        <p><strong>Input Features:</strong> [${features.join(', ')}]</p>
                    `;
                }
            })
            .catch(error => {
                const resultDiv = document.getElementById('result');
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = '<p class="error">Error: ' + error.message + '</p>';
            });
        });
    </script>
</body>
</html>
"""

# Global variables for model and scaler
model = None
scaler = None

def train_and_save_model():
    """Train a model and save it for the web app."""
    global model, scaler
    
    # Create sample dataset
    X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, 
                              random_state=42, n_redundant=0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Create directory if it doesn't exist
    os.makedirs('web_apps', exist_ok=True)
    
    # Save model and scaler
    joblib.dump(model, 'web_apps/flask_model.pkl')
    joblib.dump(scaler, 'web_apps/flask_scaler.pkl')
    
    print("Model and scaler saved successfully!")
    return model, scaler

def load_model():
    """Load the trained model and scaler."""
    global model, scaler
    
    try:
        model = joblib.load('web_apps/flask_model.pkl')
        scaler = joblib.load('web_apps/flask_scaler.pkl')
        print("Model and scaler loaded successfully!")
        return True
    except FileNotFoundError:
        print("Model files not found. Training new model...")
        train_and_save_model()
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@app.route('/')
def home():
    """Serve the main page."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if 'features' not in data:
            return jsonify({'error': 'No features provided'}), 400
        
        features = data['features']
        
        # Validate input
        if len(features) != 4:
            return jsonify({'error': 'Expected 4 features'}), 400
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0].max()
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'features': features
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model-info')
def model_info():
    """Get information about the loaded model."""
    try:
        info = {
            'model_type': type(model).__name__,
            'n_features': model.n_features_in_,
            'n_classes': len(model.classes_),
            'classes': model.classes_.tolist()
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=== Flask ML Web App ===")
    
    # Load or train model
    if load_model():
        print("Starting Flask development server...")
        print("Open your browser and go to: http://127.0.0.1:5000")
        print("Press Ctrl+C to stop the server")
        
        # Run the Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Cannot start web app.")
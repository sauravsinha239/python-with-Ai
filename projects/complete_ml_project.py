"""
Simple project template for combining machine learning with web development.

This example demonstrates how to build a complete ML-powered web application
from data processing to model training to deployment.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from flask import Flask, request, jsonify, render_template_string

class MLPipeline:
    """Complete ML pipeline from data to deployment."""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.is_trained = False
    
    def load_data(self, data_path=None):
        """Load and prepare sample data."""
        if data_path and os.path.exists(data_path):
            df = pd.read_csv(data_path)
        else:
            # Create sample data if no file provided
            np.random.seed(42)
            n_samples = 1000
            
            data = {
                'age': np.random.randint(18, 65, n_samples),
                'income': np.random.normal(50000, 20000, n_samples),
                'experience': np.random.randint(0, 40, n_samples),
                'education_score': np.random.normal(7, 2, n_samples),
                'satisfaction': np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
            }
            df = pd.DataFrame(data)
            df['income'] = np.clip(df['income'], 20000, 150000)
            df['education_score'] = np.clip(df['education_score'], 1, 10)
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data for training."""
        # Separate features and target
        if 'satisfaction' in df.columns:
            X = df.drop('satisfaction', axis=1)
            y = df['satisfaction']
        else:
            X = df
            y = None
        
        self.feature_names = X.columns.tolist()
        return X, y
    
    def train_model(self, X, y):
        """Train the machine learning model."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.is_trained = True
        print(f"Model trained with accuracy: {accuracy:.3f}")
        
        return accuracy
    
    def save_model(self, filepath):
        """Save the trained model."""
        if self.is_trained:
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained
            }
            joblib.dump(model_data, filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No trained model to save!")
    
    def load_model(self, filepath):
        """Load a trained model."""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.is_trained = model_data['is_trained']
            print(f"Model loaded from {filepath}")
            return True
        except FileNotFoundError:
            print(f"Model file {filepath} not found!")
            return False
    
    def predict(self, features):
        """Make predictions on new data."""
        if not self.is_trained:
            raise ValueError("Model not trained! Train or load a model first.")
        
        # Convert input to proper format
        if isinstance(features, dict):
            features = [features[name] for name in self.feature_names]
        
        features_array = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features_array)[0]
        probability = self.model.predict_proba(features_array)[0].max()
        
        return {
            'prediction': int(prediction),
            'probability': float(probability),
            'confidence': 'high' if probability > 0.8 else 'medium' if probability > 0.6 else 'low'
        }

# Web Application
app = Flask(__name__)
ml_pipeline = MLPipeline()

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ML Project Template</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .container { background: #f5f5f5; padding: 20px; border-radius: 10px; margin: 20px 0; }
        .form-group { margin: 15px 0; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input { padding: 10px; width: 200px; border: 1px solid #ccc; border-radius: 5px; }
        button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .result { margin-top: 20px; padding: 15px; background: #e9ecef; border-radius: 5px; }
        .success { color: green; font-weight: bold; }
        .error { color: red; font-weight: bold; }
    </style>
</head>
<body>
    <h1>🤖 ML Project Template</h1>
    <p>A complete example of integrating machine learning with web development.</p>
    
    <div class="container">
        <h2>📊 Model Information</h2>
        <p><strong>Model Type:</strong> Random Forest Classifier</p>
        <p><strong>Purpose:</strong> Predict employee satisfaction based on demographics</p>
        <p><strong>Features:</strong> Age, Income, Experience, Education Score</p>
        <p><strong>Status:</strong> <span class="success">Model Ready</span></p>
    </div>
    
    <div class="container">
        <h2>🎯 Make a Prediction</h2>
        <form id="predictionForm">
            <div class="form-group">
                <label>Age (years):</label>
                <input type="number" name="age" value="30" min="18" max="65" required>
            </div>
            <div class="form-group">
                <label>Annual Income ($):</label>
                <input type="number" name="income" value="50000" min="20000" max="150000" required>
            </div>
            <div class="form-group">
                <label>Experience (years):</label>
                <input type="number" name="experience" value="5" min="0" max="40" required>
            </div>
            <div class="form-group">
                <label>Education Score (1-10):</label>
                <input type="number" step="0.1" name="education_score" value="7.5" min="1" max="10" required>
            </div>
            <button type="submit">Predict Satisfaction</button>
        </form>
        
        <div id="result" class="result" style="display: none;"></div>
    </div>
    
    <div class="container">
        <h2>💡 About This Project</h2>
        <p>This template demonstrates:</p>
        <ul>
            <li>Data loading and preprocessing</li>
            <li>Model training and evaluation</li>
            <li>Model persistence and loading</li>
            <li>Web API creation with Flask</li>
            <li>Interactive web interface</li>
            <li>Complete ML deployment pipeline</li>
        </ul>
        <p><strong>Next Steps:</strong> Customize this template for your own ML projects!</p>
    </div>
    
    <script>
        document.getElementById('predictionForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const features = {
                age: parseInt(formData.get('age')),
                income: parseFloat(formData.get('income')),
                experience: parseInt(formData.get('experience')),
                education_score: parseFloat(formData.get('education_score'))
            };
            
            fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(features)
            })
            .then(response => response.json())
            .then(data => {
                const resultDiv = document.getElementById('result');
                resultDiv.style.display = 'block';
                
                if (data.error) {
                    resultDiv.innerHTML = `<p class="error">Error: ${data.error}</p>`;
                } else {
                    const satisfaction = data.prediction === 1 ? 'Satisfied' : 'Not Satisfied';
                    const confidence = data.confidence.charAt(0).toUpperCase() + data.confidence.slice(1);
                    
                    resultDiv.innerHTML = `
                        <h3>Prediction Result</h3>
                        <p class="success"><strong>Employee Satisfaction:</strong> ${satisfaction}</p>
                        <p><strong>Confidence:</strong> ${confidence} (${(data.probability * 100).toFixed(1)}%)</p>
                        <p><strong>Input:</strong> Age ${features.age}, Income $${features.income}, Experience ${features.experience}y, Education ${features.education_score}/10</p>
                    `;
                }
            })
            .catch(error => {
                const resultDiv = document.getElementById('result');
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = `<p class="error">Error: ${error.message}</p>`;
            });
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Serve the main page."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        data = request.get_json()
        result = ml_pipeline.predict(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train_model():
    """Endpoint to retrain the model."""
    try:
        df = ml_pipeline.load_data()
        X, y = ml_pipeline.preprocess_data(df)
        accuracy = ml_pipeline.train_model(X, y)
        ml_pipeline.save_model('projects/complete_project_model.pkl')
        
        return jsonify({
            'message': 'Model trained successfully!',
            'accuracy': accuracy
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def main():
    """Main function to demonstrate the complete pipeline."""
    print("=== Complete ML Project Template ===\n")
    
    # Create projects directory
    os.makedirs('projects', exist_ok=True)
    
    # Step 1: Data Loading
    print("Step 1: Loading data...")
    df = ml_pipeline.load_data()
    print(f"Loaded dataset with shape: {df.shape}")
    
    # Step 2: Data Preprocessing
    print("\nStep 2: Preprocessing data...")
    X, y = ml_pipeline.preprocess_data(df)
    print(f"Features: {ml_pipeline.feature_names}")
    
    # Step 3: Model Training
    print("\nStep 3: Training model...")
    accuracy = ml_pipeline.train_model(X, y)
    
    # Step 4: Save Model
    print("\nStep 4: Saving model...")
    ml_pipeline.save_model('projects/complete_project_model.pkl')
    
    # Step 5: Test Prediction
    print("\nStep 5: Testing prediction...")
    sample_features = {
        'age': 30,
        'income': 60000,
        'experience': 5,
        'education_score': 8.0
    }
    result = ml_pipeline.predict(sample_features)
    print(f"Sample prediction: {result}")
    
    # Step 6: Launch Web App
    print("\nStep 6: Launching web application...")
    print("Visit: http://localhost:5000")
    print("Press Ctrl+C to stop")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()
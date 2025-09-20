"""
Machine Learning Fundamentals

This module demonstrates basic machine learning concepts and implementations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, make_regression
import joblib

def create_regression_dataset():
    """Create a sample regression dataset."""
    X, y = make_regression(n_samples=1000, n_features=5, noise=10, random_state=42)
    
    feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df

def create_classification_dataset():
    """Create a sample classification dataset."""
    X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, 
                              random_state=42, n_redundant=0)
    
    feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df

def linear_regression_example():
    """Demonstrate linear regression."""
    print("=== Linear Regression Example ===")
    
    # Create dataset
    df = create_regression_dataset()
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.3f}")
    
    # Feature importance
    print("\nFeature Coefficients:")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"{feature}: {coef:.3f}")
    
    # Save the model
    joblib.dump(model, 'machine_learning/linear_regression_model.pkl')
    print("Model saved as 'machine_learning/linear_regression_model.pkl'")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Linear Regression: Actual vs Predicted')
    plt.savefig('machine_learning/linear_regression_results.png')
    plt.close()
    
    return model

def classification_example():
    """Demonstrate classification with multiple algorithms."""
    print("\n=== Classification Example ===")
    
    # Create dataset
    df = create_classification_dataset()
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        
        # Train model
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        print(f"Accuracy: {accuracy:.3f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        model_filename = f'machine_learning/{name.lower().replace(" ", "_")}_model.pkl'
        joblib.dump(model, model_filename)
        print(f"Model saved as '{model_filename}'")
    
    # Compare results
    print("\n=== Model Comparison ===")
    for name, accuracy in results.items():
        print(f"{name}: {accuracy:.3f}")
    
    return models

def feature_importance_analysis():
    """Demonstrate feature importance analysis."""
    print("\n=== Feature Importance Analysis ===")
    
    # Create dataset
    df = create_classification_dataset()
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Train Random Forest for feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Feature Importance:")
    print(feature_importance)
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance (Random Forest)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('machine_learning/feature_importance.png')
    plt.close()
    
    print("Feature importance plot saved as 'machine_learning/feature_importance.png'")

def model_prediction_example():
    """Demonstrate how to load and use a trained model."""
    print("\n=== Model Prediction Example ===")
    
    try:
        # Load a saved model
        model = joblib.load('machine_learning/linear_regression_model.pkl')
        
        # Create sample data for prediction
        sample_data = np.random.randn(1, 5)  # 1 sample, 5 features
        
        # Make prediction
        prediction = model.predict(sample_data)
        
        print(f"Sample input: {sample_data[0]}")
        print(f"Prediction: {prediction[0]:.2f}")
        
    except FileNotFoundError:
        print("Model file not found. Run the linear regression example first.")

if __name__ == "__main__":
    print("=== Machine Learning Fundamentals Demo ===\n")
    
    # Linear Regression
    regression_model = linear_regression_example()
    
    # Classification
    classification_models = classification_example()
    
    # Feature Importance
    feature_importance_analysis()
    
    # Model Prediction
    model_prediction_example()
    
    print(f"\n=== Demo Complete ===")
    print("Check the 'machine_learning' folder for saved models and plots.")
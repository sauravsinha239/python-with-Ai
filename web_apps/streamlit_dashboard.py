"""
Streamlit Machine Learning Dashboard

An interactive web dashboard for machine learning model exploration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# Configure Streamlit page
st.set_page_config(
    page_title="ML Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main Streamlit application."""
    st.title("🤖 Machine Learning Dashboard")
    st.markdown("Interactive dashboard for exploring machine learning concepts and models")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Data Exploration", "Model Training", "Model Prediction", "Data Visualization"]
    )
    
    if page == "Data Exploration":
        data_exploration_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Model Prediction":
        model_prediction_page()
    elif page == "Data Visualization":
        data_visualization_page()

def data_exploration_page():
    """Data exploration section."""
    st.header("📊 Data Exploration")
    
    # Dataset selection
    dataset_type = st.selectbox(
        "Choose dataset type:",
        ["Classification", "Regression"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_samples = st.slider("Number of samples", 100, 2000, 1000)
        n_features = st.slider("Number of features", 2, 10, 4)
    
    with col2:
        if dataset_type == "Classification":
            n_classes = st.slider("Number of classes", 2, 5, 2)
        noise = st.slider("Noise level", 0.0, 1.0, 0.1)
    
    # Generate dataset
    if st.button("Generate Dataset"):
        if dataset_type == "Classification":
            X, y = make_classification(
                n_samples=n_samples, 
                n_features=n_features, 
                n_classes=n_classes,
                n_redundant=0,
                random_state=42
            )
        else:
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                noise=noise * 10,
                random_state=42
            )
        
        # Create DataFrame
        feature_names = [f'Feature_{i+1}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['Target'] = y
        
        # Store in session state
        st.session_state.dataset = df
        st.session_state.dataset_type = dataset_type
        
        st.success(f"Dataset generated with {n_samples} samples and {n_features} features!")
    
    # Display dataset if available
    if 'dataset' in st.session_state:
        df = st.session_state.dataset
        
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Samples", df.shape[0])
        with col2:
            st.metric("Features", df.shape[1] - 1)
        with col3:
            if st.session_state.dataset_type == "Classification":
                st.metric("Classes", df['Target'].nunique())
            else:
                st.metric("Target Range", f"{df['Target'].min():.2f} to {df['Target'].max():.2f}")
        
        # Show data
        st.subheader("Data Sample")
        st.dataframe(df.head(10))
        
        # Basic statistics
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe())

def model_training_page():
    """Model training section."""
    st.header("🎯 Model Training")
    
    if 'dataset' not in st.session_state:
        st.warning("Please generate a dataset in the Data Exploration section first.")
        return
    
    df = st.session_state.dataset
    dataset_type = st.session_state.dataset_type
    
    # Model selection
    if dataset_type == "Classification":
        model_options = ["Random Forest", "Logistic Regression"]
    else:
        model_options = ["Random Forest", "Linear Regression"]
    
    selected_model = st.selectbox("Choose a model:", model_options)
    
    # Model parameters
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Test set size", 0.1, 0.5, 0.2)
        random_state = st.number_input("Random state", value=42)
    
    with col2:
        if "Random Forest" in selected_model:
            n_estimators = st.slider("Number of trees", 10, 200, 100)
            max_depth = st.slider("Max depth", 1, 20, 10)
    
    if st.button("Train Model"):
        # Prepare data
        X = df.drop('Target', axis=1)
        y = df['Target']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features for logistic regression
        if selected_model == "Logistic Regression":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        # Train model
        if selected_model == "Random Forest" and dataset_type == "Classification":
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
        elif selected_model == "Random Forest" and dataset_type == "Regression":
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
        elif selected_model == "Logistic Regression":
            model = LogisticRegression(random_state=random_state)
        else:  # Linear Regression
            model = LinearRegression()
        
        # Fit model
        with st.spinner("Training model..."):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Store model in session state
        st.session_state.trained_model = model
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.y_pred = y_pred
        
        if selected_model == "Logistic Regression":
            st.session_state.scaler = scaler
        
        # Display results
        st.success("Model trained successfully!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if dataset_type == "Classification":
                accuracy = accuracy_score(y_test, y_pred)
                st.metric("Accuracy", f"{accuracy:.3f}")
            else:
                mse = mean_squared_error(y_test, y_pred)
                st.metric("MSE", f"{mse:.3f}")
        
        with col2:
            if dataset_type == "Classification":
                st.text("Classification Report:")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.json(report)
            else:
                from sklearn.metrics import r2_score
                r2 = r2_score(y_test, y_pred)
                st.metric("R² Score", f"{r2:.3f}")
        
        # Feature importance for Random Forest
        if "Random Forest" in selected_model:
            st.subheader("Feature Importance")
            feature_names = X.columns if hasattr(X, 'columns') else [f'Feature_{i}' for i in range(X.shape[1])]
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h')
            st.plotly_chart(fig, use_container_width=True)

def model_prediction_page():
    """Model prediction section."""
    st.header("🔮 Model Prediction")
    
    if 'trained_model' not in st.session_state:
        st.warning("Please train a model in the Model Training section first.")
        return
    
    model = st.session_state.trained_model
    dataset_type = st.session_state.dataset_type
    
    st.subheader("Make Predictions")
    
    # Get feature names
    if 'dataset' in st.session_state:
        feature_names = [col for col in st.session_state.dataset.columns if col != 'Target']
        n_features = len(feature_names)
    else:
        n_features = 4
        feature_names = [f'Feature_{i+1}' for i in range(n_features)]
    
    # Input fields for features
    col1, col2 = st.columns(2)
    features = []
    
    for i, name in enumerate(feature_names):
        with col1 if i % 2 == 0 else col2:
            value = st.number_input(f"{name}", value=0.0, key=f"feature_{i}")
            features.append(value)
    
    if st.button("Make Prediction"):
        # Prepare input
        input_data = np.array(features).reshape(1, -1)
        
        # Scale if needed
        if 'scaler' in st.session_state:
            input_data = st.session_state.scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        if dataset_type == "Classification":
            probability = model.predict_proba(input_data)[0]
            st.success(f"Predicted Class: {prediction}")
            
            # Show probabilities
            classes = model.classes_
            prob_df = pd.DataFrame({
                'Class': classes,
                'Probability': probability
            })
            
            fig = px.bar(prob_df, x='Class', y='Probability')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success(f"Predicted Value: {prediction:.3f}")

def data_visualization_page():
    """Data visualization section."""
    st.header("📈 Data Visualization")
    
    if 'dataset' not in st.session_state:
        st.warning("Please generate a dataset in the Data Exploration section first.")
        return
    
    df = st.session_state.dataset
    
    # Visualization type selection
    viz_type = st.selectbox(
        "Choose visualization type:",
        ["Distribution Plots", "Correlation Heatmap", "Scatter Plot", "Box Plot"]
    )
    
    if viz_type == "Distribution Plots":
        # Select features to plot
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_features = st.multiselect("Select features:", numeric_columns, default=numeric_columns[:3])
        
        if selected_features:
            fig, axes = plt.subplots(1, len(selected_features), figsize=(15, 5))
            if len(selected_features) == 1:
                axes = [axes]
            
            for i, feature in enumerate(selected_features):
                axes[i].hist(df[feature], bins=30, alpha=0.7)
                axes[i].set_title(f'Distribution of {feature}')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Frequency')
            
            st.pyplot(fig)
    
    elif viz_type == "Correlation Heatmap":
        numeric_df = df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    
    elif viz_type == "Scatter Plot":
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            x_feature = st.selectbox("X-axis feature:", numeric_columns)
        with col2:
            y_feature = st.selectbox("Y-axis feature:", numeric_columns)
        
        if x_feature and y_feature:
            fig = px.scatter(df, x=x_feature, y=y_feature, color='Target')
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Box Plot":
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_feature = st.selectbox("Select feature:", numeric_columns)
        
        if selected_feature:
            fig = px.box(df, y=selected_feature, color='Target')
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
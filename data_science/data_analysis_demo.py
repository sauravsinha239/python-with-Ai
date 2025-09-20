"""
Data Science Fundamentals

This module demonstrates core data science concepts and techniques.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_sample_data():
    """Create a sample dataset for analysis."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'education_years': np.random.normal(16, 3, n_samples),
        'experience': np.random.normal(10, 5, n_samples),
        'satisfaction': np.random.normal(7, 2, n_samples)
    }
    
    # Ensure realistic constraints
    data['age'] = np.clip(data['age'], 18, 65)
    data['income'] = np.clip(data['income'], 20000, 150000)
    data['education_years'] = np.clip(data['education_years'], 12, 25)
    data['experience'] = np.clip(data['experience'], 0, 40)
    data['satisfaction'] = np.clip(data['satisfaction'], 1, 10)
    
    return pd.DataFrame(data)

def exploratory_data_analysis(df):
    """Perform basic exploratory data analysis."""
    print("=== Exploratory Data Analysis ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Dataset info:")
    print(df.info())
    print(f"\nDescriptive statistics:")
    print(df.describe())
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    return df

def data_visualization(df):
    """Create visualizations for data analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Histogram of age distribution
    axes[0, 0].hist(df['age'], bins=30, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Age Distribution')
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Frequency')
    
    # Scatter plot: Income vs Experience
    axes[0, 1].scatter(df['experience'], df['income'], alpha=0.6)
    axes[0, 1].set_title('Income vs Experience')
    axes[0, 1].set_xlabel('Experience (years)')
    axes[0, 1].set_ylabel('Income ($)')
    
    # Box plot of satisfaction by education level
    education_bins = pd.cut(df['education_years'], bins=3, labels=['Low', 'Medium', 'High'])
    df_temp = df.copy()
    df_temp['education_level'] = education_bins
    df_temp.boxplot(column='satisfaction', by='education_level', ax=axes[1, 0])
    axes[1, 0].set_title('Satisfaction by Education Level')
    axes[1, 0].set_xlabel('Education Level')
    axes[1, 0].set_ylabel('Satisfaction Score')
    
    # Correlation heatmap
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=axes[1, 1])
    axes[1, 1].set_title('Correlation Heatmap')
    
    plt.tight_layout()
    plt.savefig('data_science/data_analysis_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Data analysis plots saved as 'data_science/data_analysis_plots.png'")

def statistical_analysis(df):
    """Perform basic statistical analysis."""
    print("\n=== Statistical Analysis ===")
    
    # Correlation analysis
    income_experience_corr = df['income'].corr(df['experience'])
    print(f"Correlation between Income and Experience: {income_experience_corr:.3f}")
    
    # Hypothesis testing
    high_education = df[df['education_years'] > df['education_years'].median()]
    low_education = df[df['education_years'] <= df['education_years'].median()]
    
    # T-test for satisfaction difference
    t_stat, p_value = stats.ttest_ind(high_education['satisfaction'], 
                                      low_education['satisfaction'])
    
    print(f"T-test for satisfaction difference:")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value: {p_value:.3f}")
    
    if p_value < 0.05:
        print("Significant difference in satisfaction between education levels")
    else:
        print("No significant difference in satisfaction between education levels")

def data_preprocessing_example(df):
    """Demonstrate data preprocessing techniques."""
    print("\n=== Data Preprocessing ===")
    
    # Handle outliers using IQR method
    def remove_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    # Remove outliers from income
    df_clean = remove_outliers_iqr(df, 'income')
    print(f"Original dataset size: {len(df)}")
    print(f"After outlier removal: {len(df_clean)}")
    
    # Feature scaling (normalization)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    # Scale numerical features
    numerical_features = ['age', 'income', 'education_years', 'experience']
    df_scaled = df_clean.copy()
    df_scaled[numerical_features] = scaler.fit_transform(df_clean[numerical_features])
    
    print(f"\nScaled features (first 5 rows):")
    print(df_scaled[numerical_features].head())
    
    return df_clean, df_scaled

if __name__ == "__main__":
    print("=== Data Science Fundamentals Demo ===\n")
    
    # Load and explore data
    df = load_sample_data()
    df = exploratory_data_analysis(df)
    
    # Create visualizations
    data_visualization(df)
    
    # Perform statistical analysis
    statistical_analysis(df)
    
    # Data preprocessing
    df_clean, df_scaled = data_preprocessing_example(df)
    
    print(f"\n=== Demo Complete ===")
    print("Check the 'data_science' folder for generated plots and analysis results.")
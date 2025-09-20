"""
Python Basics for Data Science and Machine Learning

This module covers fundamental Python concepts needed for ML/DS work.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def basic_data_types():
    """Demonstrate basic Python data types used in data science."""
    # Lists - fundamental for data collection
    numbers = [1, 2, 3, 4, 5]
    names = ['Alice', 'Bob', 'Charlie']
    
    # Dictionaries - key for data organization
    person = {
        'name': 'John',
        'age': 30,
        'city': 'New York'
    }
    
    return numbers, names, person

def numpy_basics():
    """Introduction to NumPy for numerical computing."""
    # Creating arrays
    arr1d = np.array([1, 2, 3, 4, 5])
    arr2d = np.array([[1, 2, 3], [4, 5, 6]])
    
    # Basic operations
    print(f"1D Array: {arr1d}")
    print(f"2D Array:\n{arr2d}")
    print(f"Array shape: {arr2d.shape}")
    print(f"Array sum: {arr1d.sum()}")
    print(f"Array mean: {arr1d.mean()}")
    
    return arr1d, arr2d

def pandas_basics():
    """Introduction to Pandas for data manipulation."""
    # Creating a simple dataset
    data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
        'Age': [25, 30, 35, 28],
        'City': ['New York', 'London', 'Tokyo', 'Paris'],
        'Salary': [50000, 60000, 70000, 55000]
    }
    
    df = pd.DataFrame(data)
    
    print("DataFrame:")
    print(df)
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nAverage salary: {df['Salary'].mean()}")
    
    return df

def simple_visualization():
    """Create basic plots for data visualization."""
    # Sample data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='sin(x)')
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.title('Simple Sine Wave')
    plt.legend()
    plt.grid(True)
    plt.savefig('basics/sine_wave.png')
    plt.close()
    
    print("Sine wave plot saved as 'basics/sine_wave.png'")

if __name__ == "__main__":
    print("=== Python Basics for Data Science ===\n")
    
    # Run basic examples
    numbers, names, person = basic_data_types()
    print(f"Sample numbers: {numbers}")
    print(f"Sample names: {names}")
    print(f"Sample person: {person}\n")
    
    print("=== NumPy Basics ===")
    arr1d, arr2d = numpy_basics()
    print()
    
    print("=== Pandas Basics ===")
    df = pandas_basics()
    print()
    
    print("=== Basic Visualization ===")
    simple_visualization()
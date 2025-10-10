"""
Outlier analysis and visualization for journey data.
Creates box plots to visualize outliers before and after capping.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

def create_outlier_boxplots():
    """Create box plots for outlier analysis of journey data"""
    
    # Load the processed journey data
    try:
        # Use relative path to outputs directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, '..', 'outputs', 'processed_journey_master.csv')
        df = pd.read_csv(data_path)
        print(f"Loaded journey data: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        print("Error: processed_journey_master.csv not found. Run data_integration.py first.")
        return
    
    # Define the numeric columns for outlier analysis
    numeric_cols = ['journey_travel_time', 'journey_distance', 'journey_elapsed_time']
    
    # Check which columns exist
    available_cols = [col for col in numeric_cols if col in df.columns]
    if not available_cols:
        print("Error: No numeric journey columns found in the dataset")
        return
    
    print(f"Analyzing outliers in columns: {available_cols}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, len(available_cols), figsize=(5*len(available_cols), 6))
    if len(available_cols) == 1:
        axes = [axes]
    
    # Create box plots for each variable
    for i, col in enumerate(available_cols):
        # Remove missing values
        data = pd.to_numeric(df[col], errors='coerce').dropna()
        
        # Create box plot
        axes[i].boxplot(data, patch_artist=True, 
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))
        
        axes[i].set_title(f'{col.replace("_", " ").title()}\nOutlier Distribution', 
                         fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Value')
        axes[i].grid(True, alpha=0.3)
        
        # Add statistics as text
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = ((data < lower_bound) | (data > upper_bound)).sum()
        outlier_pct = (outliers / len(data)) * 100
        
        # Add summary statistics
        stats_text = f'Outliers: {outliers} ({outlier_pct:.1f}%)\n'
        stats_text += f'Median: {data.median():.1f}\n'
        stats_text += f'Q1: {q1:.1f}, Q3: {q3:.1f}\n'
        stats_text += f'Upper bound: {upper_bound:.1f}'
        
        axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='white', alpha=0.8), fontsize=9)
        
        print(f"\n{col} outlier analysis:")
        print(f"  Outliers: {outliers} ({outlier_pct:.2f}%)")
        print(f"  Median: {data.median():.2f}")
        print(f"  IQR bounds: {lower_bound:.2f} to {upper_bound:.2f}")
        print(f"  99th percentile: {data.quantile(0.99):.2f}")
    
    plt.tight_layout()
    plt.suptitle('Journey Data Outlier Analysis', fontsize=14, fontweight='bold', y=1.02)
    
    # Save the plot
    output_path = os.path.join(script_dir, '..', 'outputs', 'outlier_boxplots.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nBox plots saved to: {output_path}")
    
    # Show the plot
    plt.show()
    
    return fig

def create_before_after_comparison():
    """Create comparison of data before and after outlier capping"""
    
    # This would require the original uncapped data
    # For now, we'll note this as a future enhancement
    print("\nNote: To create before/after comparison plots, we would need to:")
    print("1. Save the original data before capping in data_integration.py")
    print("2. Load both original and capped versions here")
    print("3. Create side-by-side box plots showing the difference")

if __name__ == "__main__":
    print("="*70)
    print("OUTLIER ANALYSIS AND VISUALIZATION")
    print("="*70)
    
    # Create box plots
    create_outlier_boxplots()
    
    # Note about before/after comparison
    create_before_after_comparison()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
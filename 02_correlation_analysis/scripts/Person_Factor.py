import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import math
import os

# This script analyzes two person-related factors (occupation and age group) 
# and calculates their Normalized Mutual Information (NMI) with WFH adoption (Yes/No). 
# It then identifies which factor shows the strongest association with WFH adoption.


# Get current script directory and define output directory path
script_dir = os.path.dirname(os.path.abspath(__file__))  
output_dir = Path(os.path.join(script_dir, "../outputs")) 
# Create output directory if it doesn't exist
output_dir.mkdir(exist_ok=True)

# Define data file path and load data
data_path = os.path.abspath(os.path.join(script_dir, "../../01_preprocessing/outputs/processed_person_master_readable.csv"))
try:
    df = pd.read_csv(data_path)
    print(f"Loaded person data successfully: {df.shape[0]} rows, {df.shape[1]} columns")
except FileNotFoundError:
    print(f"Error: File not found at {data_path}. Run preprocessing pipeline first.")
    exit(1)
except Exception as e:
    print(f"Error loading person data: {e}")
    exit(1)

# Filter valid samples: only "Yes/No" for works_from_home_any, and non-null analysis_weight
valid_df = df[
    (df["works_from_home_any"].isin(["Yes", "No"])) 
].copy()

total_overall_weight = valid_df["analysis_weight"].sum()

# Define function to calculate weighted NMI and WFH adoption rate
def calculate_weighted_nmi(df, var_name, target_var="works_from_home_any", weight_col="analysis_weight"):
    # Create weighted contingency table using analysis_weight
    weighted_ct = pd.crosstab(
        df[var_name], df[target_var],
        values=df[weight_col], aggfunc="sum"
    ).fillna(0)  # Fill 0 for categories with no samples
    
    class_total_weight = df.groupby(var_name)[weight_col].sum()
    total_weight_all = class_total_weight.sum()
    class_weight_pct = (class_total_weight / total_weight_all * 100).round(2) if total_weight_all > 0 else pd.Series(0, index=class_total_weight.index)
    
    # Calculate joint probability
    total_weight = weighted_ct.sum().sum()
    joint_prob = weighted_ct / total_weight if total_weight > 0 else weighted_ct * 0
    
    # Calculate marginal probabilities for the variable and target
    marginal_var = joint_prob.sum(axis=1)
    marginal_wfh = joint_prob.sum(axis=0)
    
    # Calculate weighted Mutual Information (MI)
    # I acknowledge that the formulas 
    # and calculation methods in this section were developed with the assistance of AI suggestions.
    mi = 0.0
    for feature_value in joint_prob.index:
        for wfh_value in joint_prob.columns:
            p_xy = joint_prob.loc[feature_value, wfh_value]
            if p_xy > 0:  # Avoid log(0)
                p_x = marginal_var[feature_value]
                p_y = marginal_wfh[wfh_value]
                if p_x > 0 and p_y > 0:  # Avoid division by zero
                    mi += p_xy * math.log2(p_xy / (p_x * p_y))
    
    # Calculate weighted Entropy
    h_var = -sum(p * math.log2(p) for p in marginal_var if p > 0)
    h_wfh = -sum(p * math.log2(p) for p in marginal_wfh if p > 0)
    
    # Calculate Normalized Mutual Information (NMI)
    nmi_score = mi / min(h_var, h_wfh) if (h_var > 0 and h_wfh > 0) else 0.0
    
    # Calculate WFH adoption rate (%) for each category
    adopt_rate = (weighted_ct["Yes"] / weighted_ct.sum(axis=1) * 100).dropna()
    
    return nmi_score, weighted_ct, adopt_rate, class_weight_pct

# Calculate NMI and adoption rate for occupation and age group
occ_nmi, occ_weighted_ct, occ_adopt_rate, occ_weight_pct = calculate_weighted_nmi(
    df=valid_df, var_name="occupation_major_group"
)
age_nmi, age_weighted_ct, age_adopt_rate, age_weight_pct = calculate_weighted_nmi(
    df=valid_df, var_name="age_group"
)

# Create combined plot (2 rows, 1 column)
plt.figure(figsize=(14, 12))

# Subplot 1: WFH Adoption Rate by Occupation Major Group
plt.subplot(2, 1, 1)
ax1 = sns.barplot(x=occ_adopt_rate.index, y=occ_adopt_rate.values, palette="Set2")
plt.title("WFH Adoption Rate by Occupation Major Group (%)", fontsize=12, pad=15)
plt.xlabel("Occupation Major Group", fontsize=10)
plt.ylabel("Adoption Rate (%)", fontsize=10)
plt.xticks(rotation=45, ha="right", fontsize=9)  # Rotate labels to avoid overlap
# Add value lables on bars
# I acknowledge that the addition of the weight labels was done with reference to AI.
for i, (adopt_val, weight_val) in enumerate(zip(occ_adopt_rate.values, occ_weight_pct.values)):
    ax1.text(i, adopt_val + 0.5, f"Adopt: {adopt_val:.1f}%", ha="center", va="bottom", fontsize=9, color="darkred")  # WFH rate
    ax1.text(i, adopt_val + 5, f"W: {weight_val:.1f}%", ha="center", va="bottom", fontsize=8, color="darkblue")   # Weight

# Add NMI info text box
ax1.text(0.02, 0.95, f"Weighted NMI: {occ_nmi:.2f}", 
         transform=ax1.transAxes, verticalalignment="top",
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), fontsize=10)

# Subplot 2: WFH Adoption Rate by Age Group
plt.subplot(2, 1, 2)
ax2 = sns.barplot(x=age_adopt_rate.index, y=age_adopt_rate.values, palette="Set3")
plt.title("WFH Adoption Rate by Age Group (%)", fontsize=12, pad=15)
plt.xlabel("Age Group", fontsize=10)
plt.ylabel("Adoption Rate (%)", fontsize=10)
plt.xticks(rotation=45, ha="right", fontsize=9)  # Rotate labels to avoid overlap

# Add value labels on bars
# I acknowledge that the addition of the weight labels was done with reference to AI.
for i, (adopt_val, weight_val) in enumerate(zip(age_adopt_rate.values, age_weight_pct.values)):
    ax2.text(i, adopt_val + 0.5, f"Adopt: {adopt_val:.1f}%", ha="center", va="bottom", fontsize=9, color="darkred")  # WFH rate
    ax2.text(i, adopt_val + 5, f"W: {weight_val:.1f}%", ha="center", va="bottom", fontsize=8, color="darkblue")   # Weight

# Determine variable with stronger association
stronger_var = "Occupation Major Group" if occ_nmi > age_nmi else "Age Group"
association_conclusion = f"Stronger Association with WFH: {stronger_var}" 

# Add NMI info and stronger association text box
ax2.text(0.02, 0.95, f"Weighted NMI: {age_nmi:.2f}", 
         transform=ax2.transAxes, verticalalignment="top",
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), fontsize=10)

# Adjust layout to prevent label cutoff
plt.tight_layout()
plt.suptitle(association_conclusion, fontsize=12, y=0.98, ha='left', x=0.02)

# Save plot and close
save_path = output_dir / "Age_vs_Occupation.png"
plt.savefig(save_path, dpi=300)
plt.close()

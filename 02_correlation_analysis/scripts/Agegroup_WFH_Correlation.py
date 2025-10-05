import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import math

# Create folder for saving outputs
output_dir = Path("../outputs")
output_dir.mkdir(exist_ok=True)

# Read processed CSV data
data_path = "../../01_preprocessing/outputs/processed_person_master.csv"
df = pd.read_csv(data_path)

# Keep rows with valid WFH intensity, age group, and weights
valid_df = df[
    (df["wfh_intensity_total"].between(0, 7)) &  
    (df["wfh_intensity_total"].notna()) &        
    (df["agegroup"].notna()) &
    (df["perspoststratweight"].notna())
].copy()

# Crosstab with age group as rows, WFH intensity as columns, values weighted by perspoststratweight
weighted_table = pd.crosstab(
    valid_df["agegroup"], 
    valid_df["wfh_intensity_total"],  
    values=valid_df["perspoststratweight"],
    aggfunc="sum" 
).fillna(0)  

# Joint probability P(agegroup, WFH) and marginals P(agegroup), P(WFH)
joint_prob = weighted_table / weighted_table.sum().sum()
px = joint_prob.sum(axis=1)  
py = joint_prob.sum(axis=0)  

# Compute weighted mutual information
mi = 0.0
for agegroup in joint_prob.index:          
    for wfh_int in joint_prob.columns:      
        p_xy = joint_prob.loc[agegroup, wfh_int]
        if p_xy > 0:
            mi += p_xy * math.log2(p_xy / (px[agegroup] * py[wfh_int]))

# Compute entropy and NMI
def calculate_entropy(prob_series):
    return -sum(p * math.log2(p) for p in prob_series if p > 0)

hx = calculate_entropy(px)
hy = calculate_entropy(py)
nmi_score = mi / min(hx, hy) if (hx > 0 and hy > 0) else 0.0

plt.figure(figsize=(12, 8))

# Convert to percentages within age group and create string labels with '%'
plot_data = weighted_table.div(weighted_table.sum(axis=1), axis=0) * 100
annot_labels = plot_data.round(1).astype(str) + "%"

# Plot heatmap
sns.heatmap(
    plot_data, 
    annot=annot_labels,        
    fmt="",         
    cmap="YlGnBu",     
    cbar_kws={"label": "Percentage within Age Group (%)"},  
    xticklabels=True,
    yticklabels=True
)

# Add labels and NMI annotation
plt.title("Weighted Distribution of WFH Intensity (0-7) by Age Group", fontsize=14)
plt.xlabel("WFH Intensity (wfh_intensity_total)", fontsize=12)
plt.ylabel("Age Group", fontsize=12)
plt.xticks(rotation=0)  
plt.yticks(rotation=0)


stat_text = f"NMI: {nmi_score:.4f}, no strong association between Age Group and WFH Intensity."
plt.text(
    0, 0.07, 
    stat_text, 
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)



# Save figure
plt.tight_layout()
plt.savefig(output_dir / "age_group_vs_wfh_intensity_nmi_original.png", dpi=300)
plt.close()

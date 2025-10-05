" "
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import normalized_mutual_info_score
import math

# This section investigates the statistical relationship between occupation type 
# (a demographic characteristic) and WFH adoption (i.e., whether individuals work 
# from home or not). The analysis is conducted using the Chi-square test method 
# to examine the association between these categorical variables.

# Create the output directory if it doesn't already exist
output_dir = Path("../outputs")
output_dir.mkdir(exist_ok=True)

# Load the dataset and keep only valid samples (respondents who answered "Yes" or "No" to WFH adoption)
data_path = "../../01_preprocessing/outputs/processed_person_master.csv"
df = pd.read_csv(data_path)
valid_df = df[df["anywfh"].isin(["Yes", "No"])]


# Weighted contingency table: occupation (anzsco1) vs WFH status (anywfh) using post-stratification weights
weighted_adoption = pd.crosstab(
    valid_df["anzsco1"], valid_df["anywfh"], 
    values=valid_df["perspoststratweight"], aggfunc="sum"
).fillna(0)

# Normalize the weighted contingency table to get probabilities summing to 1
joint = weighted_adoption / weighted_adoption.sum().sum()  

# Calculate marginal probabilities
px = joint.sum(axis=1)  
py = joint.sum(axis=0)  

# Calculate Mutual Information (MI)
mi = 0.0
for i in joint.index:
    for j in joint.columns:
        pxy = joint.loc[i, j]
        if pxy > 0:
            mi += pxy * math.log2(pxy / (px[i] * py[j]))

# Calculate marginal entropies
hx = -sum(p * math.log2(p) for p in px if p > 0)
hy = -sum(p * math.log2(p) for p in py if p > 0)

# Calculate Normalized Mutual Information (NMI)
nmi_score = mi / min(hx, hy) if (hx > 0 and hy > 0) else 0.0

# Create a figure with a specific size
plt.figure(figsize=(14,6))

# Calculate WFH adoption rate (%)
adopt_rate = (weighted_adoption["Yes"] / weighted_adoption.sum(axis=1) * 100).dropna()

# Plot bar chart
ax = sns.barplot(x=adopt_rate.index, y=adopt_rate.values, palette="Set2")
plt.title("WFH Adoption Rate by Occupation (%)")
plt.xlabel("Occupation"); plt.ylabel("Rate (%)")
plt.xticks(rotation=45, ha="right")

# Add value labels inside bars
for i, v in enumerate(adopt_rate.values):
    ax.text(i, v / 2, f"{v:.1f}%", ha='center', va='bottom', fontsize=10)

# Add stats text
stat_text = (
    f"NMI: {nmi_score:.2f}"  
)
plt.text(0.02, 0.98, stat_text, transform=plt.gca().transAxes, 
         verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# Adjust layout, save, and close
plt.tight_layout()
plt.savefig(output_dir / "adoption_and_intensity.png", dpi=300)
plt.close()


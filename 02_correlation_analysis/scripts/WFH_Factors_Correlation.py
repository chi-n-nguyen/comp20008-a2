" "
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# This section investigates the statistical relationship between occupation type 
# (a demographic characteristic) and WFH adoption (i.e., whether individuals work 
# from home or not). The analysis is conducted using the Chi-square test method 
# to examine the association between these categorical variables.

# Create the output directory if it doesn't already exist
output_dir = Path("/Users/luchang/Desktop/eodp_asmt2/comp20008-a2/02_correlation_analysis/outputs")
output_dir.mkdir(exist_ok=True)

# Load the dataset and keep only valid samples (respondents who answered "Yes" or "No" to WFH adoption)
data_path = "/Users/luchang/Desktop/eodp_asmt2/comp20008-a2/01_preprocessing/outputs/processed_person_master.csv"
df = pd.read_csv(data_path)
valid_df = df[df["anywfh"].isin(["Yes", "No"])]


# Weighted contingency table: occupation (anzsco1) vs WFH status (anywfh) using post-stratification weights
weighted_adoption = pd.crosstab(
    valid_df["anzsco1"], valid_df["anywfh"], 
    values=valid_df["perspoststratweight"], aggfunc="sum"
).fillna(0)

# Chi-square test: check if occupation and WFH are significantly associated
chi2, p_adopt, dof, _ = stats.chi2_contingency(weighted_adoption)

# Cramer's V: measure strength of association (0-1)
cramers_v = np.sqrt(chi2 / (weighted_adoption.sum().sum() * min(weighted_adoption.shape[0]-1, weighted_adoption.shape[1]-1)))

# Create a figure with a specific size
plt.figure(figsize=(14,6))

# Calculate WFH adoption rate (%)
adopt_rate = (weighted_adoption["Yes"] / weighted_adoption.sum(axis=1) * 100).dropna()

# Plot bar chart
sns.barplot(x=adopt_rate.index, y=adopt_rate.values, palette="Set2")
plt.title("WFH Adoption Rate by Occupation (%)")
plt.xlabel("Occupation"); plt.ylabel("Rate (%)")
plt.xticks(rotation=45, ha="right")

# Add value labels inside bars
ax = sns.barplot(x=adopt_rate.index, y=adopt_rate.values, palette="Set2")
for i, v in enumerate(adopt_rate.values):
    ax.text(i, v / 2, f"{v:.1f}%", ha='center', va='bottom', fontsize=10)

# Add stats text
stat_text = f"Chi2: {chi2:.2f}\nP-value: {p_adopt:.3f}\nCramer's V: {cramers_v:.2f}"
plt.text(0.02, 0.98, stat_text, transform=plt.gca().transAxes, 
         verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# Adjust layout, save, and close
plt.tight_layout()
plt.savefig(output_dir / "adoption_and_intensity.png", dpi=300)
plt.close()


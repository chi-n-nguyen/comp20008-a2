import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import math
import os

# Path settings
script_dir = os.path.dirname(os.path.abspath(__file__))  
output_dir = Path(os.path.join(script_dir, "../outputs")) 
output_dir.mkdir(exist_ok=True)

# Data loading and filtering 
data_path = os.path.abspath(os.path.join(script_dir, "../../01_preprocessing/outputs/processed_journey_master.csv"))
df = pd.read_csv(data_path)

valid_df = df[
    (df["end_loc"] == "TO_WORK") &
    (df["wfh_adopter"].isin([0, 1])) &
    (df["journey_travel_time"] >= 0)
].copy()

# Discretizes journey travel time into categorical intervals
def discretize_travel_time(time):
    if time <= 15:
        return "0-15 min (Short)"
    elif time <= 30:
        return "15-30 min (Medium-Short)"
    elif time <= 60:
        return "30-60 min (Medium-Long)"
    else:
        return ">60 min (Long)"
valid_df["travel_time_category"] = valid_df["journey_travel_time"].apply(discretize_travel_time)

# Weighted NMI calculation function (with class weight) 
def calculate_weighted_nmi_and_weights(df, var_name, target_var="wfh_adopter", weight_col="analysis_weight"):
    
    # Calculate weighted cross table (for WFH adopter rate)
    weighted_ct = pd.crosstab(
        df[var_name], df[target_var],
        values=df[weight_col], aggfunc="sum"
    ).fillna(0)
    
    # Calculate total weight per class (for weight percentage)
    class_total_weight = df.groupby(var_name)[weight_col].sum()  
    total_weight_all = class_total_weight.sum()  
    class_weight_pct = (class_total_weight / total_weight_all * 100).round(2)  
    
    # Calculate WFH adopter rate (weighted)
    adopter_rate = (weighted_ct[1] / weighted_ct.sum(axis=1) * 100).round(2)  
    
    # Calculate NMI
    # I acknowledge that the formulas 
    # and calculation methods in this section were developed with the assistance of AI suggestions.
    joint_prob = weighted_ct / weighted_ct.sum().sum() if weighted_ct.sum().sum() != 0 else weighted_ct
    marginal_feature = joint_prob.sum(axis=1)
    marginal_wfh = joint_prob.sum(axis=0)
    
    mi = 0.0
    for feature_value in joint_prob.index:
        for wfh_value in joint_prob.columns:
            p_xy = joint_prob.loc[feature_value, wfh_value]
            if p_xy > 0:
                p_x = marginal_feature[feature_value]
                p_y = marginal_wfh[wfh_value]
                mi += p_xy * math.log2(p_xy / (p_x * p_y))
    
    h_var = -sum(p * math.log2(p) for p in marginal_feature if p > 0)
    h_wfh = -sum(p * math.log2(p) for p in marginal_wfh if p > 0)
    nmi_score = mi / min(h_var, h_wfh) if (h_var > 0 and h_wfh > 0) else 0.0
    
    return nmi_score, weighted_ct, adopter_rate, class_weight_pct

# Calculate NMI + class weight for 3 core variables 
# Main Journey Mode
mode_nmi, mode_ct, mode_adopt, mode_weight_pct = calculate_weighted_nmi_and_weights(
    df=valid_df, var_name="main_journey_mode"
)

# Journey Purpose
purp_nmi, purp_ct, purp_adopt, purp_weight_pct = calculate_weighted_nmi_and_weights(
    df=valid_df, var_name="destpurp1_desc_01"
)

# Journey Travel Time
time_nmi, time_ct, time_adopt, time_weight_pct = calculate_weighted_nmi_and_weights(
    df=valid_df, var_name="travel_time_category"
)

# Visualization (with weight percentage labels) 
plt.figure(figsize=(14, 20))  # Adjust size for clear labels

# Subplot 1: Main Journey Mode
plt.subplot(3, 1, 1)
ax1 = sns.barplot(x=mode_adopt.index, y=mode_adopt.values, palette="Set2", legend=False)
plt.title("WFH Adopter Rate by Main Journey Mode (End Loc=TO_WORK)", fontsize=13, pad=15)
plt.xlabel("Main Journey Mode", fontsize=11)
plt.ylabel("WFH Adopter Rate (%)", fontsize=11)
plt.xticks(rotation=45, ha="right", fontsize=10)
# Add 2 labels: WFH rate (top) + weight % (bottom)
for i, (adopt_val, weight_val) in enumerate(zip(mode_adopt.values, mode_weight_pct.values)):
    # WFH adopter rate (top of bar)
    ax1.text(i, adopt_val + 0.5, f"Adopt: {adopt_val}%", ha="center", va="bottom", fontsize=9, color="darkred")
    # Weight percentage (bottom of bar, near x-axis)
    ax1.text(i, adopt_val + 5, f"Weight: {weight_val}%", ha="center", va="bottom", fontsize=9, color="darkblue")
ax1.text(0.02, 0.8, f"Weighted NMI: {mode_nmi:.2f}", transform=ax1.transAxes, 
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), fontsize=11)

# Subplot 2: Journey Purpose
plt.subplot(3, 1, 2)
ax2 = sns.barplot(x=purp_adopt.index, y=purp_adopt.values, palette="Set3", legend=False)
plt.title("WFH Adopter Rate by Journey Purpose (End Loc=TO_WORK)", fontsize=13, pad=15)
plt.xlabel("Journey Purpose (destpurp1_desc_01)", fontsize=11)
plt.ylabel("WFH Adopter Rate (%)", fontsize=11)
plt.xticks(rotation=45, ha="right", fontsize=10)

# Add 2 labels
# I acknowledge that the addition of the weight labels was done with reference to AI.
for i, (adopt_val, weight_val) in enumerate(zip(purp_adopt.values, purp_weight_pct.values)):
    ax2.text(i, adopt_val + 0.5, f"Adopt: {adopt_val}%", ha="center", va="bottom", fontsize=9, color="darkred")
    ax2.text(i, adopt_val + 5, f"Weight: {weight_val}%", ha="center", va="bottom", fontsize=9, color="darkblue")
ax2.text(0.02, 0.8, f"Weighted NMI: {purp_nmi:.2f}", transform=ax2.transAxes, 
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), fontsize=11)

# Subplot 3: Journey Travel Time
plt.subplot(3, 1, 3)
ax3 = sns.barplot(x=time_adopt.index, y=time_adopt.values, palette="Oranges", legend=False)
plt.title("WFH Adopter Rate by Journey Travel Time (End Loc=TO_WORK)", fontsize=13, pad=15)
plt.xlabel("Journey Travel Time (Discretized)", fontsize=11)
plt.ylabel("WFH Adopter Rate (%)", fontsize=11)
plt.xticks(rotation=30, ha="right", fontsize=10)

# Add 2 labels
# I acknowledge that the addition of the weight labels was done with reference to AI.
for i, (adopt_val, weight_val) in enumerate(zip(time_adopt.values, time_weight_pct.values)):
    ax3.text(i, adopt_val + 0.5, f"Adopt: {adopt_val}%", ha="center", va="bottom", fontsize=9, color="darkred")
    ax3.text(i, adopt_val + 5, f"Weight: {weight_val}%", ha="center", va="bottom", fontsize=9, color="darkblue")
ax3.text(0.02, 0.8, f"Weighted NMI: {time_nmi:.2f}", transform=ax3.transAxes, 
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), fontsize=11)

# Association Conclusion 
nmi_dict = {
    "Main Journey Mode": mode_nmi,
    "Journey Purpose": purp_nmi,
    "Journey Travel Time": time_nmi
}
strongest_var = max(nmi_dict, key=nmi_dict.get)
association_conclusion = (
    f"Strongest Association with WFH Adopter: {strongest_var}\n"
    f"NMI Comparison: Mode={mode_nmi:.2f}, Purpose={purp_nmi:.2f}, Time={time_nmi:.2f}"
)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.suptitle(association_conclusion, fontsize=13, y=0.98, ha='left', x=0.02,
             bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8), fontweight="bold")

# Save plot
save_path = output_dir / "Journey_Mode_Purpose_Time.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()
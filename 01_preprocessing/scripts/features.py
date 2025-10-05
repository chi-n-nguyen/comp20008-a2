"""Feature engineering utilities for WFH analysis.

Functions:
- create_wfh_features: builds WFH intensity, adoption, categories, and worker flag
- create_household_wfh_metrics: aggregates person WFH to household metrics
"""

import pandas as pd
import config

def create_wfh_features(df):
    df_features = df.copy()
    existing_weekdays = [c for c in config.WFH_WEEKDAYS if c in df_features.columns]
    existing_weekends = [c for c in config.WFH_WEEKENDS if c in df_features.columns]
    existing_all_days = [c for c in config.WFH_ALL_DAYS if c in df_features.columns]

    # Process all WFH day columns
    for col in existing_all_days:
        df_features[col] = df_features[col].map(config.WFH_MAPPING)
        df_features[col] = pd.to_numeric(df_features[col], errors='coerce').fillna(0)

    # Calculate different intensity metrics
    if existing_weekdays:
        df_features['wfh_intensity_weekdays'] = df_features[existing_weekdays].sum(axis=1)
    else:
        df_features['wfh_intensity_weekdays'] = 0
    
    if existing_weekends:
        df_features['wfh_intensity_weekends'] = df_features[existing_weekends].sum(axis=1)
    else:
        df_features['wfh_intensity_weekends'] = 0
    
    if existing_all_days:
        df_features['wfh_intensity_total'] = df_features[existing_all_days].sum(axis=1)
    else:
        df_features['wfh_intensity_total'] = 0
    
    # Keep original wfh_intensity as weekdays for backward compatibility
    df_features['wfh_intensity'] = df_features['wfh_intensity_weekdays']

    df_features['wfh_adopter'] = (df_features['wfh_intensity_total'] > 0).astype(int)

    def categorize_wfh(score):
        if score == 0:
            return 'No_WFH'
        elif score <= 2:
            return 'Occasional_WFH'
        elif score <= 4:
            return 'Frequent_WFH'
        return 'Full_WFH'

    # Use total intensity for categorization (including weekends)
    df_features['wfh_category'] = df_features['wfh_intensity_total'].apply(categorize_wfh)

    if 'anywork' in df_features.columns:
        df_features['is_worker'] = (df_features['anywork'] == 'Yes').astype(int)
    else:
        df_features['is_worker'] = 1

    return df_features

def create_household_wfh_metrics(persons_df, households_df):
    hh_wfh = persons_df.groupby('hhid').agg({
        'wfh_adopter': ['sum', 'mean'],
        'wfh_intensity': ['mean', 'max'],
        'is_worker': 'sum',
        config.PERSON_WEIGHT: 'first'
    }).reset_index()

    hh_wfh.columns = ['hhid', 'total_wfh_adopters', 'prop_wfh_adopters',
                      'avg_wfh_intensity', 'max_wfh_intensity',
                      'total_workers', 'hhweight']

    hh_wfh['has_worker'] = (hh_wfh['total_workers'] > 0).astype(int)
    hh_wfh['hh_wfh_saturation'] = hh_wfh.apply(
        lambda r: (r['total_wfh_adopters'] / r['total_workers']) if r['total_workers'] > 0 else 0.0,
        axis=1
    )

    households_enhanced = households_df.merge(hh_wfh, on='hhid', how='left')
    return households_enhanced



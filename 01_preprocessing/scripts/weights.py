"""Weight utilities for person and journey datasets.

Functions:
- apply_weights_to_person_data: set person-level `weight`
- set_journey_analysis_weight: set `analysis_weight` and `weight_source`
"""

import pandas as pd
import config

def apply_weights_to_person_data(df):
    df_weighted = df.copy()
    if config.PERSON_WEIGHT in df_weighted.columns:
        df_weighted['weight'] = df_weighted[config.PERSON_WEIGHT]
    else:
        df_weighted['weight'] = 1.0
    return df_weighted

def set_journey_analysis_weight(df):
    if config.JOURNEY_WEIGHT in df.columns:
        df['analysis_weight'] = df[config.JOURNEY_WEIGHT]
        df['weight_source'] = config.JOURNEY_WEIGHT
    else:
        df['analysis_weight'] = df[config.PERSON_WEIGHT]
        df['weight_source'] = config.PERSON_WEIGHT
    return df

def set_trip_analysis_weight(df):
    """Set analysis weight for trip-level data using trip weights"""
    if 'trippoststratweight' in df.columns:
        df['analysis_weight'] = df['trippoststratweight']
        df['weight_source'] = 'trippoststratweight'
    elif config.PERSON_WEIGHT in df.columns:
        df['analysis_weight'] = df[config.PERSON_WEIGHT]
        df['weight_source'] = config.PERSON_WEIGHT
    else:
        df['analysis_weight'] = 1.0
        df['weight_source'] = 'unweighted'
    return df

def set_stop_analysis_weight(df):
    """Set analysis weight for stop-level data using stop weights"""
    if 'stoppoststratweight' in df.columns:
        df['analysis_weight'] = df['stoppoststratweight']
        df['weight_source'] = 'stoppoststratweight'
    elif config.PERSON_WEIGHT in df.columns:
        df['analysis_weight'] = df[config.PERSON_WEIGHT]
        df['weight_source'] = config.PERSON_WEIGHT
    else:
        df['analysis_weight'] = 1.0
        df['weight_source'] = 'unweighted'
    return df



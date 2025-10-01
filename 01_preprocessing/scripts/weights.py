"""Weight utilities for person and journey datasets.

Functions:
- apply_weights_to_person_data: set person-level `weight`
- set_journey_analysis_weight: set `analysis_weight` and `weight_source`
"""

import pandas as pd
import config

def apply_weights_to_person_data(df: pd.DataFrame) -> pd.DataFrame:
    df_weighted = df.copy()
    if config.PERSON_WEIGHT in df_weighted.columns:
        df_weighted['weight'] = df_weighted[config.PERSON_WEIGHT]
    else:
        df_weighted['weight'] = 1.0
    return df_weighted

def set_journey_analysis_weight(df: pd.DataFrame) -> pd.DataFrame:
    if config.JOURNEY_WEIGHT in df.columns:
        df['analysis_weight'] = df[config.JOURNEY_WEIGHT]
        df['weight_source'] = config.JOURNEY_WEIGHT
    else:
        df['analysis_weight'] = df[config.PERSON_WEIGHT]
        df['weight_source'] = config.PERSON_WEIGHT
    return df



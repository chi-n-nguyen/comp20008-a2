"""Aggregation and dataset construction utilities.

Functions:
- analyze_travel_start_times: morning trip categorization joined to persons
- create_person_level_dataset: person+household merged dataset for ML
- create_household_level_dataset: household+trip aggregates for clustering
- create_journey_level_dataset: work journeys with person characteristics
"""

import pandas as pd

def analyze_travel_start_times(persons_df, trips_df):
    person_cols = ['persid', 'hhid', 'wfh_intensity', 'wfh_adopter', 'wfh_category', 'perspoststratweight']
    person_cols = [c for c in person_cols if c in persons_df.columns]
    trips_with_person = trips_df.merge(persons_df[person_cols], on='persid', how='left')
    if 'starthour' not in trips_with_person.columns:
        return pd.DataFrame()
    trips_with_person['starthour'] = pd.to_numeric(trips_with_person['starthour'], errors='coerce')
    morning = trips_with_person[trips_with_person['starthour'] < 10].copy()
    def categorize_start_time(hour):
        if pd.isna(hour):
            return 'Unknown'
        if hour < 6:
            return 'Very_Early'
        if hour < 7:
            return 'Early'
        if hour < 8:
            return 'Peak_Early'
        if hour < 9:
            return 'Peak_Late'
        return 'Late'
    morning['start_time_category'] = morning['starthour'].apply(categorize_start_time)
    morning['weight'] = morning['perspoststratweight']
    return morning

def create_person_level_dataset(persons_df, households_df):
    hh_cols = ['hhid', 'hhsize', 'dwelltype', 'totalvehs', 'hhinc_group', 'homelga', 'homeregion_ASGS']
    if 'hhpoststratweight' in households_df.columns:
        hh_cols.append('hhpoststratweight')
    hh_cols = [c for c in hh_cols if c in households_df.columns]
    person_analysis = persons_df.merge(households_df[hh_cols], on='hhid', how='left')
    
    # Apply worker filter with explicit reporting
    if 'is_worker' in person_analysis.columns:
        print(f"\nFiltering to workers only:")
        print(f"  Before filter: {len(person_analysis):,} records")
        person_analysis = person_analysis[person_analysis['is_worker'] == 1].copy()
        print(f"  After filter: {len(person_analysis):,} records")
        print(f"  Non-workers excluded: {len(persons_df) - len(person_analysis):,} records")
    
    person_analysis['analysis_weight'] = person_analysis['perspoststratweight']
    return person_analysis

def create_household_level_dataset(households_enhanced, trips_df):
    trip_agg_cols = {}
    if 'tripno' in trips_df.columns:
        trip_agg_cols['tripno'] = 'count'
    if 'duration' in trips_df.columns:
        trip_agg_cols['duration'] = 'mean'
    if 'starthour' in trips_df.columns:
        trip_agg_cols['starthour'] = 'mean'
    if trip_agg_cols:
        trip_stats = trips_df.groupby('hhid').agg(trip_agg_cols).reset_index()
        new_names = ['hhid']
        if 'tripno' in trip_agg_cols:
            new_names.append('total_trips')
        if 'duration' in trip_agg_cols:
            new_names.append('avg_trip_duration')
        if 'starthour' in trip_agg_cols:
            new_names.append('avg_start_hour')
        trip_stats.columns = new_names
        hh_master = households_enhanced.merge(trip_stats, on='hhid', how='left')
    else:
        hh_master = households_enhanced.copy()
    if 'hhpoststratweight' in hh_master.columns:
        hh_master['analysis_weight'] = hh_master['hhpoststratweight']
    else:
        hh_master['analysis_weight'] = 1.0
    return hh_master

def create_journey_level_dataset(journey_work_df, persons_df):
    person_cols = ['persid', 'wfh_intensity', 'wfh_adopter', 'wfh_category', 'perspoststratweight']
    optional = ['agegroup', 'sex', 'carlicence', 'emptype', 'persinc']
    for col in optional:
        if col in persons_df.columns:
            person_cols.append(col)
    journey_analysis = journey_work_df.merge(persons_df[person_cols], on='persid', how='left')
    return journey_analysis



"""Variable mapping utilities for cleaner feature names.

Creates human-readable column names for machine learning models,
replacing cryptic VISTA variable names with descriptive alternatives.
"""

import pandas as pd

# Mapping dictionary for complex variable names
VARIABLE_MAPPING = {
    # Survey weights (keep original but add clarity)
    'perspoststratweight': 'person_survey_weight',
    'hhpoststratweight': 'household_survey_weight',
    'analysis_weight': 'analysis_weight',  # Keep as is - already clear
    
    # Weight groups (demographic stratification)
    'perspoststratweight_GROUP_1': 'weight_demo_group_1',
    'perspoststratweight_GROUP_2': 'weight_demo_group_2', 
    'perspoststratweight_GROUP_3': 'weight_demo_group_3',
    'perspoststratweight_GROUP_4': 'weight_demo_group_4',
    'perspoststratweight_GROUP_5': 'weight_demo_group_5',
    'perspoststratweight_GROUP_6': 'weight_demo_group_6',
    'perspoststratweight_GROUP_7': 'weight_demo_group_7',
    'perspoststratweight_GROUP_8': 'weight_demo_group_8',
    'perspoststratweight_GROUP_9': 'weight_demo_group_9',
    'perspoststratweight_GROUP_10': 'weight_demo_group_10',
    
    # Occupation classifications
    'anzsco1': 'occupation_major_group',  # 1-digit ANZSCO
    'anzsco2': 'occupation_minor_group',  # 2-digit ANZSCO
    'anzsic1': 'industry_division',      # 1-digit ANZSIC
    'anzsic2': 'industry_subdivision',   # 2-digit ANZSIC
    
    # Geographic variables
    'homeregion_ASGS': 'home_region',
    'homesubregion_ASGS': 'home_subregion', 
    'homelga': 'home_local_gov_area',
    
    # Age groups
    'agegroup': 'age_group',
    'youngestgroup_5': 'youngest_person_age_group',
    'aveagegroup_5': 'average_household_age_group',
    'oldestgroup_5': 'oldest_person_age_group',
    
    # Household characteristics
    'hhsize': 'household_size',
    'totalvehs': 'vehicles_owned',
    'totalbikes': 'bicycles_owned',
    'dwelltype': 'dwelling_type',
    'owndwell': 'owns_dwelling',
    'hhinc_group': 'household_income_group',
    
    # Personal characteristics
    'persinc': 'personal_income',
    'emptype': 'employment_type',
    'mainact': 'main_activity',
    'carlicence': 'has_car_license',
    'mbikelicence': 'has_motorbike_license',
    'otherlicence': 'has_other_license',
    'nolicence': 'has_no_license',
    
    # Work patterns
    'fulltimework': 'works_full_time',
    'parttimework': 'works_part_time', 
    'casualwork': 'works_casual',
    'anywork': 'has_any_work',
    'anywfh': 'works_from_home_any',
    
    # WFH by day
    'wfhmon': 'wfh_monday',
    'wfhtue': 'wfh_tuesday',
    'wfhwed': 'wfh_wednesday', 
    'wfhthu': 'wfh_thursday',
    'wfhfri': 'wfh_friday',
    'wfhsat': 'wfh_saturday',
    'wfhsun': 'wfh_sunday',
    'wfhtravday': 'wfh_on_travel_day',
    
    # Trip characteristics
    'travdow': 'travel_day_of_week',
    'numstops': 'number_of_stops',
    'startplace': 'trip_start_place',
    'faretype': 'public_transport_fare_type',
    'anytoll': 'paid_road_tolls',
    'anyvehwalk': 'used_vehicle_or_walk',
    'anypaidpark': 'paid_for_parking',
    
    # Journey specifics
    'journey_travel_time': 'commute_time_minutes',
    'journey_distance': 'commute_distance_km',
    'journey_elapsed_time': 'total_journey_time',
    'main_journey_mode': 'primary_transport_mode',
    
    # Household aggregates
    'total_wfh_adopters': 'num_wfh_workers',
    'prop_wfh_adopters': 'proportion_wfh_workers',
    'avg_wfh_intensity': 'avg_household_wfh_intensity',
    'max_wfh_intensity': 'max_household_wfh_intensity',
    'total_workers': 'num_working_members',
    'hh_wfh_saturation': 'household_wfh_saturation'
}

# Category mappings for better readability
CATEGORY_MAPPINGS = {
    'agegroup': {
        'transform_name': 'age_group',
        'categories': {
            '5 to 12': 'Child_5_12',
            '13 to 17': 'Teen_13_17', 
            '18 to 24': 'Young_Adult_18_24',
            '25 to 34': 'Adult_25_34',
            '35 to 49': 'Adult_35_49',
            '50 to 59': 'Adult_50_59',
            '60 to 69': 'Senior_60_69',
            '70+': 'Senior_70_Plus'
        }
    },
    'emptype': {
        'transform_name': 'employment_type',
        'categories': {
            'Employee': 'Employee',
            'Self employed': 'Self_Employed',
            'Employer': 'Business_Owner',
            'Contributing family worker': 'Family_Worker',
            'Other': 'Other_Employment'
        }
    },
    'hhinc_group': {
        'transform_name': 'household_income_group',
        'categories': {
            '$1-$19,999': 'Low_Income_Under_20k',
            '$20,000-$39,999': 'Lower_Mid_Income_20k_40k',
            '$40,000-$59,999': 'Mid_Income_40k_60k', 
            '$60,000-$79,999': 'Upper_Mid_Income_60k_80k',
            '$80,000-$99,999': 'High_Income_80k_100k',
            '$100,000+': 'Very_High_Income_100k_Plus'
        }
    }
}

def create_readable_columns(df):
    """
    Transform VISTA variable names to human-readable column names.
    
    Args:
        df: DataFrame with original VISTA column names
        
    Returns:
        DataFrame with readable column names
    """
    df_readable = df.copy()
    
    # Apply direct column name mappings
    df_readable = df_readable.rename(columns=VARIABLE_MAPPING)
    
    print("Column name transformations applied:")
    for old_name, new_name in VARIABLE_MAPPING.items():
        if old_name in df.columns:
            print(f"  {old_name} → {new_name}")
    
    return df_readable

def create_feature_dictionary():
    """
    Create a data dictionary explaining all transformed features.
    
    Returns:
        Dictionary with feature explanations
    """
    feature_dict = {
        # Demographics
        'age_group': 'Age group classification (Child_5_12, Teen_13_17, Young_Adult_18_24, etc.)',
        'personal_income': 'Individual income level category',
        'employment_type': 'Type of employment (Employee, Self_Employed, Business_Owner, etc.)',
        'main_activity': 'Primary daily activity (work, study, home duties, etc.)',
        
        # Household characteristics
        'household_size': 'Number of people living in the household',
        'vehicles_owned': 'Number of vehicles owned by household',
        'dwelling_type': 'Type of housing (house, apartment, etc.)',
        'household_income_group': 'Household income bracket (Low_Income_Under_20k, etc.)',
        
        # Work patterns
        'has_any_work': 'Whether person has any form of employment (Yes/No)',
        'works_from_home_any': 'Whether person works from home at all (Yes/No)',
        'wfh_monday': 'Works from home on Mondays (1=Yes, 0=No)',
        'wfh_tuesday': 'Works from home on Tuesdays (1=Yes, 0=No)', 
        'wfh_wednesday': 'Works from home on Wednesdays (1=Yes, 0=No)',
        'wfh_thursday': 'Works from home on Thursdays (1=Yes, 0=No)',
        'wfh_friday': 'Works from home on Fridays (1=Yes, 0=No)',
        
        # Location
        'home_region': 'Melbourne region of residence',
        'home_local_gov_area': 'Local government area of residence',
        
        # Transport
        'has_car_license': 'Has valid car driving license (Yes/No)',
        'commute_time_minutes': 'Daily commute time in minutes',
        'commute_distance_km': 'Daily commute distance in kilometers',
        'primary_transport_mode': 'Main mode of transport for work journeys',
        
        # Derived metrics
        'wfh_intensity_total': 'Total WFH days per week (0-7 scale)',
        'wfh_adopter': 'Binary indicator of WFH adoption (1=adopts WFH, 0=no WFH)',
        'wfh_category': 'WFH frequency category (No_WFH, Occasional_WFH, Regular_WFH, Full_WFH)',
        
        # Survey weights
        'person_survey_weight': 'Statistical weight for individual-level analysis',
        'household_survey_weight': 'Statistical weight for household-level analysis',
        'analysis_weight': 'Recommended weight for current analysis type'
    }
    
    return feature_dict

def save_readable_dataset(df, output_path, include_dictionary=True):
    """
    Save dataset with readable column names and optional data dictionary.
    
    Args:
        df: DataFrame to transform and save
        output_path: Path to save the readable dataset
        include_dictionary: Whether to save feature dictionary as JSON
    """
    # Transform column names
    df_readable = create_readable_columns(df)
    
    # Save readable dataset
    df_readable.to_csv(output_path, index=False)
    print(f"Readable dataset saved to: {output_path}")
    
    # Save feature dictionary if requested
    if include_dictionary:
        dict_path = output_path.replace('.csv', '_feature_dictionary.json')
        feature_dict = create_feature_dictionary()
        
        import json
        with open(dict_path, 'w') as f:
            json.dump(feature_dict, f, indent=2)
        print(f"Feature dictionary saved to: {dict_path}")
    
    # Print summary
    print(f"\nDataset transformation summary:")
    print(f"  Original columns: {len(df.columns)}")
    print(f"  Transformed columns: {len(df_readable.columns)}")
    print(f"  Rows: {len(df_readable)}")
    
    return df_readable

# Usage example for ML team
def get_ml_ready_features():
    """
    Returns list of recommended features for ML models with readable names.
    """
    ml_features = [
        # Demographics (most predictive)
        'age_group',
        'personal_income', 
        'employment_type',
        'main_activity',
        
        # Household context
        'household_size',
        'vehicles_owned',
        'household_income_group',
        'dwelling_type',
        
        # Location factors
        'home_region',
        
        # Transport access
        'has_car_license',
        'commute_time_minutes',
        'commute_distance_km',
        'primary_transport_mode',
        
        # Target variable
        'wfh_adopter'  # Binary: 1=adopts WFH, 0=no WFH
    ]
    
    return ml_features

if __name__ == "__main__":
    print("Variable Mapping Utility")
    print("=" * 50)
    print("Available functions:")
    print("- create_readable_columns(df): Transform column names")
    print("- create_feature_dictionary(): Get feature explanations") 
    print("- save_readable_dataset(df, path): Save with readable names")
    print("- get_ml_ready_features(): Get recommended ML features")
    print("\nExample usage:")
    print("df_clean = create_readable_columns(df_original)")
    print("ml_features = get_ml_ready_features()")
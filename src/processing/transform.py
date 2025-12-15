"""Data processing and transformation functions."""

import pandas as pd


def aggregate_by_vehicle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate data by vehicle.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Aggregated DataFrame with vehicle-level statistics
    """
    aggregated = df.groupby('vehicle_id').agg({
        'speed_mph': ['mean', 'max', 'min'],
        'fuel_consumption_mpg': 'mean',
        'engine_temp_f': 'mean',
        'rpm': 'mean',
        'distance_miles': 'max',
        'location': 'first'
    }).reset_index()
    
    aggregated.columns = ['_'.join(col).strip('_') for col in aggregated.columns.values]
    return aggregated


def aggregate_by_time(df: pd.DataFrame, freq: str = 'H') -> pd.DataFrame:
    """
    Aggregate data by time period.
    
    Args:
        df: Input DataFrame
        freq: Frequency for aggregation ('H' for hourly, 'D' for daily)
        
    Returns:
        Time-aggregated DataFrame
    """
    df_copy = df.copy()
    df_copy.set_index('timestamp', inplace=True)
    
    aggregated = df_copy.groupby('vehicle_id').resample(freq).agg({
        'speed_mph': 'mean',
        'fuel_consumption_mpg': 'mean',
        'engine_temp_f': 'mean',
        'rpm': 'mean'
    }).reset_index()
    
    return aggregated


def calculate_efficiency_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate efficiency score based on speed and fuel consumption.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with efficiency score added
    """
    df_copy = df.copy()
    df_copy['efficiency_score'] = (df_copy['fuel_consumption_mpg'] / 
                                   (df_copy['speed_mph'] + 1)) * 100
    return df_copy


def identify_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify potential anomalies in the data.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with anomaly flags
    """
    df_copy = df.copy()
    
    # Flag high temperature
    df_copy['high_temp'] = df_copy['engine_temp_f'] > 205
    
    # Flag low fuel efficiency
    df_copy['low_efficiency'] = df_copy['fuel_consumption_mpg'] < 24
    
    # Flag high RPM
    df_copy['high_rpm'] = df_copy['rpm'] > 3200
    
    return df_copy

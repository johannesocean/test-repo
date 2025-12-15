"""Utility functions for the automotive data dashboard."""

import pandas as pd
from pathlib import Path


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load automotive data from CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with parsed timestamps
    """
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def filter_by_vehicle(df: pd.DataFrame, vehicle_ids: list) -> pd.DataFrame:
    """
    Filter data by vehicle IDs.
    
    Args:
        df: Input DataFrame
        vehicle_ids: List of vehicle IDs to filter
        
    Returns:
        Filtered DataFrame
    """
    if not vehicle_ids:
        return df
    return df[df['vehicle_id'].isin(vehicle_ids)]


def filter_by_date_range(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """
    Filter data by date range.
    
    Args:
        df: Input DataFrame
        start_date: Start date
        end_date: End date
        
    Returns:
        Filtered DataFrame
    """
    return df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]


def filter_by_location(df: pd.DataFrame, locations: list) -> pd.DataFrame:
    """
    Filter data by locations.
    
    Args:
        df: Input DataFrame
        locations: List of locations to filter
        
    Returns:
        Filtered DataFrame
    """
    if not locations:
        return df
    return df[df['location'].isin(locations)]


def get_summary_stats(df: pd.DataFrame) -> dict:
    """
    Calculate summary statistics for the dataset.
    Averages exclude idle vehicles (speed > 0) for more accurate metrics.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary of summary statistics
    """
    # Filter out idle records for meaningful averages
    active_df = df[df['speed_mph'] > 0]
    
    return {
        'total_records': len(df),
        'unique_vehicles': df['vehicle_id'].nunique(),
        'avg_speed': active_df['speed_mph'].mean() if len(active_df) > 0 else 0,
        'avg_fuel_consumption': active_df['fuel_consumption_mpg'].mean() if len(active_df) > 0 else 0,
        'avg_engine_temp': df['engine_temp_f'].mean(),
        'total_distance': df.groupby('vehicle_id')['distance_miles'].max().sum()
    }

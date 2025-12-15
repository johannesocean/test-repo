"""Visualization functions for automotive data dashboard."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_speed_over_time_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a line chart showing vehicle speed over time.
    
    Args:
        df: DataFrame with timestamp, speed_mph, and vehicle_id columns
        
    Returns:
        Plotly figure object
    """
    fig = px.line(
        df,
        x='timestamp',
        y='speed_mph',
        color='vehicle_id',
        title='Vehicle Speed Over Time',
        labels={'speed_mph': 'Speed (mph)', 'timestamp': 'Time'}
    )
    fig.update_layout(height=400)
    return fig


def create_rpm_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a histogram showing RPM distribution by vehicle.
    
    Args:
        df: DataFrame with rpm and vehicle_id columns
        
    Returns:
        Plotly figure object
    """
    fig = px.histogram(
        df,
        x='rpm',
        color='vehicle_id',
        title='RPM Distribution by Vehicle',
        labels={'rpm': 'RPM', 'count': 'Frequency'},
        nbins=30
    )
    fig.update_layout(height=400)
    return fig


def create_fuel_consumption_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a line chart showing fuel consumption over time.
    
    Args:
        df: DataFrame with timestamp, fuel_consumption_mpg, and vehicle_id columns
        
    Returns:
        Plotly figure object
    """
    fig = px.line(
        df,
        x='timestamp',
        y='fuel_consumption_mpg',
        color='vehicle_id',
        title='Fuel Consumption (MPG) Over Time',
        labels={'fuel_consumption_mpg': 'MPG', 'timestamp': 'Time'}
    )
    fig.update_layout(height=400)
    return fig


def create_efficiency_score_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a box plot showing efficiency score by vehicle.
    
    Args:
        df: DataFrame with efficiency_score and vehicle_id columns
        
    Returns:
        Plotly figure object
    """
    fig = px.box(
        df,
        x='vehicle_id',
        y='efficiency_score',
        color='vehicle_id',
        title='Efficiency Score by Vehicle',
        labels={'efficiency_score': 'Efficiency Score', 'vehicle_id': 'Vehicle'}
    )
    fig.update_layout(height=400)
    return fig


def create_engine_temp_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a line chart showing engine temperature over time.
    
    Args:
        df: DataFrame with timestamp, engine_temp_f, and vehicle_id columns
        
    Returns:
        Plotly figure object
    """
    fig = px.line(
        df,
        x='timestamp',
        y='engine_temp_f',
        color='vehicle_id',
        title='Engine Temperature Over Time',
        labels={'engine_temp_f': 'Temperature (°F)', 'timestamp': 'Time'}
    )
    fig.add_hline(
        y=205,
        line_dash="dash",
        line_color="red",
        annotation_text="High Temp Threshold"
    )
    fig.update_layout(height=400)
    return fig


def create_vehicle_comparison_chart(vehicle_agg: pd.DataFrame) -> go.Figure:
    """
    Create a grouped bar chart comparing vehicle performance.
    
    Args:
        vehicle_agg: Aggregated DataFrame with vehicle statistics
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Avg Speed',
        x=vehicle_agg['vehicle_id'],
        y=vehicle_agg['speed_mph_mean'],
        yaxis='y',
        offsetgroup=1
    ))
    
    fig.add_trace(go.Bar(
        name='Avg MPG',
        x=vehicle_agg['vehicle_id'],
        y=vehicle_agg['fuel_consumption_mpg_mean'],
        yaxis='y2',
        offsetgroup=2
    ))
    
    fig.update_layout(
        title='Vehicle Performance Comparison',
        xaxis=dict(title='Vehicle ID'),
        yaxis=dict(title='Avg Speed (mph)', side='left'),
        yaxis2=dict(title='Avg MPG', side='right', overlaying='y'),
        barmode='group',
        height=400
    )
    
    return fig

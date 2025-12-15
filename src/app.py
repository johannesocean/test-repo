"""Main Streamlit application for Automotive Data Dashboard."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    load_data,
    filter_by_vehicle,
    filter_by_date_range,
    filter_by_location,
    get_summary_stats
)
from processing.transform import (
    aggregate_by_vehicle,
    aggregate_by_time,
    calculate_efficiency_score,
    identify_anomalies
)


# Page configuration
st.set_page_config(
    page_title="Automotive Data Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🚗 Automotive Data Dashboard")
st.markdown("### Process, analyze, and visualize automotive data")

# Load data
@st.cache_data
def get_data():
    data_path = Path(__file__).parent / "data" / "automotive_data.csv"
    return load_data(str(data_path))

try:
    df = get_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Vehicle filter
all_vehicles = sorted(df['vehicle_id'].unique())
selected_vehicles = st.sidebar.multiselect(
    "Select Vehicles",
    options=all_vehicles,
    default=all_vehicles
)

# Location filter
all_locations = sorted(df['location'].unique())
selected_locations = st.sidebar.multiselect(
    "Select Locations",
    options=all_locations,
    default=all_locations
)

# Date range filter
min_date = df['timestamp'].min().date()
max_date = df['timestamp'].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range[0]

# Apply filters
filtered_df = df.copy()
filtered_df = filter_by_vehicle(filtered_df, selected_vehicles)
filtered_df = filter_by_location(filtered_df, selected_locations)
# Add time component to end_date to include the entire day
filtered_df = filter_by_date_range(
    filtered_df,
    pd.Timestamp(start_date),
    pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
)

# Display summary statistics
st.header("📊 Summary Statistics")

if len(filtered_df) > 0:
    stats = get_summary_stats(filtered_df)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Records", f"{stats['total_records']:,}")
    with col2:
        st.metric("Unique Vehicles", stats['unique_vehicles'])
    with col3:
        st.metric("Avg Speed (mph)", f"{stats['avg_speed']:.1f}")
    with col4:
        st.metric("Avg MPG", f"{stats['avg_fuel_consumption']:.1f}")
    with col5:
        st.metric("Avg Engine Temp (°F)", f"{stats['avg_engine_temp']:.1f}")
    
    # Main visualizations
    st.header("📈 Interactive Visualizations")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Speed & Performance", "Fuel Efficiency", "Engine Metrics", "Vehicle Comparison"])
    
    with tab1:
        st.subheader("Speed Over Time")
        fig1 = px.line(
            filtered_df,
            x='timestamp',
            y='speed_mph',
            color='vehicle_id',
            title='Vehicle Speed Over Time',
            labels={'speed_mph': 'Speed (mph)', 'timestamp': 'Time'}
        )
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)
        
        st.subheader("RPM Distribution")
        fig2 = px.histogram(
            filtered_df,
            x='rpm',
            color='vehicle_id',
            title='RPM Distribution by Vehicle',
            labels={'rpm': 'RPM', 'count': 'Frequency'},
            nbins=30
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.subheader("Fuel Consumption Over Time")
        fig3 = px.line(
            filtered_df,
            x='timestamp',
            y='fuel_consumption_mpg',
            color='vehicle_id',
            title='Fuel Consumption (MPG) Over Time',
            labels={'fuel_consumption_mpg': 'MPG', 'timestamp': 'Time'}
        )
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)
        
        # Calculate efficiency score
        efficiency_df = calculate_efficiency_score(filtered_df)
        st.subheader("Efficiency Score")
        fig4 = px.box(
            efficiency_df,
            x='vehicle_id',
            y='efficiency_score',
            color='vehicle_id',
            title='Efficiency Score by Vehicle',
            labels={'efficiency_score': 'Efficiency Score', 'vehicle_id': 'Vehicle'}
        )
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)
    
    with tab3:
        st.subheader("Engine Temperature Over Time")
        fig5 = px.line(
            filtered_df,
            x='timestamp',
            y='engine_temp_f',
            color='vehicle_id',
            title='Engine Temperature Over Time',
            labels={'engine_temp_f': 'Temperature (°F)', 'timestamp': 'Time'}
        )
        fig5.add_hline(y=205, line_dash="dash", line_color="red", 
                      annotation_text="High Temp Threshold")
        fig5.update_layout(height=400)
        st.plotly_chart(fig5, use_container_width=True)
        
        # Anomaly detection
        anomaly_df = identify_anomalies(filtered_df)
        st.subheader("Anomaly Detection")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            high_temp_count = anomaly_df['high_temp'].sum()
            st.metric("High Temperature Events", high_temp_count)
        with col2:
            low_eff_count = anomaly_df['low_efficiency'].sum()
            st.metric("Low Efficiency Events", low_eff_count)
        with col3:
            high_rpm_count = anomaly_df['high_rpm'].sum()
            st.metric("High RPM Events", high_rpm_count)
    
    with tab4:
        st.subheader("Vehicle Performance Comparison")
        
        # Aggregate by vehicle
        vehicle_agg = aggregate_by_vehicle(filtered_df)
        
        # Create comparison chart
        fig6 = go.Figure()
        
        fig6.add_trace(go.Bar(
            name='Avg Speed',
            x=vehicle_agg['vehicle_id'],
            y=vehicle_agg['speed_mph_mean'],
            yaxis='y',
            offsetgroup=1
        ))
        
        fig6.add_trace(go.Bar(
            name='Avg MPG',
            x=vehicle_agg['vehicle_id'],
            y=vehicle_agg['fuel_consumption_mpg_mean'],
            yaxis='y2',
            offsetgroup=2
        ))
        
        fig6.update_layout(
            title='Vehicle Performance Comparison',
            xaxis=dict(title='Vehicle ID'),
            yaxis=dict(title='Avg Speed (mph)', side='left'),
            yaxis2=dict(title='Avg MPG', side='right', overlaying='y'),
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig6, use_container_width=True)
        
        # Show aggregated table
        st.subheader("Vehicle Statistics Table")
        st.dataframe(
            vehicle_agg.style.format({
                'speed_mph_mean': '{:.2f}',
                'speed_mph_max': '{:.2f}',
                'speed_mph_min': '{:.2f}',
                'fuel_consumption_mpg_mean': '{:.2f}',
                'engine_temp_f_mean': '{:.2f}',
                'rpm_mean': '{:.0f}',
                'distance_miles_max': '{:.2f}'
            }),
            use_container_width=True
        )
    
    # Raw data table
    st.header("📋 Raw Data")
    
    # Time aggregation option
    time_agg = st.selectbox(
        "Aggregate by time period",
        options=['None', 'Hourly', 'Daily'],
        index=0
    )
    
    if time_agg == 'Hourly':
        display_df = aggregate_by_time(filtered_df, 'H')
    elif time_agg == 'Daily':
        display_df = aggregate_by_time(filtered_df, 'D')
    else:
        display_df = filtered_df
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Download button
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name="automotive_data_filtered.csv",
        mime="text/csv"
    )
    
else:
    st.warning("No data matches the selected filters. Please adjust your filter criteria.")

# Footer
st.markdown("---")
st.markdown("**Automotive Data Dashboard** | Built with Streamlit, Pandas, and Plotly")

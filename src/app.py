"""Main Streamlit application for Automotive Data Dashboard."""

import streamlit as st
import pandas as pd
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
from plotting.charts import (
    create_speed_over_time_chart,
    create_rpm_distribution_chart,
    create_fuel_consumption_chart,
    create_efficiency_score_chart,
    create_engine_temp_chart,
    create_vehicle_comparison_chart
)


def configure_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Automotive Data Dashboard",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def render_header():
    """Render the page header."""
    st.title("🚗 Automotive Data Dashboard")
    st.markdown("### Process, analyze, and visualize automotive data")


@st.cache_data
def get_data():
    """Load and cache automotive data."""
    data_path = Path(__file__).parent / "data" / "automotive_data.csv"
    return load_data(str(data_path))


def render_sidebar_filters(df: pd.DataFrame) -> tuple:
    """
    Render sidebar filters and return selected filter values.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (selected_vehicles, selected_locations, start_date, end_date)
    """
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
    
    return selected_vehicles, selected_locations, start_date, end_date


def apply_filters(
    df: pd.DataFrame,
    selected_vehicles: list,
    selected_locations: list,
    start_date,
    end_date
) -> pd.DataFrame:
    """
    Apply filters to the DataFrame.
    
    Args:
        df: Input DataFrame
        selected_vehicles: List of selected vehicle IDs
        selected_locations: List of selected locations
        start_date: Start date for filtering
        end_date: End date for filtering
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    filtered_df = filter_by_vehicle(filtered_df, selected_vehicles)
    filtered_df = filter_by_location(filtered_df, selected_locations)
    # Add time component to end_date to include the entire day
    filtered_df = filter_by_date_range(
        filtered_df,
        pd.Timestamp(start_date),
        pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    )
    return filtered_df


def render_summary_statistics(filtered_df: pd.DataFrame):
    """
    Render summary statistics section.
    
    Args:
        filtered_df: Filtered DataFrame
    """
    st.header("📊 Summary Statistics")
    
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


def render_speed_performance_tab(filtered_df: pd.DataFrame):
    """
    Render Speed & Performance tab content.
    
    Args:
        filtered_df: Filtered DataFrame
    """
    st.subheader("Speed Over Time")
    fig1 = create_speed_over_time_chart(filtered_df)
    st.plotly_chart(fig1, use_container_width=True)
    
    st.subheader("RPM Distribution")
    fig2 = create_rpm_distribution_chart(filtered_df)
    st.plotly_chart(fig2, use_container_width=True)


def render_fuel_efficiency_tab(filtered_df: pd.DataFrame):
    """
    Render Fuel Efficiency tab content.
    
    Args:
        filtered_df: Filtered DataFrame
    """
    st.subheader("Fuel Consumption Over Time")
    fig3 = create_fuel_consumption_chart(filtered_df)
    st.plotly_chart(fig3, use_container_width=True)
    
    # Calculate efficiency score
    efficiency_df = calculate_efficiency_score(filtered_df)
    st.subheader("Efficiency Score")
    fig4 = create_efficiency_score_chart(efficiency_df)
    st.plotly_chart(fig4, use_container_width=True)


def render_engine_metrics_tab(filtered_df: pd.DataFrame):
    """
    Render Engine Metrics tab content.
    
    Args:
        filtered_df: Filtered DataFrame
    """
    st.subheader("Engine Temperature Over Time")
    fig5 = create_engine_temp_chart(filtered_df)
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


def render_vehicle_comparison_tab(filtered_df: pd.DataFrame):
    """
    Render Vehicle Comparison tab content.
    
    Args:
        filtered_df: Filtered DataFrame
    """
    st.subheader("Vehicle Performance Comparison")
    
    # Aggregate by vehicle
    vehicle_agg = aggregate_by_vehicle(filtered_df)
    
    # Create comparison chart
    fig6 = create_vehicle_comparison_chart(vehicle_agg)
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


def render_visualizations(filtered_df: pd.DataFrame):
    """
    Render the main visualizations section with tabs.
    
    Args:
        filtered_df: Filtered DataFrame
    """
    st.header("📈 Interactive Visualizations")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "Speed & Performance",
        "Fuel Efficiency",
        "Engine Metrics",
        "Vehicle Comparison"
    ])
    
    with tab1:
        render_speed_performance_tab(filtered_df)
    
    with tab2:
        render_fuel_efficiency_tab(filtered_df)
    
    with tab3:
        render_engine_metrics_tab(filtered_df)
    
    with tab4:
        render_vehicle_comparison_tab(filtered_df)


def render_raw_data_section(filtered_df: pd.DataFrame):
    """
    Render raw data section with optional aggregation.
    
    Args:
        filtered_df: Filtered DataFrame
    """
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


def render_footer():
    """Render the page footer."""
    st.markdown("---")
    st.markdown("**Automotive Data Dashboard** | Built with Streamlit, Pandas, and Plotly")


def main():
    """Main application function."""
    # Configure page
    configure_page()
    
    # Render header
    render_header()
    
    # Load data
    try:
        df = get_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    # Render filters and get selections
    selected_vehicles, selected_locations, start_date, end_date = render_sidebar_filters(df)
    
    # Apply filters
    filtered_df = apply_filters(df, selected_vehicles, selected_locations, start_date, end_date)
    
    # Check if filtered data is empty
    if len(filtered_df) > 0:
        # Render summary statistics
        render_summary_statistics(filtered_df)
        
        # Render visualizations
        render_visualizations(filtered_df)
        
        # Render raw data section
        render_raw_data_section(filtered_df)
    else:
        st.warning("No data matches the selected filters. Please adjust your filter criteria.")
    
    # Render footer
    render_footer()


if __name__ == "__main__":
    main()

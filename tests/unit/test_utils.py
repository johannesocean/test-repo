"""Unit tests for utils module."""

from unittest.mock import mock_open, patch

import pandas as pd
import pytest

from utils import (
    filter_by_date_range,
    filter_by_location,
    filter_by_vehicle,
    get_summary_stats,
    load_data,
)


class TestLoadData:
    """Test suite for load_data function."""

    @patch("builtins.open", new_callable=mock_open, read_data="timestamp,value\n2024-01-01,10")
    @patch("pandas.read_csv")
    def test_load_data_basic(self, mock_read_csv: pytest.fixture, mock_file: pytest.fixture) -> None:
        """Test basic data loading from CSV file."""
        # Arrange
        mock_df = pd.DataFrame({
            'timestamp': ['2024-01-01 10:00:00'],
            'value': [10]
        })
        mock_read_csv.return_value = mock_df

        # Act
        result_df = load_data("test.csv")

        # Assert
        mock_read_csv.assert_called_once_with("test.csv")
        assert 'timestamp' in result_df.columns


class TestFilterByVehicle:
    """Test suite for filter_by_vehicle function."""

    def test_filter_by_vehicle_basic(self) -> None:
        """Test basic vehicle filtering."""
        # Arrange
        input_df = pd.DataFrame({
            'vehicle_id': ['V1', 'V2', 'V3'],
            'value': [10, 20, 30]
        })

        # Act
        result_df = filter_by_vehicle(input_df, ['V1', 'V3'])

        # Assert
        assert len(result_df) == 2
        assert 'V1' in result_df['vehicle_id'].to_numpy()
        assert 'V3' in result_df['vehicle_id'].to_numpy()

    def test_filter_by_vehicle_empty_list(self) -> None:
        """Test filtering with empty vehicle list returns all data."""
        # Arrange
        input_df = pd.DataFrame({
            'vehicle_id': ['V1', 'V2', 'V3'],
            'value': [10, 20, 30]
        })

        # Act
        result_df = filter_by_vehicle(input_df, [])

        # Assert
        assert len(result_df) == 3


class TestFilterByDateRange:
    """Test suite for filter_by_date_range function."""

    def test_filter_by_date_range_basic(self) -> None:
        """Test basic date range filtering."""
        # Arrange
        input_df = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-01-01', '2024-01-05', '2024-01-10']),
            'value': [10, 20, 30]
        })

        # Act
        result_df = filter_by_date_range(
            input_df,
            pd.Timestamp('2024-01-01'),
            pd.Timestamp('2024-01-05')
        )

        # Assert
        assert len(result_df) == 2


class TestFilterByLocation:
    """Test suite for filter_by_location function."""

    def test_filter_by_location_basic(self) -> None:
        """Test basic location filtering."""
        # Arrange
        input_df = pd.DataFrame({
            'location': ['NYC', 'LA', 'SF'],
            'value': [10, 20, 30]
        })

        # Act
        result_df = filter_by_location(input_df, ['NYC', 'SF'])

        # Assert
        assert len(result_df) == 2

    def test_filter_by_location_empty_list(self) -> None:
        """Test filtering with empty location list returns all data."""
        # Arrange
        input_df = pd.DataFrame({
            'location': ['NYC', 'LA', 'SF'],
            'value': [10, 20, 30]
        })

        # Act
        result_df = filter_by_location(input_df, [])

        # Assert
        assert len(result_df) == 3


class TestGetSummaryStats:
    """Test suite for get_summary_stats function."""

    def test_get_summary_stats_basic(self) -> None:
        """Test basic summary statistics calculation."""
        # Arrange
        input_df = pd.DataFrame({
            'vehicle_id': ['V1', 'V1', 'V2'],
            'speed_mph': [60.0, 70.0, 50.0],
            'fuel_consumption_mpg': [25.0, 30.0, 28.0],
            'engine_temp_f': [180.0, 190.0, 175.0],
            'distance_miles': [10.0, 20.0, 15.0]
        })

        # Act
        stats = get_summary_stats(input_df)

        # Assert
        assert stats['total_records'] == 3
        assert stats['unique_vehicles'] == 2
        assert stats['avg_speed'] == pytest.approx(60.0)
        assert stats['avg_fuel_consumption'] == pytest.approx(27.666666, rel=1e-5)
        assert stats['avg_engine_temp'] == pytest.approx(181.666666, rel=1e-5)
        assert stats['total_distance'] == pytest.approx(35.0)

    def test_get_summary_stats_with_idle_vehicles(self) -> None:
        """Test summary statistics excludes idle vehicles for speed and fuel."""
        # Arrange
        input_df = pd.DataFrame({
            'vehicle_id': ['V1', 'V1', 'V2'],
            'speed_mph': [0.0, 60.0, 70.0],
            'fuel_consumption_mpg': [0.0, 25.0, 30.0],
            'engine_temp_f': [180.0, 190.0, 175.0],
            'distance_miles': [0.0, 20.0, 15.0]
        })

        # Act
        stats = get_summary_stats(input_df)

        # Assert
        assert stats['avg_speed'] == pytest.approx(65.0)
        assert stats['avg_fuel_consumption'] == pytest.approx(27.5)

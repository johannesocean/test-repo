"""Unit tests for processing.transform module."""

import pandas as pd
import pytest

from processing.transform import (
    aggregate_by_time,
    aggregate_by_vehicle,
    calculate_efficiency_score,
    identify_anomalies,
)


class TestAggregateByVehicle:
    """Test suite for aggregate_by_vehicle function."""

    def test_aggregate_by_vehicle_basic(self) -> None:
        """Test basic aggregation by vehicle with expected columns and aggregations."""
        # Arrange
        input_df = pd.DataFrame({
            'vehicle_id': ['V1', 'V1', 'V2', 'V2'],
            'speed_mph': [60, 70, 50, 55],
            'fuel_consumption_mpg': [25, 28, 30, 32],
            'engine_temp_f': [180, 190, 175, 185],
            'rpm': [2000, 2200, 1800, 1900],
            'distance_miles': [10, 20, 5, 15],
            'location': ['NYC', 'NYC', 'LA', 'LA']
        })

        # Act
        result_df = aggregate_by_vehicle(input_df)

        # Assert
        assert len(result_df) == 2
        assert 'vehicle_id' in result_df.columns
        assert 'speed_mph_mean' in result_df.columns
        assert 'speed_mph_max' in result_df.columns
        assert 'speed_mph_min' in result_df.columns
        assert 'fuel_consumption_mpg_mean' in result_df.columns
        assert 'engine_temp_f_mean' in result_df.columns
        assert 'rpm_mean' in result_df.columns
        assert 'distance_miles_max' in result_df.columns
        assert 'location_first' in result_df.columns

    def test_aggregate_by_vehicle_calculations(self) -> None:
        """Test that aggregation calculations are correct."""
        # Arrange
        input_df = pd.DataFrame({
            'vehicle_id': ['V1', 'V1', 'V1'],
            'speed_mph': [60, 70, 80],
            'fuel_consumption_mpg': [25, 30, 35],
            'engine_temp_f': [180, 190, 200],
            'rpm': [2000, 2500, 3000],
            'distance_miles': [10, 20, 30],
            'location': ['NYC', 'NYC', 'NYC']
        })

        # Act
        result_df = aggregate_by_vehicle(input_df)

        # Assert
        assert len(result_df) == 1
        assert result_df['speed_mph_mean'].iloc[0] == pytest.approx(70.0)
        assert result_df['speed_mph_max'].iloc[0] == 80
        assert result_df['speed_mph_min'].iloc[0] == 60
        assert result_df['fuel_consumption_mpg_mean'].iloc[0] == pytest.approx(30.0)
        assert result_df['engine_temp_f_mean'].iloc[0] == pytest.approx(190.0)
        assert result_df['rpm_mean'].iloc[0] == pytest.approx(2500.0)
        assert result_df['distance_miles_max'].iloc[0] == 30
        assert result_df['location_first'].iloc[0] == 'NYC'

    def test_aggregate_by_vehicle_single_record(self) -> None:
        """Test aggregation with a single record per vehicle."""
        # Arrange
        input_df = pd.DataFrame({
            'vehicle_id': ['V1'],
            'speed_mph': [60],
            'fuel_consumption_mpg': [25],
            'engine_temp_f': [180],
            'rpm': [2000],
            'distance_miles': [10],
            'location': ['NYC']
        })

        # Act
        result_df = aggregate_by_vehicle(input_df)

        # Assert
        assert len(result_df) == 1
        assert result_df['speed_mph_mean'].iloc[0] == 60
        assert result_df['speed_mph_max'].iloc[0] == 60
        assert result_df['speed_mph_min'].iloc[0] == 60

    def test_aggregate_by_vehicle_empty_dataframe(self) -> None:
        """Test aggregation with an empty DataFrame."""
        # Arrange
        input_df = pd.DataFrame({
            'vehicle_id': [],
            'speed_mph': [],
            'fuel_consumption_mpg': [],
            'engine_temp_f': [],
            'rpm': [],
            'distance_miles': [],
            'location': []
        })

        # Act
        result_df = aggregate_by_vehicle(input_df)

        # Assert
        assert len(result_df) == 0


class TestAggregateByTime:
    """Test suite for aggregate_by_time function."""

    @pytest.mark.parametrize(("freq", "expected_freq"), [
        ("H", "H"),
        ("D", "D"),
    ])
    def test_aggregate_by_time_frequencies(self, freq: str, expected_freq: str) -> None:
        """Test time aggregation with different frequencies."""
        # Arrange
        input_df = pd.DataFrame({
            'timestamp': pd.to_datetime([
                '2024-01-01 10:00:00',
                '2024-01-01 10:30:00',
                '2024-01-01 11:00:00'
            ]),
            'vehicle_id': ['V1', 'V1', 'V1'],
            'speed_mph': [60.0, 65.0, 70.0],
            'fuel_consumption_mpg': [25.0, 27.0, 29.0],
            'engine_temp_f': [180.0, 185.0, 190.0],
            'rpm': [2000.0, 2200.0, 2400.0]
        })

        # Act
        result_df = aggregate_by_time(input_df, freq=freq)

        # Assert
        assert 'timestamp' in result_df.columns
        assert 'vehicle_id' in result_df.columns
        assert 'speed_mph' in result_df.columns
        assert 'fuel_consumption_mpg' in result_df.columns
        assert 'engine_temp_f' in result_df.columns
        assert 'rpm' in result_df.columns

    def test_aggregate_by_time_hourly(self) -> None:
        """Test hourly time aggregation calculations."""
        # Arrange
        input_df = pd.DataFrame({
            'timestamp': pd.to_datetime([
                '2024-01-01 10:00:00',
                '2024-01-01 10:30:00',
                '2024-01-01 11:00:00',
                '2024-01-01 11:30:00'
            ]),
            'vehicle_id': ['V1', 'V1', 'V1', 'V1'],
            'speed_mph': [60.0, 65.0, 70.0, 75.0],
            'fuel_consumption_mpg': [25.0, 27.0, 29.0, 31.0],
            'engine_temp_f': [180.0, 185.0, 190.0, 195.0],
            'rpm': [2000.0, 2200.0, 2400.0, 2600.0]
        })

        # Act
        result_df = aggregate_by_time(input_df, freq='H')

        # Assert
        assert len(result_df) == 2  # Two hourly buckets
        # First hour average
        first_hour_data = result_df[result_df['timestamp'] == pd.Timestamp('2024-01-01 10:00:00')]
        assert len(first_hour_data) == 1
        assert first_hour_data['speed_mph'].iloc[0] == pytest.approx(62.5)

    def test_aggregate_by_time_multiple_vehicles(self) -> None:
        """Test time aggregation with multiple vehicles."""
        # Arrange
        input_df = pd.DataFrame({
            'timestamp': pd.to_datetime([
                '2024-01-01 10:00:00',
                '2024-01-01 10:00:00',
                '2024-01-01 11:00:00',
                '2024-01-01 11:00:00'
            ]),
            'vehicle_id': ['V1', 'V2', 'V1', 'V2'],
            'speed_mph': [60.0, 50.0, 70.0, 55.0],
            'fuel_consumption_mpg': [25.0, 30.0, 29.0, 32.0],
            'engine_temp_f': [180.0, 175.0, 190.0, 185.0],
            'rpm': [2000.0, 1800.0, 2400.0, 1900.0]
        })

        # Act
        result_df = aggregate_by_time(input_df, freq='H')

        # Assert
        assert len(result_df) == 4  # 2 vehicles * 2 hours
        v1_data = result_df[result_df['vehicle_id'] == 'V1']
        assert len(v1_data) == 2

    def test_aggregate_by_time_preserves_original(self) -> None:
        """Test that original DataFrame is not modified."""
        # Arrange
        input_df = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-01-01 10:00:00']),
            'vehicle_id': ['V1'],
            'speed_mph': [60.0],
            'fuel_consumption_mpg': [25.0],
            'engine_temp_f': [180.0],
            'rpm': [2000.0]
        })
        original_index = input_df.index.to_numpy()

        # Act
        aggregate_by_time(input_df, freq='H')

        # Assert
        assert (input_df.index.to_numpy() == original_index).all()


class TestCalculateEfficiencyScore:
    """Test suite for calculate_efficiency_score function."""

    def test_calculate_efficiency_score_basic(self) -> None:
        """Test basic efficiency score calculation for moving vehicles."""
        # Arrange
        input_df = pd.DataFrame({
            'speed_mph': [60.0, 50.0],
            'fuel_consumption_mpg': [30.0, 25.0]
        })

        # Act
        result_df = calculate_efficiency_score(input_df)

        # Assert
        assert 'efficiency_score' in result_df.columns
        assert result_df['efficiency_score'].iloc[0] == pytest.approx(50.0)
        assert result_df['efficiency_score'].iloc[1] == pytest.approx(50.0)

    def test_calculate_efficiency_score_zero_speed(self) -> None:
        """Test efficiency score is zero when vehicle is not moving."""
        # Arrange
        input_df = pd.DataFrame({
            'speed_mph': [0.0, 0.0],
            'fuel_consumption_mpg': [30.0, 25.0]
        })

        # Act
        result_df = calculate_efficiency_score(input_df)

        # Assert
        assert result_df['efficiency_score'].iloc[0] == 0.0
        assert result_df['efficiency_score'].iloc[1] == 0.0

    def test_calculate_efficiency_score_mixed_speeds(self) -> None:
        """Test efficiency score calculation with mixed moving and stationary vehicles."""
        # Arrange
        input_df = pd.DataFrame({
            'speed_mph': [60.0, 0.0, 40.0],
            'fuel_consumption_mpg': [30.0, 25.0, 20.0]
        })

        # Act
        result_df = calculate_efficiency_score(input_df)

        # Assert
        assert result_df['efficiency_score'].iloc[0] == pytest.approx(50.0)
        assert result_df['efficiency_score'].iloc[1] == 0.0
        assert result_df['efficiency_score'].iloc[2] == pytest.approx(50.0)

    @pytest.mark.parametrize(("speed", "mpg", "expected_score"), [
        (60.0, 30.0, 50.0),
        (50.0, 25.0, 50.0),
        (100.0, 40.0, 40.0),
        (80.0, 32.0, 40.0),
    ])
    def test_calculate_efficiency_score_parameterized(
        self, speed: float, mpg: float, expected_score: float
    ) -> None:
        """Test efficiency score calculation with various parameter combinations."""
        # Arrange
        input_df = pd.DataFrame({
            'speed_mph': [speed],
            'fuel_consumption_mpg': [mpg]
        })

        # Act
        result_df = calculate_efficiency_score(input_df)

        # Assert
        assert result_df['efficiency_score'].iloc[0] == pytest.approx(expected_score)

    def test_calculate_efficiency_score_preserves_original(self) -> None:
        """Test that original DataFrame is not modified."""
        # Arrange
        input_df = pd.DataFrame({
            'speed_mph': [60.0],
            'fuel_consumption_mpg': [30.0]
        })
        original_columns = input_df.columns.to_list()

        # Act
        result_df = calculate_efficiency_score(input_df)

        # Assert
        assert input_df.columns.to_list() == original_columns
        assert 'efficiency_score' not in input_df.columns
        assert 'efficiency_score' in result_df.columns


class TestIdentifyAnomalies:
    """Test suite for identify_anomalies function."""

    def test_identify_anomalies_basic(self) -> None:
        """Test basic anomaly identification adds expected columns."""
        # Arrange
        input_df = pd.DataFrame({
            'engine_temp_f': [200.0],
            'fuel_consumption_mpg': [25.0],
            'rpm': [3000.0]
        })

        # Act
        result_df = identify_anomalies(input_df)

        # Assert
        assert 'high_temp' in result_df.columns
        assert 'low_efficiency' in result_df.columns
        assert 'high_rpm' in result_df.columns

    def test_identify_anomalies_high_temp(self) -> None:
        """Test high temperature anomaly detection."""
        # Arrange
        input_df = pd.DataFrame({
            'engine_temp_f': [200.0, 206.0, 210.0],
            'fuel_consumption_mpg': [25.0, 26.0, 27.0],
            'rpm': [3000.0, 3100.0, 3150.0]
        })

        # Act
        result_df = identify_anomalies(input_df)

        # Assert
        assert result_df['high_temp'].iloc[0] == False
        assert result_df['high_temp'].iloc[1] == True
        assert result_df['high_temp'].iloc[2] == True

    def test_identify_anomalies_low_efficiency(self) -> None:
        """Test low fuel efficiency anomaly detection."""
        # Arrange
        input_df = pd.DataFrame({
            'engine_temp_f': [200.0, 200.0, 200.0],
            'fuel_consumption_mpg': [25.0, 23.0, 20.0],
            'rpm': [3000.0, 3000.0, 3000.0]
        })

        # Act
        result_df = identify_anomalies(input_df)

        # Assert
        assert result_df['low_efficiency'].iloc[0] == False
        assert result_df['low_efficiency'].iloc[1] == True
        assert result_df['low_efficiency'].iloc[2] == True

    def test_identify_anomalies_high_rpm(self) -> None:
        """Test high RPM anomaly detection."""
        # Arrange
        input_df = pd.DataFrame({
            'engine_temp_f': [200.0, 200.0, 200.0],
            'fuel_consumption_mpg': [25.0, 26.0, 27.0],
            'rpm': [3000.0, 3201.0, 3500.0]
        })

        # Act
        result_df = identify_anomalies(input_df)

        # Assert
        assert result_df['high_rpm'].iloc[0] == False
        assert result_df['high_rpm'].iloc[1] == True
        assert result_df['high_rpm'].iloc[2] == True

    @pytest.mark.parametrize(("temp", "mpg", "rpm", "expected_high_temp", "expected_low_eff", "expected_high_rpm"), [
        (200.0, 25.0, 3000.0, False, False, False),
        (206.0, 25.0, 3000.0, True, False, False),
        (200.0, 23.0, 3000.0, False, True, False),
        (200.0, 25.0, 3201.0, False, False, True),
        (210.0, 20.0, 3500.0, True, True, True),
    ])
    def test_identify_anomalies_parameterized(
        self,
        temp: float,
        mpg: float,
        rpm: float,
        expected_high_temp: bool,
        expected_low_eff: bool,
        expected_high_rpm: bool
    ) -> None:
        """Test anomaly identification with various parameter combinations."""
        # Arrange
        input_df = pd.DataFrame({
            'engine_temp_f': [temp],
            'fuel_consumption_mpg': [mpg],
            'rpm': [rpm]
        })

        # Act
        result_df = identify_anomalies(input_df)

        # Assert
        assert result_df['high_temp'].iloc[0] == expected_high_temp
        assert result_df['low_efficiency'].iloc[0] == expected_low_eff
        assert result_df['high_rpm'].iloc[0] == expected_high_rpm

    def test_identify_anomalies_preserves_original(self) -> None:
        """Test that original DataFrame is not modified."""
        # Arrange
        input_df = pd.DataFrame({
            'engine_temp_f': [200.0],
            'fuel_consumption_mpg': [25.0],
            'rpm': [3000.0]
        })
        original_columns = input_df.columns.to_list()

        # Act
        result_df = identify_anomalies(input_df)

        # Assert
        assert input_df.columns.to_list() == original_columns
        assert 'high_temp' not in input_df.columns
        assert 'low_efficiency' not in input_df.columns
        assert 'high_rpm' not in input_df.columns
        assert 'high_temp' in result_df.columns
        assert 'low_efficiency' in result_df.columns
        assert 'high_rpm' in result_df.columns

    def test_identify_anomalies_empty_dataframe(self) -> None:
        """Test anomaly identification with an empty DataFrame."""
        # Arrange
        input_df = pd.DataFrame({
            'engine_temp_f': [],
            'fuel_consumption_mpg': [],
            'rpm': []
        })

        # Act
        result_df = identify_anomalies(input_df)

        # Assert
        assert len(result_df) == 0
        assert 'high_temp' in result_df.columns
        assert 'low_efficiency' in result_df.columns
        assert 'high_rpm' in result_df.columns

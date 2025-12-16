# Automotive Data Dashboard

A lightweight Streamlit app to process, analyze, and visualize automotive data.

## Key Features

- **Data Processing**: Clean, transform, and aggregate automotive data with pandas
- **Interactive Visualizations**: Charts and tables with filters and time-range selectors
- **Real-time Analytics**: Monitor vehicle performance, fuel consumption, and engine metrics
- **Anomaly Detection**: Identify potential issues with vehicle performance

## Tech Stack

- **pandas**: Data manipulation and analysis
- **plotly**: Interactive charts and visualizations
- **streamlit**: Web application framework

## Installation

1. Clone the repository:
```bash
git clone https://github.com/johannesocean/test-repo.git
cd test-repo
```

2. Create a virtual environment with Python 3.12:
```bash
uv venv --python 3.12
```

3. Activate the virtual environment:
```bash
# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

4. Install dependencies:
```bash
uv sync --extra dev
```

## Running the App

```bash
streamlit run ./src/app.py
```

The app will open in your default browser at `http://localhost:8501`.

## Project Layout

```
.
├── .github/              # CI workflows
│   └── workflows/
│       └── ci.yml
├── src/
│   ├── app.py           # Main Streamlit application
│   ├── utils.py         # Utility functions
│   ├── pages/           # Additional Streamlit pages (future)
│   ├── processing/      # Data processing modules
│   │   ├── __init__.py
│   │   └── transform.py
│   └── data/            # Sample data
│       └── automotive_data.csv
├── pyproject.toml       # Project dependencies
└── README.md
```

## Data

The app includes sample automotive data (`src/data/automotive_data.csv`) with the following fields:

- `timestamp`: Date and time of the reading
- `vehicle_id`: Unique vehicle identifier
- `speed_mph`: Vehicle speed in miles per hour
- `fuel_consumption_mpg`: Fuel consumption in miles per gallon
- `engine_temp_f`: Engine temperature in Fahrenheit
- `rpm`: Engine RPM
- `distance_miles`: Cumulative distance traveled
- `location`: Vehicle location
- `status`: Vehicle status (normal/idle)

## Features

### Filters
- **Vehicle Selection**: Filter by specific vehicles
- **Location Filter**: Filter by geographic location
- **Date Range**: Select custom date ranges for analysis

### Visualizations
- Speed and performance over time
- Fuel efficiency metrics
- Engine temperature monitoring
- RPM distribution
- Vehicle comparison charts
- Anomaly detection

### Data Export
- Download filtered data as CSV
- Time-based aggregation (hourly/daily)

## License

MIT
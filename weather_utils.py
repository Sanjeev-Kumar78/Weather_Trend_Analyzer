# Description: This file contains utility functions for the weather data.
import os
import googlemaps
import requests
import datetime as dt
import pandas as pd
import ast
import csv
import tempfile


def get_location(location, API_KEY=None):
    gmaps = googlemaps.Client(key=API_KEY)
    geocordinates = gmaps.geocode(location)[0]['geometry']['location']
    coordinates = (geocordinates['lat'], geocordinates['lng'])
    return coordinates


def load_data(latitude, longitude, start_date, end_date, frequency):

    start_date = str(start_date)
    end_date = str(end_date)
    file_name = f"{str(latitude)}-{str(longitude)};{start_date}-{end_date}.csv"
    # Data\Hourly\24.217924-82.62766;2023-12-01-2023-12-31.csv
    file_path = os.path.join("Data", frequency.title(), file_name)

    # Check if data already exists
    # print(f"Checking for data at {file_path}...")
    if required_data_exists(latitude, longitude, start_date, end_date, frequency):
        return extract_required_data(latitude, longitude, start_date, end_date, frequency)
    else:
        url = "https://archive-api.open-meteo.com/v1/archive?"
        # Fetch data using Open Meteo if not available locally
        # print(f"Fetching data for {latitude,longitude} from {start_date} to {end_date}...")
        if frequency == 'hourly':
            hourly_variables = [
                "temperature_2m",
                "relative_humidity_2m",
                "dewpoint_2m",
                "apparent_temperature",
                "precipitation",
                "rain",
                "snowfall",
                "snow_depth",
                "weather_code",
                "pressure_msl",
                "surface_pressure",
                "cloud_cover",
                "cloud_cover_low",
                "cloud_cover_mid",
                "cloud_cover_high",
                "et0_fao_evapotranspiration",
                "vapour_pressure_deficit",
                "wind_speed_10m",
                "wind_speed_100m",
                "wind_direction_10m",
                "wind_direction_100m",
                "wind_gusts_10m",
                "soil_temperature_0_to_7cm",
                "soil_temperature_7_to_28cm",
                "soil_temperature_28_to_100cm",
                "soil_temperature_100_to_255cm",
                "soil_moisture_0_to_7cm",
                "soil_moisture_7_to_28cm",
                "soil_moisture_28_to_100cm",
                "soil_moisture_100_to_255cm"
            ]
            url += f"latitude={latitude}&longitude={longitude}&start_date={
                start_date}&end_date={end_date}&hourly="
            url += ",".join(hourly_variables)
            url += "&timezone=GMT&models=best_match"  # GMT is the default timezone
            try:
                data = requests.get(url).json()
                df = get_to_csv(data, latitude, longitude, start_date,
                                end_date, frequency, file_path)  # Save the fetched data
                print(f"Data saved to {file_path}")
                return df
            except Exception as e:
                print(f"Error fetching data: {str(e)}")
                return str(e)

        elif frequency == 'daily':
            daily_variables = [
                "weather_code",
                "temperature_2m_max",
                "temperature_2m_min",
                "temperature_2m_mean",
                "apparent_temperature_max",
                "apparent_temperature_min",
                "apparent_temperature_mean",
                "sunrise",
                "sunset",
                "daylight_duration",
                "sunshine_duration",
                "precipitation_sum",
                "rain_sum",
                "snowfall_sum",
                "precipitation_hours",
                "wind_speed_10m_max",
                "wind_gusts_10m_max",
                "wind_direction_10m_dominant",
                "shortwave_radiation_sum",
                "et0_fao_evapotranspiration"
            ]
            url += f"latitude={latitude}&longitude={longitude}&start_date={
                start_date}&end_date={end_date}&daily="
            url += ",".join(daily_variables)
            url += "&timezone=GMT&models=best_match"
            try:
                data = requests.get(url).json()
                df = get_to_csv(data, latitude, longitude, start_date,
                                end_date, frequency, file_path)  # Save the fetched data
                return df
            except Exception as e:
                return str(e)
        else:
            raise ValueError(
                "Invalid frequency. Choose either 'hourly' or 'daily'.")


"""

Function to check if the required data exists in the log.
"""
def required_data_exists(latitude: float, longitude: float, start_date: str, end_date: str, frequency: str) -> bool:
    """
    Check if the required weather data exists in the log file.

    Args:
        latitude: The latitude coordinate
        longitude: The longitude coordinate
        start_date: The start date in YYYY-MM-DD format
        end_date: The end date in YYYY-MM-DD format
        frequency: Data frequency ('hourly' or 'daily')

    Returns:
        bool: True if data exists, False otherwise
    """
    try:
        log_df = pd.read_csv(os.path.join("Data", "log.csv"), index_col=0)

        log_df["start-time"] = pd.to_datetime(log_df["start-time"])
        log_df["end-time"] = pd.to_datetime(log_df["end-time"])
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        exists = (
            (log_df["latitude"] == latitude) &
            (log_df["longitude"] == longitude) &
            (log_df["type"] == frequency) &
            (log_df["start-time"] <= start_dt) &
            (log_df["end-time"] >= end_dt)
        ).any()

        return exists

    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"Error checking data existence: {str(e)}")
        return False


"""
Function to extract the required data from the saved CSV files.
"""


def extract_required_data(latitude, longitude, start_date, end_date, frequency):
    log = pd.read_csv(os.path.join("Data", "log.csv"), index_col=0)
    match = log[
        (log['latitude'] == latitude) &
        (log['longitude'] == longitude) &
        (log['type'] == frequency) &
        (pd.to_datetime(log['start-time']) <= pd.to_datetime(start_date)) &
        (pd.to_datetime(log['end-time']) >= pd.to_datetime(end_date))
    ]

    if match.empty:
        raise ValueError("No matching entry found in log")

    row = match.iloc[0]
    data = pd.read_csv(row['file-path'], index_col=0)

    if row['start-time'] == start_date and row['end-time'] == end_date:
        return data

    column_name = "hourly" if frequency == "hourly" else "daily"
    time_list = ast.literal_eval(data[column_name].iloc[0])

    if frequency == "hourly":
        start_idx = time_list.index(str(start_date) + "T00:00")
        end_idx = time_list.index(str(end_date) + "T23:00") + 1
    else:
        start_idx = time_list.index(start_date)
        end_idx = time_list.index(end_date) + 1

    temp_data = data[column_name].copy()
    print("Extracting data...")
    for i in range(len(temp_data)):
        temp_data.iloc[i] = ast.literal_eval(temp_data.iloc[i])[
            start_idx:end_idx]

    data[column_name] = temp_data
    return data


"""
Function to save the fetched data to a CSV file and update the log
"""


def get_to_csv(data, latitude, longitude, start_date, end_date, frequency, file_path):
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=True)
    
    try:
        log = pd.read_csv(os.path.join("Data", "log.csv"), index_col=0)
    except FileNotFoundError:
        log = pd.DataFrame(columns=["latitude", "longitude", "type", "start-time", "end-time", "file-path"])
    
    log.loc[len(log)] = {
            'latitude': latitude,
            'longitude': longitude,
            'type': frequency,
            'start-time': start_date,
            'end-time': end_date,
            'file-path': file_path
        }
    log.to_csv(os.path.join("Data", "log.csv"))
    return df


"""
Function to get the weather data: For visulaization purposes
"""


def weather_data(df, frequency):
    # Ensure index is included so we can combine row number with variable name
    weather_dictionary = {}

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_file:
        temp_file_path = tmp_file.name
        df.to_csv(temp_file_path, index=True)

    try:
        with open(temp_file_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # skip headers
            for row in reader:
                # Combine the index (row[0]) and the variable name (row[1])
                variable_key = f"{row[0]}"
                values_list = ast.literal_eval(row[-1])
                weather_dictionary[variable_key] = values_list

        # Adjust dictionary if there's a "time" key
        # (you may adapt these conditions to suit your needs)
        if frequency == "hourly":
            for k in list(weather_dictionary.keys()):
                if "time" in k:
                    weather_dictionary["hour"] = [t.split('T')[-1] for t in weather_dictionary[k]]
                    del weather_dictionary[k]
        elif frequency == "daily":
            for k in list(weather_dictionary.keys()):
                if "time" in k:
                    weather_dictionary["date"] = weather_dictionary[k]
                    del weather_dictionary[k]

    finally:
        os.remove(temp_file_path)
    return pd.DataFrame(weather_dictionary)


"""
Function to get the units of the variables (weather data)
"""
def get_units(df):
    units = {}
    for index, row in df.iterrows():
        units[str(index)] = row.iloc[-2]
    return units

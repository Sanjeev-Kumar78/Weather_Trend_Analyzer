import requests
import pandas as pd
import numpy as np
import os
import ast
import googlemaps
from datetime import datetime, date, timedelta

def get_location(location_input, api_key):
    """Convert location string to lat/long coordinates using Google Maps API"""
    try:
        gmaps = googlemaps.Client(key=api_key)
        geocode_result = gmaps.geocode(location_input)
        if not geocode_result:
            return None, None
        location = geocode_result[0]['geometry']['location']
        return location['lat'], location['lng']
    except Exception as e:
        print(f"Error getting location: {e}")
        return None, None

def load_data(latitude, longitude, start_date, end_date, frequency):
    """Load weather data from cache or fetch from API if not available"""
    # Check if data exists in log
    log_path = "Data/log.csv"
    
    try:
        # Ensure directories exist
        os.makedirs("Data/Daily", exist_ok=True)
        os.makedirs("Data/Hourly", exist_ok=True)
        
        # Convert dates to string format if needed - Fix type checking
        if isinstance(start_date, (datetime, date)):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, (datetime, date)):
            end_date = end_date.strftime('%Y-%m-%d')
        elif end_date == "today":
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if os.path.exists(log_path):
            log = pd.read_csv(log_path)
            
            # Check for existing data with exact match
            matches = log[(abs(log['latitude'] - latitude) < 0.01) & 
                         (abs(log['longitude'] - longitude) < 0.01) & 
                         (log['type'] == frequency) & 
                         (log['start-time'] == start_date) & 
                         (log['end-time'] == end_date)]
                         
            if not matches.empty:
                file_path = matches.iloc[0]['file-path']
                df = pd.read_csv(f"Data/{file_path}")
                return df
            
            # If no exact match, check for data that completely covers the requested period
            potential_matches = log[(abs(log['latitude'] - latitude) < 0.01) & 
                                   (abs(log['longitude'] - longitude) < 0.01) & 
                                   (log['type'] == frequency)]
            
            # Convert to datetime for comparison
            potential_matches['start_dt'] = pd.to_datetime(potential_matches['start-time'])
            potential_matches['end_dt'] = pd.to_datetime(potential_matches['end-time'])
            req_start_dt = pd.to_datetime(start_date)
            req_end_dt = pd.to_datetime(end_date)
            
            covering_matches = potential_matches[
                (potential_matches['start_dt'] <= req_start_dt) & 
                (potential_matches['end_dt'] >= req_end_dt)
            ]
            
            if not covering_matches.empty:
                # Use the smallest covering dataset
                file_path = covering_matches.iloc[
                    covering_matches['end_dt'].sub(covering_matches['start_dt']).argmin()
                ]['file-path']
                df = pd.read_csv(f"Data/{file_path}")
                
                # Filter to requested date range
                # This assumes the data has a 'time' column
                return filter_data_to_range(df, start_date, end_date)
        else:
            # Create log file if it doesn't exist
            log = pd.DataFrame(columns=['latitude', 'longitude', 'type', 'start-time', 'end-time', 'file-path'])
            log.to_csv(log_path, index=False)
        
        # If we get here, data doesn't exist or log doesn't exist
        # Fetch from API
        url = build_api_url(latitude, longitude, start_date, end_date, frequency)
        response = requests.get(url)
        
        if response.status_code != 200:
            return f"Error: API returned status code {response.status_code}"
            
        # Process response data
        data = response.json()
        
        # Save data and update log
        df = save_to_csv(data, latitude, longitude, start_date, end_date, frequency)
        return df
        
    except Exception as e:
        return f"Error: {str(e)}"

def build_api_url(latitude, longitude, start_date, end_date, frequency):
    """Build the API URL based on parameters"""
    base_url = "https://archive-api.open-meteo.com/v1/archive?"
    url = f"{base_url}latitude={latitude}&longitude={longitude}&start_date={start_date}&end_date={end_date}"
    
    if frequency == "hourly":
        variables = [
            "temperature_2m", "relative_humidity_2m", "dewpoint_2m",
            "apparent_temperature", "precipitation", "rain", "snowfall",
            "snow_depth", "weather_code", "pressure_msl", "surface_pressure",
            "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
            "et0_fao_evapotranspiration", "vapour_pressure_deficit",
            "wind_speed_10m", "wind_speed_100m", "wind_direction_10m",
            "wind_direction_100m", "wind_gusts_10m", "soil_temperature_0_to_7cm",
            "soil_temperature_7_to_28cm", "soil_temperature_28_to_100cm",
            "soil_temperature_100_to_255cm", "soil_moisture_0_to_7cm",
            "soil_moisture_7_to_28cm", "soil_moisture_28_to_100cm",
            "soil_moisture_100_to_255cm"
        ]
        url += f"&hourly={','.join(variables)}"
    else:  # daily
        variables = [
            "weather_code", "temperature_2m_max", "temperature_2m_min",
            "temperature_2m_mean", "apparent_temperature_max",
            "apparent_temperature_min", "apparent_temperature_mean",
            "sunrise", "sunset", "daylight_duration", "sunshine_duration",
            "precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours",
            "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
            "shortwave_radiation_sum", "et0_fao_evapotranspiration"
        ]
        url += f"&daily={','.join(variables)}"
    
    url += "&timezone=GMT&models=best_match"
    return url

def save_to_csv(data, latitude, longitude, start_date, end_date, frequency):
    """Save data to CSV and update log"""
    try:
        # Convert to DataFrame
        if frequency == "hourly":
            # Extract hourly data
            hourly_data = {
                'time': pd.to_datetime(data['hourly']['time']),
            }
            
            # Add all available hourly variables
            for var in data['hourly']:
                if var != 'time':
                    hourly_data[var] = data['hourly'][var]
            
            df = pd.DataFrame(hourly_data)
            df.set_index('time', inplace=True)
            
        else:  # daily
            # Extract daily data
            daily_data = {
                'time': pd.to_datetime(data['daily']['time']),
            }
            
            # Add all available daily variables
            for var in data['daily']:
                if var != 'time':
                    daily_data[var] = data['daily'][var]
            
            df = pd.DataFrame(daily_data)
            df.set_index('time', inplace=True)
        
        # Add metadata columns
        for meta in ['latitude', 'longitude', 'elevation', 'utc_offset_seconds', 'timezone', 'timezone_abbreviation']:
            if meta in data:
                df[meta] = data[meta]
        
        # Define file path and name
        file_name = f"{latitude:.6f}-{longitude:.6f};{start_date}-{end_date}.csv"
        
        if frequency == "hourly":
            file_path = f"Hourly/{file_name}"
        else:
            file_path = f"Daily/{file_name}"
        
        # Save to CSV
        df.to_csv(f"Data/{file_path}")
        
        # Update log
        log_path = "Data/log.csv"
        log = pd.read_csv(log_path)
        
        new_entry = pd.DataFrame({
            'latitude': [latitude],
            'longitude': [longitude],
            'type': [frequency],
            'start-time': [start_date],
            'end-time': [end_date],
            'file-path': [file_path]
        })
        
        log = pd.concat([log, new_entry], ignore_index=True)
        log.to_csv(log_path, index=False)
        
        return df
        
    except Exception as e:
        print(f"Error saving data: {e}")
        raise

def filter_data_to_range(df, start_date, end_date):
    """Filter dataset to the requested date range"""
    try:
        # Convert index to datetime if it's not already
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
            else:
                # Try to parse the index as datetime
                df.index = pd.to_datetime(df.index)
        
        # Filter to the requested date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        filtered_df = df[(df.index >= start_dt) & (df.index <= end_dt)]
        
        return filtered_df
        
    except Exception as e:
        print(f"Error filtering data: {e}")
        return df  # Return original df if filtering fails

def weather_data(df, frequency):
    """Process raw dataframe into a format suitable for analysis"""
    try:
        # Check if we have the expected data structure
        if frequency == 'hourly' and 'temperature_2m' in df.columns:
            # Data is already in the right format
            result_df = df.copy()
            
            # Add hour column for easier plotting
            if isinstance(result_df.index, pd.DatetimeIndex):
                result_df['hour'] = result_df.index
            
        elif frequency == 'daily' and 'temperature_2m_min' in df.columns:
            # Data is already in the right format
            result_df = df.copy()
            
            # Add date column for easier plotting
            if isinstance(result_df.index, pd.DatetimeIndex):
                result_df['date'] = result_df.index
                
        else:
            # Try to extract data from JSON-like format
            if frequency == 'hourly' and 'hourly' in df:
                data = {}
                data['time'] = pd.to_datetime(df['hourly']['time'])
                
                # Extract all hourly variables
                for var in df['hourly']:
                    if var != 'time':
                        data[var] = df['hourly'][var]
                
                result_df = pd.DataFrame(data)
                result_df.set_index('time', inplace=True)
                result_df['hour'] = result_df.index
                
            elif frequency == 'daily' and 'daily' in df:
                data = {}
                data['time'] = pd.to_datetime(df['daily']['time'])
                
                # Extract all daily variables
                for var in df['daily']:
                    if var != 'time':
                        data[var] = df['daily'][var]
                
                result_df = pd.DataFrame(data)
                result_df.set_index('time', inplace=True)
                result_df['date'] = result_df.index
                
            else:
                # Attempt to parse using string columns that might contain arrays
                result_df = parse_string_arrays(df, frequency)
        
        return result_df
        
    except Exception as e:
        print(f"Error processing data: {e}")
        # Return original df if processing fails
        return df

def parse_string_arrays(df, frequency):
    """Parse string columns that might contain arrays from API response"""
    try:
        result_data = {}
        
        # Find column containing time data
        time_col = None
        for col in df.columns:
            if col == 'time' or (isinstance(col, str) and 'time' in col.lower()):
                time_col = col
                break
        
        if time_col is not None:
            time_data = df[time_col].iloc[0]
            if isinstance(time_data, str) and '[' in time_data:
                # Parse array string
                times = ast.literal_eval(time_data)
                result_data['time'] = pd.to_datetime(times)
            
                # Parse other columns that might contain arrays
                for col in df.columns:
                    if col != time_col:
                        val = df[col].iloc[0]
                        if isinstance(val, str) and '[' in val:
                            try:
                                result_data[col] = ast.literal_eval(val)
                            except:
                                continue
            
                # Create dataframe
                result_df = pd.DataFrame(result_data)
                result_df.set_index('time', inplace=True)
                
                # Add hour/date column
                if frequency == 'hourly':
                    result_df['hour'] = result_df.index
                else:
                    result_df['date'] = result_df.index
                
                return result_df
        
        # If we get here, couldn't parse the data
        return df
        
    except Exception as e:
        print(f"Error parsing string arrays: {e}")
        return df

def get_units(df):
    """Extract units from the dataframe"""
    units = {}
    
    try:
        if 'hourly_units' in df.columns:
            hourly_units = df['hourly_units'].iloc[0]
            hourly_vars = df['hourly'].iloc[0]
            
            # Check if these are strings that need parsing
            if isinstance(hourly_units, str) and isinstance(hourly_vars, str):
                hourly_units = ast.literal_eval(hourly_units)
                hourly_vars = ast.literal_eval(hourly_vars)
            
            # Create units dictionary
            units = dict(zip(hourly_vars, hourly_units))
        elif 'daily_units' in df.columns:
            daily_units = df['daily_units'].iloc[0]
            daily_vars = df['daily'].iloc[0]
            
            # Check if these are strings that need parsing
            if isinstance(daily_units, str) and isinstance(daily_vars, str):
                daily_units = ast.literal_eval(daily_units)
                daily_vars = ast.literal_eval(daily_vars)
            
            # Create units dictionary
            units = dict(zip(daily_vars, daily_units))
        else:
            # If units aren't available, provide defaults for common variables
            if 'temperature_2m' in df.columns:
                units['temperature_2m'] = '°C'
            if 'temperature_2m_max' in df.columns:
                units['temperature_2m_max'] = '°C'
            if 'precipitation' in df.columns:
                units['precipitation'] = 'mm'
            if 'precipitation_sum' in df.columns:
                units['precipitation_sum'] = 'mm'
            if 'wind_speed_10m' in df.columns:
                units['wind_speed_10m'] = 'km/h'
            if 'relative_humidity_2m' in df.columns:
                units['relative_humidity_2m'] = '%'
    except Exception as e:
        print(f"Error extracting units: {e}")
    
    return units

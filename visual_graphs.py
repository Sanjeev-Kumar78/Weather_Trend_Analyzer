import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# from windrose import WindroseAxes  # Removed windrose import
import streamlit as st

def plot_temperature(df, freq='hourly'):
    """
    Plots temperature trends based on the specified frequency.

    Parameters:
    df (pd.DataFrame): DataFrame containing weather data.
    freq (str): Frequency of data ('hourly' or 'daily').
    """
    if freq == 'hourly':
        # Line graph for hourly temperature data
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(df['hour'], df['temperature_2m'], label='Temperature °C')
        ax.plot(df['hour'], df['apparent_temperature'], label='Apparent Temperature °C')
        ax.plot(df['hour'], df['dewpoint_2m'], label='Dewpoint °C')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Value')
        ax.set_title('Temperature')
        ax.legend()
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
        plt.tight_layout()  # Adjust layout to fit labels


    elif freq == 'daily':
        # Line graph for daily temperature data
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['date'], df['temperature_2m_max'], label='Max Temperature')
        ax.plot(df['date'], df['temperature_2m_min'], label='Min Temperature')
        ax.plot(df['date'], df['temperature_2m_mean'], label='Mean Temperature')
        ax.plot(df['date'], df['apparent_temperature_max'], label='Apparent Max Temperature')
        ax.plot(df['date'], df['apparent_temperature_min'], label='Apparent Min Temperature')
        ax.plot(df['date'], df['apparent_temperature_mean'], label='Apparent Mean Temperature')
        ax.set_xlabel('Date')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Temperature Trends')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
        plt.tight_layout()  # Adjust layout to fit labels
        ax.legend()

    return fig

def plot_humidity(df, freq='hourly'):
    """
    Plots humidity trends based on the specified frequency.

    Parameters:
    df (pd.DataFrame): DataFrame containing weather data.
    freq (str): Frequency of data ('hourly' or 'daily').
    """
    if freq == 'hourly':
        # Line graph for hourly humidity data
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(df['hour'], df['relative_humidity_2m'], label='Humidity %')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Value')
        ax.set_title('Humidity')
        ax.legend()
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
        plt.tight_layout()  # Adjust layout to fit labels

    elif freq == 'daily':
        # Plot temperature trends
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['date'], df['temperature_2m_max'], label='Max Temperature')
        ax.plot(df['date'], df['temperature_2m_min'], label='Min Temperature')
        ax.plot(df['date'], df['temperature_2m_mean'], label='Mean Temperature')
        ax.plot(df['date'], df['apparent_temperature_max'], label='Apparent Max Temperature')
        ax.plot(df['date'], df['apparent_temperature_min'], label='Apparent Min Temperature')
        ax.plot(df['date'], df['apparent_temperature_mean'], label='Apparent Mean Temperature')
        ax.set_xlabel('Date')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Temperature Trends')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
        plt.tight_layout()  # Adjust layout to fit labels
        ax.legend()
    return fig

def plot_pressure(df, freq='hourly'):
    """
    Plots pressure trends based on the specified frequency.

    Parameters:
    df (pd.DataFrame): DataFrame containing weather data.
    freq (str): Frequency of data ('hourly' or 'daily').
    """
    if freq == 'hourly':
        # Line graphs for hourly pressure data
        fig, ax = plt.subplots(figsize=(15,6))
        ax.plot(df['hour'], df['pressure_msl'], label='Pressure (msl)')
        ax.plot(df['hour'], df['surface_pressure'], label='Surface Pressure (hPa)')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Pressure (hPa)')
        ax.set_title('Pressure')
        ax.legend()
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
        plt.tight_layout()  # Adjust layout to fit labels

    elif freq == 'daily':
        # Plot precipitation patterns
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['date'], df['precipitation_sum'], label='Precipitation Sum')
        ax.plot(df['date'], df['rain_sum'], label='Rain Sum')
        ax.plot(df['date'], df['snowfall_sum'], label='Snowfall Sum')
        ax.set_xlabel('Date')
        ax.set_ylabel('Precipitation (mm)')
        ax.set_title('Precipitation Patterns')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
        plt.tight_layout()  # Adjust layout to fit labels
        ax.legend()
    return fig

def plot_precipitation(df, freq='hourly'):
    """
    Plots precipitation patterns based on the specified frequency.

    Parameters:
    df (pd.DataFrame): DataFrame containing weather data.
    freq (str): Frequency of data ('hourly' or 'daily').
    """
    if freq == 'hourly':
        # Bar chart for hourly precipitation
        fig, ax = plt.subplots(figsize=(15,6))
        ax.bar(df['hour'], df['precipitation'], label='Precipitation')
        ax.bar(df['hour'], df['rain'], label='Rain')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Value')
        ax.set_title('Precipitation')
        ax.legend()
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
        plt.tight_layout()  # Adjust layout to fit labels

    elif freq == 'daily':
        # Plot wind speed and direction
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['date'], df['wind_speed_10m_max'], label='Wind Speed')
        ax.plot(df['date'], df['wind_gusts_10m_max'], label='Wind Gusts')
        ax.set_xlabel('Date')
        ax.set_ylabel('Wind Speed (m/s)')
        ax.set_title('Wind Speed and Direction')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
        plt.tight_layout()  # Adjust layout to fit labels
        ax.legend()
    return fig

def plot_wind_speed(df, freq='hourly'):
    """
    Plots wind speed and related metrics based on the specified frequency.

    Parameters:
    df (pd.DataFrame): DataFrame containing weather data.
    freq (str): Frequency of data ('hourly' or 'daily').
    """
    if freq == 'hourly':
        # Plot wind speed
        fig, ax = plt.subplots(figsize=(15,6))
        ax.plot(df['hour'], df['wind_speed_10m'], label='10m')
        ax.plot(df['hour'], df['wind_speed_100m'], label='100m')
        ax.plot(df['hour'], df['wind_gusts_10m'], label='Gusts')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Wind Speed (km/h)')
        ax.set_title('Wind Speed')
        ax.legend()
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
        plt.tight_layout()  # Adjust layout to fit labels

    elif freq == 'daily':
        # Plot shortwave radiation and evapotranspiration
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['date'], df['shortwave_radiation_sum'], label='Shortwave Radiation')
        ax.plot(df['date'], df['et0_fao_evapotranspiration'], label='Evapotranspiration')
        ax.set_xlabel('Date')
        ax.set_ylabel('Radiation (W/m²) / Evapotranspiration (mm)')
        ax.set_title('Shortwave Radiation and Evapotranspiration')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
        plt.tight_layout()  # Adjust layout to fit labels
        ax.legend()
    return fig

def plot_cloud_cover(df, freq='hourly'):
    """
    Plots cloud cover based on the specified frequency.

    Parameters:
    df (pd.DataFrame): DataFrame containing weather data.
    freq (str): Frequency of data ('hourly' or 'daily').
    """
    if freq == 'hourly':
        # Plot cloud cover
        fig, ax = plt.subplots(figsize=(15,6))
        ax.plot(df['hour'], df['cloud_cover'], label='Cloud Cover (%)')
        ax.plot(df['hour'], df['cloud_cover_low'], label='Low (%)')
        ax.plot(df['hour'], df['cloud_cover_mid'], label='Mid (%)')
        ax.plot(df['hour'], df['cloud_cover_high'], label='High (%)')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Cloud Cover (%)')
        ax.set_title('Cloud Cover')
        ax.legend()
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
        plt.tight_layout()  # Adjust layout to fit labels

    elif freq == 'daily':
        # Plot weather code distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df['weather_code'], bins=7, edgecolor='black')
        ax.set_xlabel('Weather Code')
        ax.set_ylabel('Frequency')
        ax.set_title('Weather Code Distribution')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
        plt.tight_layout()  # Adjust layout to fit labels
    return fig

def plot_evapotranspiration_vapour(df, freq='hourly'):
    """
    Plots evapotranspiration and vapour pressure deficit.

    Parameters:
    df (pd.DataFrame): DataFrame containing weather data.
    freq (str): Frequency of data ('hourly' or 'daily').
    """

    if freq == 'hourly':
        # Plot evapotranspiration and vapour pressure deficit
        fig, ax = plt.subplots(figsize=(15,6))
        ax.plot(df['hour'], df['et0_fao_evapotranspiration'], label='Evapotranspiration (mm)')
        ax.plot(df['hour'], df['vapour_pressure_deficit'], label='Vapour Pressure Deficit (kPa)')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Value')
        ax.set_title('Evapotranspiration and Vapour Pressure Deficit')
        ax.legend()
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
        plt.tight_layout()  # Adjust layout to fit labels

    elif freq == 'daily':
        # Convert sunrise and sunset times to hours
        df['sunrise_hour'] = pd.to_datetime(df['sunrise']).dt.hour + pd.to_datetime(df['sunrise']).dt.minute / 60
        df['sunset_hour'] = pd.to_datetime(df['sunset']).dt.hour + pd.to_datetime(df['sunset']).dt.minute / 60

        # Plot sunrise times
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['date'], df['sunrise_hour'], label='Sunrise')
        ax.set_xlabel('Date')
        ax.set_ylabel('Time (GMT+0)')
        ax.set_title('Sunrise Times')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
        plt.tight_layout()  # Adjust layout to fit labels
        ax.legend()
    return fig

def plot_soil_temperature(df, freq='hourly'):
    """
    Plots soil temperature at different depths.

    Parameters:
    df (pd.DataFrame): DataFrame containing weather data.
    freq (str): Frequency of data ('hourly' or 'daily').
    """
    if freq == 'hourly':
        # Plot soil temperature
        fig, ax = plt.subplots(figsize=(15,6))
        ax.plot(df['hour'], df['soil_temperature_0_to_7cm'], label='0-7cm')
        ax.plot(df['hour'], df['soil_temperature_7_to_28cm'], label='7-28cm')
        ax.plot(df['hour'], df['soil_temperature_28_to_100cm'], label='28-100cm')
        ax.plot(df['hour'], df['soil_temperature_100_to_255cm'], label='100-255cm')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Soil Temperature (°C)')
        ax.set_title('Soil Temperature')
        ax.legend()
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
        plt.tight_layout()  # Adjust layout to fit labels

    elif freq == 'daily':
        # Plot sunset times
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['date'], df['sunset_hour'], label='Sunset')
        ax.set_xlabel('Date')
        ax.set_ylabel('Time (GMT+0)')
        ax.set_title('Sunset Times')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
        plt.tight_layout()  # Adjust layout to fit labels
        ax.legend()
    return fig

def plot_soil_moisture(df, freq='hourly'):
    """
    Plots soil moisture at different depths.

    Parameters:
    df (pd.DataFrame): DataFrame containing weather data.
    freq (str): Frequency of data ('hourly' or 'daily').
    """
    if freq == 'hourly':
        # Plot soil moisture
        fig, ax = plt.subplots(figsize=(15,6))
        ax.plot(df['hour'], df['soil_moisture_0_to_7cm'], label='0-7cm')
        ax.plot(df['hour'], df['soil_moisture_7_to_28cm'], label='7-28cm')
        ax.plot(df['hour'], df['soil_moisture_28_to_100cm'], label='28-100cm')
        ax.plot(df['hour'], df['soil_moisture_100_to_255cm'], label='100-255cm')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Soil Moisture (%)')
        ax.set_title('Soil Moisture')
        ax.legend()
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
        plt.tight_layout()  # Adjust layout to fit labels

    elif freq == 'daily':
        # Plot daylight duration
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['date'], df['daylight_duration'] / 3600, label='Daylight Duration')
        ax.set_xlabel('Date')
        ax.set_ylabel('Duration (hours)')
        ax.set_title('Daylight Duration')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
        plt.tight_layout()  # Adjust layout to fit labels
        ax.legend()
    return fig
        
def plot_sunshine_duration(df, freq='daily'):
    if freq == 'daily':
        # Plot sunshine duration
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['sunshine_duration'])
        ax.set_title('Sunshine Duration')
        ax.set_xlabel('Day')
        ax.set_ylabel('Duration (seconds)')
        ax.axhline(y=df['sunshine_duration'].mean(), color='r', linestyle='--', label='Average')
        ax.legend()
        return fig

def plot_sunset_time(df, freq='daily'):
    if freq == 'daily':
        # Plot sunset time
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['sunset'])
        ax.set_title('Sunset Time')
        ax.set_xlabel('Day')
        ax.set_ylabel('Time (hours)')
        ax.axhline(y=df['sunset'].mean(), color='r', linestyle='--', label='Average')
        ax.legend()
        return fig

def plot_weather_code(df, freq='daily'):
    if freq == 'daily':
        # Plot weather code
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['weather_code'])
        ax.set_title('Weather Code')
        ax.set_xlabel('Day')
        ax.set_ylabel('Weather Code')
        ax.legend()
        return fig

def plot_correlation_matrix(df, freq = 'hourly'):
    """
    Plots the correlation matrix of the selected weather variables.

    Parameters:
    df (pd.DataFrame): DataFrame containing weather data.
    """
    if freq == 'hourly':
        # Correlation analysis
        corr_matrix = df[['temperature_2m', 'relative_humidity_2m', 'dewpoint_2m', 'apparent_temperature', 'precipitation', 'rain', 'snowfall', 'snow_depth', 'surface_pressure', 'cloud_cover', 'et0_fao_evapotranspiration', 'vapour_pressure_deficit', 'wind_speed_10m', 'wind_speed_100m', 'wind_direction_10m', 'wind_direction_100m', 'wind_gusts_10m', 'soil_temperature_0_to_7cm', 'soil_temperature_7_to_28cm', 'soil_temperature_28_to_100cm', 'soil_temperature_100_to_255cm', 'soil_moisture_0_to_7cm', 'soil_moisture_7_to_28cm', 'soil_moisture_28_to_100cm', 'soil_moisture_100_to_255cm']].corr()
        print("Correlation Matrix:")
        print(corr_matrix)

        # Heatmap of correlation matrix
        fig, ax = plt.subplots(figsize=(30,10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Matrix')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
        plt.tight_layout()  # Adjust layout to fit labels

    elif freq == 'daily':
        # Correlation analysis
        corr_matrix = df[["temperature_2m_max",
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
                "et0_fao_evapotranspiration"]].corr()
        print("Correlation Matrix:")
        print(corr_matrix)

        # Heatmap of correlation matrix
        fig, ax = plt.subplots(figsize=(30,10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Matrix')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
        plt.tight_layout()  # Adjust layout to fit labels
    return fig
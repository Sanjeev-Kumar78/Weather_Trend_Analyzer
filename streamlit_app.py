# app.py
import streamlit as st
# -------------------------------------------------------------------
import numpy as np
import weather_utils as wu
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import seasonal_decompose



def main():
    st.title('Weather Data Dashboard')

    with st.form('User Input'):
        frequency = st.radio('Select Frequency: (Hourly Data or Daily Data)', [
                             'hourly', 'daily'])
        date_range = [st.date_input('Start Date', value=None, max_value='today', format='YYYY-MM-DD'),
                      st.date_input('End Date', value="today", max_value='today', format='YYYY-MM-DD')]
        location = st.text_input(
            'Enter Location:', placeholder='City, State, Country', autocomplete='on')
        submitted = st.form_submit_button('Submit')

        if submitted:
            with st.spinner('Loading data...'):
                # Fetch Coordinates using Google Maps API
                @st.cache_data
                def fetch_coordinates(location, API_KEY):
                    return wu.get_location(location, API_KEY)
                latitude, longitude = fetch_coordinates(
                    location, st.secrets['GOOGLE_MAPS_API_KEY'])

                # Load data using the weather_utils module
                @st.cache_data
                def load_data(latitude, longitude, start_date, end_date, frequency):
                    return wu.load_data(latitude=latitude, longitude=longitude, start_date=start_date, end_date=end_date, frequency=frequency)
                df = load_data(latitude, longitude,
                               date_range[0], date_range[1], frequency)
            if isinstance(df, str):
                st.error(f'Error: {df}')
                return  # Exit the function if data loading fails
            st.success('Data loaded successfully!')
            # Display the first few rows of the data
            # st.write(df.head())

            # Weather_data for analysis
            weather_data = wu.weather_data(df, frequency)

            with st.expander('Terms related to the dataset'):
                if frequency == 'hourly':
                    st.markdown("""### Hourly Variables
| Variable                            | Impact on Weather or Climate                                                                                                           |
|-------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| Temperature (2 m)                   | Directly affects evaporation rates, air density, and influences overall weather patterns and climate.                                   |
| Relative Humidity (2 m)             | Determines moisture in the air, influencing precipitation potential and perceived comfort levels.                                       |
| Dewpoint (2 m)                      | Indicates moisture saturation; when reached, it leads to cloud formation and potential precipitation.                                   |
| Apparent Temperature                | Reflects perceived temperature by combining actual temperature and humidity, affecting how organisms experience weather.                |
| Precipitation (rain + snow)         | Supplies water to ecosystems, influences soil moisture, and drives hydrological processes.                                              |
| Rain                                | Critical for agriculture, water resources, and vegetation growth, affecting local and regional climates.                                |
| Snowfall                            | Contributes to snowpack and water reserves, with implications for winter sports and ecosystem health.                                   |
| Snow depth                          | Influences road conditions, agriculture, and overall water availability upon melting.                                                   |
| Weather code                        | Summarizes prevailing conditions, offering quick reference to weather patterns and potential climate impacts.                           |
| Sealevel Pressure                   | Affects wind formation and large-scale weather systems, driving movements of air masses.                                                 |
| Surface Pressure                    | Influences local wind patterns and can signal approaching weather systems or changes.                                                    |
| Cloud cover Total                   | Modulates solar radiation, temperature, and can indicate the likelihood of precipitation.                                                |
| Cloud cover Low                     | Influences near-surface temperatures and local precipitation events.                                                                    |
| Cloud cover Mid                     | Affects weather system development and mid-level atmospheric conditions.                                                                 |
| Cloud cover High                    | Impacts radiation balance and can signal approaching large-scale weather changes.                                                        |
| Reference Evapotranspiration (ET₀) | Guides water management in agriculture, indicating how much water crops need based on atmospheric conditions.                            |
| Vapour Pressure Deficit             | Shows the atmosphere’s drying power; higher values increase plant water demand and evaporation.                                          |
| Wind Speed (10 m)                   | Affects local weather conditions, air dispersion, and can impact transportation and outdoor activities.                                 |
| Wind Speed (100 m)                  | Important for wind energy assessments and forecasting broader weather systems.                                                           |
| Wind Direction (10 m)               | Drives local weather patterns and pollution dispersion.                                                                                 |
| Wind Direction (100 m)              | Influences large-scale weather systems and is relevant to aviation.                                                                     |
| Wind Gusts (10 m)                   | Can cause sudden changes in conditions, affecting outdoor safety and infrastructure.                                                    |
| Soil Temperature (0-7 cm)           | Impacts seed germination, microbial activity, and plant emergence.                                                                      |
| Soil Temperature (7-28 cm)          | Influences root development and nutrient cycling in deeper soil layers.                                                                 |
| Soil Temperature (28-100 cm)        | Affects growth of deep-rooted plants, moderating soil processes over time.                                                              |
| Soil Temperature (100-255 cm)       | Plays a role in groundwater recharge and long-term soil and climatic interactions.                                                      |
| Soil Moisture (0-7 cm)              | Critical for surface runoff, plant water availability, and soil erosion processes.                                                      |
| Soil Moisture (7-28 cm)             | Influences root uptake, plant stress, and evaporation rates.                                                                            |
| Soil Moisture (28-100 cm)           | Affects water availability for deeper root systems and soil water storage capacity.                                                     |
| Soil Moisture (100-255 cm)          | Helps maintain long-term moisture reserves, impacting groundwater and broader climate feedbacks.                                        |
""")
                elif frequency == 'daily':
                    st.markdown("""## Daily Variables
| Variable                            | Impact on Weather or Climate                                                                                                           |
|-------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| Weather code                        | Describes the overall weather conditions, aiding in forecasting and climate analysis.                                                 |
| Maximum Temperature (2 m)           | Indicates the highest temperature reached, affecting heatwaves and energy consumption.                                               |
| Minimum Temperature (2 m)           | Indicates the lowest temperature achieved, impacting frost formation and agriculture.                                                |
| Mean Temperature (2 m)              | Represents the average temperature, crucial for climate studies and trend analysis.                                                   |
| Maximum Apparent Temperature (2 m)  | Reflects the highest perceived temperature, combining actual temperature and humidity, affecting human comfort.                        |
| Minimum Apparent Temperature (2 m)  | Reflects the lowest perceived temperature, influencing human comfort and heating requirements.                                        |
| Mean Apparent Temperature (2 m)     | Represents the average perceived temperature, important for assessing overall comfort levels.                                         |
| Sunrise                             | Marks the start of daylight, influencing daily cycles and solar energy availability.                                                  |
| Sunset                              | Marks the end of daylight, affecting daily cycles and solar energy planning.                                                           |
| Daylight Duration                   | Duration of daylight affects photosynthesis, energy consumption, and daily activities.                                                 |
| Sunshine Duration                   | Amount of sunshine influences solar power generation, plant growth, and UV exposure.                                                 |
| Precipitation Sum                   | Total precipitation is vital for water resource management, agriculture, and flood forecasting.                                         |
| Rain Sum                            | Total rainfall affects agriculture, water supply, and flood risks.                                                                     |
| Snowfall Sum                        | Total snowfall impacts water reserves, transportation, and winter activities.                                                          |
| Precipitation Hours                 | Number of hours with precipitation affects soil moisture and flood risk assessment.                                                   |
| Maximum Wind Speed (10 m)           | Affects wind energy potential, weather severity, and structural safety.                                                                |
| Maximum Wind Gusts (10 m)           | Indicates extreme wind events, impacting building safety and outdoor activities.                                                        |
| Dominant Wind Direction (10 m)      | Influences weather patterns, pollution dispersion, and maritime navigation.                                                             |
| Shortwave Radiation Sum             | Total incoming solar radiation impacts temperature regulation, photosynthesis, and energy balance.                                    |
| Reference Evapotranspiration (ET₀)  | Guides irrigation planning and water resource management by estimating atmospheric demand for moisture.                                   |""")

            with st.expander('Data Summary'):

                # Display basic statistics
                st.write(weather_data.describe())
            with st.sidebar.expander("Units"):
                # Display table of units
                units = wu.get_units(df)
                markdown_text = '| Variable | Unit |\n'
                markdown_text += '|----------|------|\n'
                for key, value in units.items():
                    markdown_text += f'| {key} | {value} |\n'
                st.markdown(markdown_text)

            if frequency == 'hourly':
                # Plot temperature
                fig = px.line(weather_data, x=weather_data['hour'], y=['temperature_2m', 'apparent_temperature', 'dewpoint_2m'],
                              title='Hourly Temperature Trends',
                              labels={'value': 'Temperature (°C)', 'hour': 'Time GMT+0', 'variable': 'Temperature Type'})
                fig.update_layout(legend_title_text='')
                st.plotly_chart(fig)

                # Plot humidity
                fig = px.line(weather_data, x=weather_data['hour'], y=['relative_humidity_2m'],
                              title='Hourly Humidity Trends', labels={'value': 'Humidity (%)', 'hour': 'Time GMT+0'})
                fig.update_layout(legend_title_text='')
                st.plotly_chart(fig)

                # Plot pressure
                fig = px.line(weather_data, x=weather_data['hour'], y=['pressure_msl', 'surface_pressure'],
                              title='Hourly Pressure Trends',
                              labels={'value': 'Pressure (hPa)', 'hour': 'Time GMT+0', 'variable': 'Pressure Type'})
                fig.update_layout(legend_title_text='')
                st.plotly_chart(fig)

                # Plot precipitation
                fig = px.bar(weather_data, x=weather_data['hour'], y='precipitation',
                             title='Hourly Precipitation', labels={'precipitation': 'Precipitation (mm)', 'hour': 'Time GMT+0'})
                st.plotly_chart(fig)

                # Plot wind speed
                fig = px.line(weather_data, x=weather_data['hour'], y=['wind_speed_10m', 'wind_gusts_10m'],
                              title='Hourly Wind Speed and Gusts', labels={'value': 'Speed (km/h)', 'hour': 'Time GMT+0'})
                fig.update_layout(legend_title_text='')
                st.plotly_chart(fig)

                # Plot cloud cover
                fig = px.line(weather_data, x=weather_data['hour'], y=['cloud_cover', 'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high'],
                              title='Hourly Cloud Cover Trends',
                              labels={'value': 'Cloud Cover (%)', 'hour': 'Time GMT+0', 'variable': 'Cloud Cover Type'})
                fig.update_layout(legend_title_text='')
                st.plotly_chart(fig)

            elif frequency == 'daily':
                # Plot temperature trends
                fig = px.line(weather_data, x=weather_data['date'], y=['temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean'],
                              title='Daily Temperature Trends',
                              labels={'value': 'Temperature (°C)', 'date': 'Date', 'variable': 'Temperature Type'})
                fig.update_layout(legend_title_text='')
                st.plotly_chart(fig)

                # Plot precipitation patterns
                fig = px.bar(weather_data, x=weather_data['date'], y=['precipitation_sum', 'rain_sum', 'snowfall_sum'],
                             title='Daily Precipitation Summary',
                             labels={'value': 'Precipitation (mm)', 'date': 'Date', 'variable': 'Precipitation Type'})
                st.plotly_chart(fig)

                # Plot wind speed and direction
                fig = px.line(weather_data, x=weather_data['date'], y=['wind_speed_10m_max', 'wind_gusts_10m_max'],
                              title='Daily Maximum Wind Speed and Gusts',
                              labels={'value': 'Speed (km/h)', 'date': 'Date', 'variable': 'Wind Type'})
                fig.update_layout(legend_title_text='')
                st.plotly_chart(fig)

                # Plot shortwave radiation and evapotranspiration
                fig = px.line(weather_data, x=weather_data['date'], y=['shortwave_radiation_sum', 'et0_fao_evapotranspiration'],
                              title='Daily Shortwave Radiation and Evapotranspiration',
                              labels={'value': 'Radiation (W/m²) / Evapotranspiration (mm)', 'date': 'Date', 'variable': 'Variable'})
                fig.update_layout(legend_title_text='')
                st.plotly_chart(fig)

                # Convert sunrise and sunset times to hours
                weather_data['sunrise_hour'] = pd.to_datetime(weather_data['sunrise']).dt.hour + pd.to_datetime(weather_data['sunrise']).dt.minute / 60
                weather_data['sunset_hour'] = pd.to_datetime(weather_data['sunset']).dt.hour + pd.to_datetime(weather_data['sunset']).dt.minute / 60

                # Plot sunrise and sunset times
                fig = px.line(weather_data, x=weather_data['date'], y=['sunrise_hour', 'sunset_hour'],
                            title='Daily Sunrise and Sunset Times',
                            labels={'value': 'Time (GMT+0)', 'date': 'Date', 'variable': 'Time Type'})
                fig.update_layout(legend_title_text='')
                st.plotly_chart(fig)

                # Plot daylight duration
                fig = px.line(weather_data, x=weather_data['date'], y=['daylight_duration'],
                            title='Daily Daylight Duration',
                            labels={'value': 'Duration (seconds)', 'date': 'Date'})
                fig.update_layout(legend_title_text='')
                st.plotly_chart(fig)

                # Plot sunshine duration
                fig = px.line(weather_data, x=weather_data['date'], y=['sunshine_duration'],
                            title='Daily Sunshine Duration',
                            labels={'value': 'Duration (seconds)', 'date': 'Date'})
                fig.update_layout(legend_title_text='')
                st.plotly_chart(fig)


                # Plot weather code distribution
                fig = px.bar(weather_data, x=weather_data['date'], y=['weather_code'],
                            title='Daily Weather Code Distribution',
                            labels={'weather_code': 'Weather Code', 'date': 'Date'})
                st.plotly_chart(fig)

            # Plot correlation matrix
            numeric_columns = weather_data.select_dtypes(include=[np.number]).columns
            corr_matrix = weather_data[numeric_columns].corr()
            fig, ax = plt.subplots(figsize=(30,20))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
            ax.set_title('Correlation Matrix')
            ax.tick_params(axis='x', rotation=45) # Rotate x-axis labels
            st.pyplot(fig)

            # Regression Analysis Section
            st.markdown("### Regression Analysis")
            st.write("This analysis predicts evapotranspiration based on key weather parameters.")
            
            if frequency == 'daily':
                X = weather_data[['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'wind_speed_10m_max']]
                y = weather_data['et0_fao_evapotranspiration']
                st.write("Using daily maximum temperature, minimum temperature, precipitation sum, and maximum wind speed as predictors.")
            else:  # hourly
                X = weather_data[['temperature_2m', 'relative_humidity_2m', 'precipitation', 'wind_speed_10m']]
                y = weather_data['et0_fao_evapotranspiration']
                st.write("Using temperature, humidity, precipitation, and wind speed as predictors.")

             

            # Weather Pattern Clustering
            st.markdown("### Weather Pattern Clustering")
            st.write("This analysis groups similar weather patterns into clusters.")

            if frequency == 'daily':
                features = ['temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean',
                           'apparent_temperature_max', 'precipitation_sum', 'wind_speed_10m_max']
            else:  # hourly
                features = ['temperature_2m', 'relative_humidity_2m', 'dewpoint_2m',
                           'apparent_temperature', 'precipitation', 'wind_speed_10m']
            
            # Perform clustering
            X = weather_data[features]
            X_scaled = StandardScaler().fit_transform(X)
            kmeans = KMeans(n_clusters=4, random_state=42)
            weather_data['cluster'] = kmeans.fit_predict(X_scaled)
            
            # Plot clusters
            if frequency == 'daily':
                fig = px.scatter(weather_data, x='temperature_2m_max', y='precipitation_sum', color='cluster',
                                title='Weather Clusters: Temperature vs Precipitation',
                                labels={'temperature_2m_max': 'Maximum Temperature (°C)',
                                       'precipitation_sum': 'Total Precipitation (mm)'})
            else:  # hourly
                fig = px.scatter(weather_data, x='temperature_2m', y='relative_humidity_2m', color='cluster',
                                title='Weather Clusters: Temperature vs Humidity',
                                labels={'temperature_2m': 'Temperature (°C)',
                                       'relative_humidity_2m': 'Relative Humidity (%)'})
            st.plotly_chart(fig)

            # Seasonal Decomposition
            st.markdown("### Seasonal Pattern Analysis")
            st.write("Breaking down the temperature pattern into trend, seasonal, and random components.")

            if frequency == 'daily':
                temp_col = 'temperature_2m_mean'
                period = 365  # yearly seasonality
                title_suffix = "Daily Mean Temperature"
            else:  # hourly
                temp_col = 'temperature_2m'
                period = 24  # daily seasonality
                title_suffix = "Hourly Temperature"
            
            result = seasonal_decompose(weather_data[temp_col], model='additive', period=period)
            fig, axes = plt.subplots(4, 1, figsize=(12, 12))
            result.observed.plot(ax=axes[0], title=f'Observed {title_suffix}')
            result.trend.plot(ax=axes[1], title='Long-term Trend')
            result.seasonal.plot(ax=axes[2], title='Seasonal Pattern')
            result.resid.plot(ax=axes[3], title='Random Fluctuations')
            plt.tight_layout()
            st.pyplot(fig)

            


if __name__ == '__main__':
    main()
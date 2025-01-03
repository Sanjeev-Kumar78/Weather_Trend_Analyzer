# app.py
import streamlit as st
# -------------------------------------------------------------------
import weather_utils as wu
import visual_graphs as vg


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

            # Visualization selection
                plot_type = st.sidebar.selectbox(
                    'Select Plot Type',
                    ['Temperature', 'Humidity', 'Pressure', 'Precipitation',
                     'Wind Speed', 'Cloud Cover', 'Evapotranspiration and Vapour', 'Soil Temperature', 'Soil Moisture','Correlation Matrix', 'Sunrise and Sunset Times', 'Sunshine and Daylight Duration']
                )

                # Display the selected plot
                st.header(f'{plot_type} Visualization')

                with st.container():
                    if plot_type == 'Temperature':
                        fig = vg.plot_temperature(weather_data, frequency)
                        st.pyplot(fig)

                    elif plot_type == 'Humidity':
                        fig = vg.plot_humidity(weather_data, frequency)
                        st.pyplot(fig)

                    elif plot_type == 'Pressure':
                        fig = vg.plot_pressure(weather_data, frequency)
                        st.pyplot(fig)

                    elif plot_type == 'Precipitation':
                        fig = vg.plot_precipitation(weather_data, frequency)
                        st.pyplot(fig)

                    elif plot_type == 'Wind Speed':
                        fig = vg.plot_wind_speed(weather_data, frequency)
                        st.pyplot(fig)

                    elif plot_type == 'Cloud Cover':
                        fig = vg.plot_cloud_cover(weather_data, frequency)
                        st.pyplot(fig)

                    elif plot_type == 'Evapotranspiration and Vapour':
                        fig = vg.plot_evapotranspiration_vapour(weather_data, frequency)
                        st.pyplot(fig)

                    elif plot_type == 'Soil Temperature':
                        fig = vg.plot_soil_temperature(weather_data, frequency)
                        st.pyplot(fig)

                    elif plot_type == 'Soil Moisture':
                        fig = vg.plot_soil_moisture(weather_data, frequency)
                        st.pyplot(fig)

                    elif plot_type == 'Correlation Matrix':
                        fig = vg.plot_correlation_matrix(weather_data, frequency)
                        st.pyplot(fig)

                    elif plot_type == 'Sunrise and Sunset Times':
                        if frequency == 'hourly':
                            st.warning("Sunrise and Sunset Times plot is only available for daily frequency.")
                        else:
                            fig = vg.plot_soil_temperature(weather_data, frequency)
                            st.pyplot(fig)
                            fig = vg.plot_evapotranspiration_vapour(weather_data, frequency)
                            st.pyplot(fig)

                    elif plot_type == 'Sunshine and Daylight Duration':
                        if frequency == 'hourly':
                            st.warning("Sunshine and Daylight Duration plot is only available for daily frequency.")
                        else:
                            fig = vg.plot_sunshine_duration(weather_data, frequency)
                            st.pyplot(fig)
                            fig = vg.plot_soil_moisture(weather_data, frequency)
                            st.pyplot(fig)

                # Advanced Analysis (currently commented out)
                # if st.sidebar.checkbox('Show Cluster Analysis'):
                #     st.header('Cluster Analysis')
                #     st.plotly_chart(plot_cluster_analysis(df))

                # if st.sidebar.checkbox('Show Seasonal Decomposition'):
                #     st.header('Seasonal Decomposition')
                #     st.pyplot(plot_seasonal_decomposition(df, frequency))

if __name__ == '__main__':
    main()
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
import weather_utils as wu

# Initialize session state for persisting data across reruns
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = None
if 'clusters' not in st.session_state:
    st.session_state.clusters = None
if 'corr_matrix' not in st.session_state:
    st.session_state.corr_matrix = None
# Add date strings to session state
if 'start_date_str' not in st.session_state:
    st.session_state.start_date_str = None
if 'end_date_str' not in st.session_state:
    st.session_state.end_date_str = None

# Page configuration and app intro
st.set_page_config(
    page_title="Weather Trend Analyzer",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title('Weather Trend Analyzer')
st.markdown("""
Analyze historical weather patterns for any location around the world.
Select your preferred frequency (hourly or daily), date range, and location to get started.
""")

# Sidebar input with max range limitation for hourly data
st.sidebar.header("Input Parameters")
frequency = st.sidebar.radio('Select Data Frequency:', ['hourly', 'daily'])
# Define max days for hourly frequency
MAX_HOURLY_DAYS = 37

st.sidebar.subheader("Date Range")
today = pd.Timestamp.today().date()
col1, col2 = st.sidebar.columns(2)
with col1:
    if frequency == 'hourly':
        earliest_date = today - pd.Timedelta(days=MAX_HOURLY_DAYS)
        start_date = st.date_input('Start Date', value=today - pd.Timedelta(days=7),
                                   min_value=earliest_date, max_value=today)
        st.caption(f"Hourly data limited to {MAX_HOURLY_DAYS} days maximum")
    else:
        start_date = st.date_input('Start Date', value=today - pd.Timedelta(days=30), max_value=today)
with col2:
    end_date = st.date_input('End Date', value=today, min_value=start_date, max_value=today)

# Show a warning if dates are not valid
if start_date > end_date:
    st.sidebar.warning("Start date must be before end date. Adjusting end date...")
    end_date = start_date

# Check if selected range exceeds maximum allowed for hourly data
if frequency == 'hourly':
    selected_days = (end_date - start_date).days + 1
    if selected_days > MAX_HOURLY_DAYS:
        st.sidebar.error(f"⚠️ Selected range ({selected_days} days) exceeds maximum allowed for hourly data ({MAX_HOURLY_DAYS} days). Please select a shorter range.")

st.sidebar.subheader("Location")
location_method = st.sidebar.radio("Select input method:", ["City/Location Name", "Latitude/Longitude"])
if location_method == "City/Location Name":
    location = st.sidebar.text_input('Enter Location:', placeholder='City, State, Country')
    latitude, longitude = None, None
else:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        latitude = st.number_input('Latitude', value=0.0, format="%.6f", step=0.000001)
    with col2:
        longitude = st.number_input('Longitude', value=0.0, format="%.6f", step=0.000001)
    location = None

with st.sidebar.form("submit_form"):
    submit_button = st.form_submit_button('Analyze Weather Data', type="primary")

# Main content area - only executes the heavy operations when form is submitted
if submit_button:
    with st.spinner('Fetching and analyzing weather data...'):
        try:
            if frequency == 'hourly' and ((end_date - start_date).days + 1) > MAX_HOURLY_DAYS:
                st.error(f"Selected range exceeds maximum allowed for hourly data. Please select a shorter range.")
                st.stop()

            # Get coordinates if location name was provided
            if location_method == "City/Location Name":
                latitude, longitude = wu.get_location(location, st.secrets['GOOGLE_MAPS_API_KEY'])
                if not latitude or not longitude:
                    st.error("Could not find coordinates for the given location.")
                    st.stop()
                st.success(f"Coordinates: {latitude}, {longitude}")

            # Convert date objects to strings before passing to API
            start_date_str = start_date.strftime('%Y-%m-%d') if start_date else None
            end_date_str = end_date.strftime('%Y-%m-%d') if end_date != "today" else "today"

            # Store in session state for later use
            st.session_state.start_date_str = start_date_str
            st.session_state.end_date_str = end_date_str
            df = wu.load_data(latitude, longitude, start_date_str, end_date_str, frequency)
            if isinstance(df, str) and df.startswith("Error"):
                st.error(df)
                st.stop()
            st.session_state.weather_data = wu.weather_data(df, frequency)
            st.success('Data loaded successfully!')
            st.write(f"Analyzing weather data for coordinates: {latitude}, {longitude}")
            st.write(f"Date range: {start_date} to {end_date}")
            numeric_cols = st.session_state.weather_data.select_dtypes(include=[np.number]).columns
            st.session_state.corr_matrix = st.session_state.weather_data[numeric_cols].corr()
            if frequency == 'hourly':
                features = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'cloud_cover', 'precipitation']
            else:
                features = ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'wind_speed_10m_max']
            features = [f for f in features if f in st.session_state.weather_data.columns]
            if len(features) >= 2:
                cluster_data = st.session_state.weather_data[features].dropna()
                if len(cluster_data) >= 8:
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(cluster_data)
                    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(scaled_data)
                    st.session_state.clusters = {'data': cluster_data, 'labels': cluster_labels, 'features': features}
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)

# Create tabs for different visualizations if data is loaded
if st.session_state.weather_data is not None:
    weather_data = st.session_state.weather_data  # Use the data from session state
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Basic Trends", 
        "Advanced Analysis",
        "Correlation Matrix", 
        "Weather Patterns",
        "Data Summary"
    ])
    
    with tab1:
        st.header("Basic Weather Trends")
        
        if 'hour' in weather_data.columns:  # hourly data
            # Temperature trends
            st.subheader("Temperature Analysis")
            st.markdown("""
            This chart shows the temperature variation over time. Compare actual, apparent (feels-like), 
            and dewpoint temperatures to understand thermal comfort and moisture conditions.
            """)
            
            # Check which columns exist and have valid data
            valid_cols = []
            for col in ['temperature_2m', 'apparent_temperature', 'dewpoint_2m']:
                if col in weather_data.columns:
                    # Make sure the column has data and matches the index length
                    if len(weather_data[col]) == len(weather_data.index):
                        valid_cols.append(col)
            
            if valid_cols:
                try:
                    # Create a dedicated dataframe for plotting to ensure alignment
                    plot_df = pd.DataFrame(index=weather_data.index)
                    for col in valid_cols:
                        plot_df[col] = weather_data[col]
                    
                    temp_fig = px.line(
                        plot_df,
                        labels={'value': 'Temperature (°C)', 'x': 'Time', 'variable': 'Measurement'},
                        title="Hourly Temperature Trends",
                        color_discrete_map={
                            'temperature_2m': '#FF5733',
                            'apparent_temperature': '#33A2FF',
                            'dewpoint_2m': '#33FF57'
                        }
                    )
                    temp_fig.update_layout(legend_title_text='', hovermode='x unified')
                    st.plotly_chart(temp_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error plotting temperature data: {str(e)}")
            else:
                st.warning("No temperature data available for plotting")
            
            # Humidity and pressure combined - Check if columns exist
            st.subheader("Humidity & Atmospheric Pressure")
            st.markdown("""
            This chart compares relative humidity and surface pressure. Pressure changes often precede 
            weather changes, while humidity affects how temperatures feel and precipitation likelihood.
            """)
            if 'relative_humidity_2m' in weather_data.columns and 'surface_pressure' in weather_data.columns:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(
                    go.Scatter(x=weather_data.index, y=weather_data['relative_humidity_2m'], 
                              name='Humidity', line=dict(color='#33A2FF')),
                    secondary_y=False
                )
                fig.add_trace(
                    go.Scatter(x=weather_data.index, y=weather_data['surface_pressure'], 
                              name='Surface Pressure', line=dict(color='#FF5733')),
                    secondary_y=True
                )
                fig.update_layout(
                    title_text='Humidity and Pressure Trends',
                    hovermode='x unified'
                )
                fig.update_yaxes(title_text='Humidity (%)', secondary_y=False)
                fig.update_yaxes(title_text='Pressure (hPa)', secondary_y=True)
                st.plotly_chart(fig, use_container_width=True)
            
            # Precipitation visualization - Check if column exists
            st.subheader("Precipitation Analysis")
            st.markdown("""
            This chart displays hourly precipitation amounts. Taller bars indicate heavier rainfall or snowfall,
            helping identify storm intensity and duration.
            """)
            if 'precipitation' in weather_data.columns:
                precip_fig = px.bar(weather_data, 
                    x=weather_data.index, 
                    y='precipitation',
                    labels={'precipitation': 'Precipitation (mm)', 'x': 'Time'},
                    title="Hourly Precipitation"
                )
                precip_fig.update_traces(marker_color='#3366FF')
                st.plotly_chart(precip_fig, use_container_width=True)
            
            # Add wind direction visualization - Check if columns exist
            st.subheader("Wind Direction Analysis")
            st.markdown("""
            This wind rose shows the frequency distribution of wind directions. Longer segments indicate 
            prevailing wind directions, which influence temperature patterns and air mass movements.
            """)
            if 'wind_direction_10m' in weather_data.columns and 'wind_speed_10m' in weather_data.columns:
                # Create a wind rose figure using plotly
                wind_dir = weather_data['wind_direction_10m']
                wind_speed = weather_data['wind_speed_10m']
                
                # Convert direction to categories
                dir_labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
                
                try:
                    weather_data['dir_cat'] = pd.cut(
                        wind_dir, 
                        bins=[-1, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 360],
                        labels=dir_labels
                    )
                    
                    # Count occurrences of each direction
                    wind_counts = weather_data['dir_cat'].value_counts().reindex(dir_labels).fillna(0)
                    
                    # Create a polar bar chart
                    fig = px.bar_polar(
                        r=wind_counts.values,
                        theta=wind_counts.index,
                        title="Wind Direction Distribution",
                        color_discrete_sequence=px.colors.sequential.Plasma_r
                    )
                    fig.update_layout(polar=dict(radialaxis=dict(showticklabels=False)))
                    st.plotly_chart(fig, use_container_width=True)

                    # Additional wind speed by direction analysis
                    st.subheader("Wind Speed by Direction")
                    st.markdown("""
                    This polar scatter plot displays wind speed by direction. Points further from center indicate 
                    stronger winds, with colors showing intensity. Useful for identifying dominant wind patterns.
                    """)
                    wind_data = pd.DataFrame({
                        'direction': weather_data['wind_direction_10m'],
                        'speed': weather_data['wind_speed_10m']
                    })
                    
                    fig = px.scatter_polar(
                        wind_data, 
                        r='speed', 
                        theta='direction',
                        color='speed',
                        title="Wind Speed and Direction",
                        color_continuous_scale=px.colors.sequential.Viridis
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating wind plots: {str(e)}")
            
            # Other hourly visualizations
            # ...existing code for hourly visualizations...
        
        else:  # daily data
            # Daily temperature extremes
            st.subheader("Temperature Analysis")
            st.markdown("""
            This chart displays daily temperature extremes and averages. The gap between max and min temperatures 
            shows daily temperature range, reflecting local climate characteristics and seasonal patterns.
            """)
            temp_fig = px.line(weather_data, 
                x=weather_data.index, 
                y=['temperature_2m_max', 'temperature_2m_mean', 'temperature_2m_min'],
                labels={'value': 'Temperature (°C)', 'x': 'Date', 'variable': 'Measurement'},
                title="Daily Temperature Extremes",
                color_discrete_map={
                    'temperature_2m_max': '#FF5733',    # Warm red for max
                    'temperature_2m_mean': '#FFAA33',   # Orange for mean
                    'temperature_2m_min': '#33A2FF'     # Cool blue for min
                }
            )
            temp_fig.update_layout(legend_title_text='', hovermode='x unified')
            st.plotly_chart(temp_fig, use_container_width=True)
            
            # Temperature range visualization (high-low)
            st.subheader("Daily Temperature Range")
            st.markdown("""
            This filled-area chart highlights the daily temperature range between maximum and minimum values.
            Wider bands indicate days with greater temperature fluctuation, often related to clear skies or seasonal transitions.
            """)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=weather_data.index,
                y=weather_data['temperature_2m_max'],
                fill=None,
                mode='lines',
                line_color='#FF5733',
                name='Max Temperature'
            ))
            fig.add_trace(go.Scatter(
                x=weather_data.index,
                y=weather_data['temperature_2m_min'],
                fill='tonexty',
                mode='lines',
                line_color='#33A2FF',
                name='Min Temperature'
            ))
            fig.update_layout(
                title="Daily Temperature Range",
                yaxis_title="Temperature (°C)",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Precipitation analysis
            st.subheader("Precipitation Analysis")
            st.markdown("""
            This grouped bar chart breaks down daily precipitation into total, rain, and snowfall components.
            Compare the types and amounts to identify precipitation patterns and seasonal weather events.
            """)
            precip_fig = px.bar(weather_data, 
                x=weather_data.index, 
                y=['precipitation_sum', 'rain_sum', 'snowfall_sum'],
                labels={'value': 'Amount (mm/cm)', 'x': 'Date', 'variable': 'Type'},
                title="Daily Precipitation Summary",
                barmode='group',
                color_discrete_map={
                    'precipitation_sum': '#3366FF',  # Blue
                    'rain_sum': '#33A2FF',           # Light blue
                    'snowfall_sum': '#AAAAFF'        # Very light blue
                }
            )
            precip_fig.update_layout(legend_title_text='', hovermode='x unified')
            st.plotly_chart(precip_fig, use_container_width=True)
            
            # Define soil temperature columns before using them
            soil_cols = [col for col in weather_data.columns if 'soil_temperature' in col]
            
            # Add soil temperature analysis if data available
            if soil_cols:
                st.subheader("Soil Temperature Analysis")
                st.markdown("""
                This chart shows soil temperature at different depths. Deep soil temperatures change more slowly 
                than surface layers, providing insight into long-term thermal trends and agricultural conditions.
                """)
                soil_fig = px.line(
                    weather_data,
                    x=weather_data.index,
                    y=soil_cols,
                    title="Soil Temperature at Different Depths",
                    labels={'value': 'Temperature (°C)', 'variable': 'Depth'}
                )
                soil_fig.update_layout(legend_title_text='', hovermode='x unified')
                st.plotly_chart(soil_fig, use_container_width=True)
            
            # Add sunrise/sunset visualization for daily data
            if 'sunrise' in weather_data.columns and 'sunset' in weather_data.columns:
                st.subheader("Daylight Analysis")
                st.markdown("""
                This chart shows sunrise and sunset times with the daylight period highlighted in yellow.
                Track the changing day length across seasons and its impact on temperature and sunlight availability.
                """)
                
                try:
                    # Convert sunrise and sunset to datetime
                    sunrise_times = pd.to_datetime(weather_data['sunrise'])
                    sunset_times = pd.to_datetime(weather_data['sunset'])
                    
                    # Extract hours as floats for plotting
                    weather_data['sunrise_hour'] = sunrise_times.dt.hour + sunrise_times.dt.minute / 60
                    weather_data['sunset_hour'] = sunset_times.dt.hour + sunset_times.dt.minute / 60
                    
                    # Create the daylight visualization
                    fig = go.Figure()
                    
                    # Add sunrise line
                    fig.add_trace(go.Scatter(
                        x=weather_data.index, 
                        y=weather_data['sunrise_hour'],
                        mode='lines+markers',
                        name='Sunrise',
                        line=dict(color='orange')
                    ))
                    
                    # Add sunset line
                    fig.add_trace(go.Scatter(
                        x=weather_data.index, 
                        y=weather_data['sunset_hour'],
                        mode='lines+markers',
                        name='Sunset',
                        line=dict(color='navy')
                    ))
                    
                    # Fill the area between sunrise and sunset
                    fig.add_trace(go.Scatter(
                        x=weather_data.index.tolist() + weather_data.index.tolist()[::-1],
                        y=weather_data['sunrise_hour'].tolist() + weather_data['sunset_hour'].tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(255, 200, 0, 0.2)',
                        line=dict(color='rgba(255, 255, 255, 0)'),
                        hoverinfo='skip',
                        showlegend=False
                    ))
                    
                    fig.update_layout(
                        title='Sunrise and Sunset Times (GMT+0)',
                        yaxis=dict(
                            title='Time of Day (GMT+0)',
                            tickvals=[0, 3, 6, 9, 12, 15, 18, 21, 24],
                            ticktext=['12 AM', '3 AM', '6 AM', '9 AM', '12 PM', '3 PM', '6 PM', '9 PM', '12 AM']
                        ),
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add daylight duration visualization
                    if 'daylight_duration' in weather_data.columns:
                        st.subheader("Daylight Duration")
                        st.markdown("""
                        This chart shows the total hours of daylight each day. The trend reflects 
                        seasonal changes, with implications for plant growth, energy consumption, and human activity patterns.
                        """)
                        # Convert seconds to hours for better readability
                        weather_data['daylight_hours'] = weather_data['daylight_duration'] / 3600
                        
                        fig = px.line(
                            weather_data, 
                            x=weather_data.index, 
                            y='daylight_hours',
                            labels={'daylight_hours': 'Hours', 'x': 'Date'},
                            title="Daily Daylight Duration"
                        )
                        fig.update_layout(hovermode='x unified')
                        st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error plotting sunrise/sunset data: {str(e)}")
                    st.exception(e)
            
            # Other daily visualizations
            # ...existing code for daily visualizations...
    
    with tab2:
        st.header("Advanced Analysis")
        
        # Weather Code Distribution
        st.subheader("Weather Code Distribution")
        st.markdown("""
        This chart shows the frequency of different weather conditions during the selected period.
        Higher bars indicate more common weather patterns, helping identify the predominant conditions.
        """)
        
        # Create a mapping for weather codes
        wmo_codes = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Fog",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            61: "Light rain",
            63: "Moderate rain",
            65: "Heavy rain",
            71: "Light snow",
            73: "Moderate snow",
            75: "Heavy snow",
            77: "Snow grains",
            80: "Light rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            95: "Thunderstorm"
        }
        
        weather_code_col = 'weather_code' # Same column name for both hourly and daily data
        
        if weather_code_col in weather_data.columns:
            # Count occurrences of each weather code
            code_counts = weather_data[weather_code_col].value_counts().reset_index()
            code_counts.columns = ['code', 'count']
            
            # Add weather code descriptions
            code_counts['description'] = code_counts['code'].map(
                lambda x: wmo_codes.get(x, f"Unknown code {x}")
            )
            
            # Create horizontal bar chart
            fig = px.bar(
                code_counts,
                y='description',
                x='count',
                orientation='h',
                color='code',
                title='Distribution of Weather Conditions',
                labels={'count': 'Frequency', 'description': 'Weather Condition'}
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Handle seasonal decomposition properly
        st.subheader("Seasonal Pattern Analysis")
        st.markdown("""
        This decomposition breaks down temperature data into three components:
        - **Trend**: Long-term direction (warming, cooling)
        - **Seasonal**: Repeating patterns (daily or weekly cycles)
        - **Residual**: Random fluctuations and anomalies
        
        Use this to understand underlying patterns beyond day-to-day variations.
        """)
        st.write("Breaking down the temperature pattern into trend, seasonal, and random components.")
        
        if 'hour' in weather_data.columns:
            temp_col = 'temperature_2m'
            period = 24  # daily seasonality
            title_suffix = "Hourly Temperature"
        else:  # daily
            temp_col = 'temperature_2m_mean'
            period = 7  # weekly seasonality
            title_suffix = "Daily Mean Temperature"
        
        # Check if enough data and the column exists
        if temp_col in weather_data.columns and len(weather_data) > period * 2:
            try:
                # Make a clean copy for decomposition with numeric index
                decomp_data = weather_data[temp_col].copy()
                decomp_data.index = range(len(decomp_data))  # Use simple numeric index
                
                # Handle missing values
                decomp_data = decomp_data.dropna()
                
                if len(decomp_data) > period * 2:  # Re-check length after dropping NAs
                    from statsmodels.tsa.seasonal import seasonal_decompose
                    result = seasonal_decompose(decomp_data, model='additive', period=period)
                    
                    # Create the plot
                    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
                    
                    # Plot observed
                    axes[0].plot(result.observed)
                    axes[0].set_title('Observed')
                    axes[0].set_xticklabels([])
                    
                    # Plot trend
                    axes[1].plot(result.trend)
                    axes[1].set_title('Trend')
                    axes[1].set_xticklabels([])
                    
                    # Plot seasonal
                    axes[2].plot(result.seasonal)
                    axes[2].set_title('Seasonal')
                    axes[2].set_xticklabels([])
                    
                    # Plot residual
                    axes[3].plot(result.resid)
                    axes[3].set_title('Residual')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning(f"Not enough non-missing data points for seasonal decomposition. Need at least {period * 2} data points.")
            except Exception as e:
                st.error(f"Could not perform seasonal decomposition: {str(e)}")
        else:
            st.warning(f"Not enough data for seasonal decomposition. Need at least {period * 2} data points.")
        
        # Add sunrise/sunset visualization in the Advanced Analysis tab for daily data
        if 'date' in weather_data.columns and 'sunrise' in weather_data.columns and 'sunset' in weather_data.columns:
            st.subheader("Daylight Pattern Analysis")
            st.markdown("""
            This chart tracks sunrise and sunset time trends. The converging or diverging lines indicate
            increasing or decreasing daylight hours, reflecting seasonal progression and latitude effects.
            """)
            
            try:
                # Convert sunrise and sunset to datetime
                sunrise_times = pd.to_datetime(weather_data['sunrise'])
                sunset_times = pd.to_datetime(weather_data['sunset'])
                
                # Extract hours as floats for plotting
                weather_data['sunrise_hour'] = sunrise_times.dt.hour + sunrise_times.dt.minute / 60
                weather_data['sunset_hour'] = sunset_times.dt.hour + sunset_times.dt.minute / 60
                
                # Create the visualization
                fig = go.Figure()
                
                # Add sunrise scatter points with hover info
                fig.add_trace(go.Scatter(
                    x=weather_data.index, 
                    y=weather_data['sunrise_hour'],
                    mode='lines+markers',
                    name='Sunrise',
                    line=dict(color='orange'),
                    hovertemplate='Date: %{x}<br>Sunrise: %{y:.2f} hours'
                ))
                
                # Add sunset scatter points with hover info  
                fig.add_trace(go.Scatter(
                    x=weather_data.index, 
                    y=weather_data['sunset_hour'],
                    mode='lines+markers',
                    name='Sunset',
                    line=dict(color='navy'),
                    hovertemplate='Date: %{x}<br>Sunset: %{y:.2f} hours'
                ))
                
                fig.update_layout(
                    title='Sunrise and Sunset Times Trend',
                    xaxis_title='Date',
                    yaxis_title='Hour of Day (GMT+0)',
                    yaxis=dict(
                        tickvals=[0, 3, 6, 9, 12, 15, 18, 21, 24],
                        ticktext=['12 AM', '3 AM', '6 AM', '9 AM', '12 PM', '3 PM', '6 PM', '9 PM', '12 AM']
                    ),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error plotting sunrise/sunset trends: {str(e)}")
    
    with tab3:
        st.header("Correlation Matrix")
        st.markdown("""
        This heatmap shows relationships between weather variables. Strong positive correlations (near +1) 
        appear in dark red, strong negative correlations (near -1) in dark blue, and no correlation (near 0) in white.
        Use this to identify which weather parameters tend to move together or in opposite directions.
        """)
        
        # Use pre-computed correlation matrix from session state
        corr_matrix = st.session_state.corr_matrix
        
        # Allow user to filter by feature count but don't trigger rerun
        num_features = st.slider(
            "Number of features to display", 
            5, min(15, len(corr_matrix)), 10, 
            key='corr_matrix_features'
        )
        
        # Filter correlation matrix by variance
        if len(corr_matrix.columns) > num_features:
            var_series = weather_data[corr_matrix.columns].var()
            top_vars = var_series.sort_values(ascending=False).head(num_features)
            selected_cols = top_vars.index
            filtered_matrix = corr_matrix.loc[selected_cols, selected_cols]
        else:
            filtered_matrix = corr_matrix
        
        # Create heatmap with plotly
        fig = px.imshow(filtered_matrix,
                       labels=dict(x="Feature", y="Feature", color="Correlation"),
                       x=filtered_matrix.columns,
                       y=filtered_matrix.columns,
                       color_continuous_scale=px.colors.diverging.RdBu_r,
                       zmin=-1, zmax=1)
        
        fig.update_layout(
            title='Correlation Matrix of Weather Variables',
            width=800,
            height=800
        )
        
        # Add correlation values as text annotations
        for i in range(len(filtered_matrix.columns)):
            for j in range(len(filtered_matrix.columns)):
                fig.add_annotation(
                    x=filtered_matrix.columns[i],
                    y=filtered_matrix.columns[j],
                    text=f"{filtered_matrix.iloc[j, i]:.2f}",
                    showarrow=False,
                    font=dict(color="black" if abs(filtered_matrix.iloc[j, i]) < 0.7 else "white")
                )
        
        st.plotly_chart(fig)
    
    with tab4:
        st.header("Weather Pattern Clustering")
        st.markdown("""
        This analysis groups similar weather conditions into distinct patterns or "clusters".
        Points with the same color represent similar weather states, helping identify recurring
        weather regimes despite varying individual measurements.
        """)
        
        # Use pre-computed clusters from session state
        if st.session_state.clusters:
            clusters = st.session_state.clusters
            features = clusters['features']
            cluster_data = clusters['data']
            cluster_labels = clusters['labels']
            
            # Add cluster column to data
            cluster_data_with_labels = cluster_data.copy()
            cluster_data_with_labels['cluster'] = cluster_labels
            
            # Let user select features for the scatter plot
            col1, col2 = st.columns(2)
            with col1:
                x_feature = st.selectbox('X-axis feature', features, key='x_feature')
            with col2:
                y_feature = st.selectbox('Y-axis feature', features, 
                                       index=min(1, len(features)-1), key='y_feature')
            
            # Create cluster visualization
            fig = px.scatter(
                cluster_data_with_labels, 
                x=x_feature, 
                y=y_feature,
                color='cluster',
                title=f'Weather Pattern Clusters: {x_feature} vs {y_feature}',
                labels={
                    x_feature: x_feature,
                    y_feature: y_feature,
                    'cluster': 'Weather Pattern'
                },
                color_continuous_scale=px.colors.qualitative.Bold
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show cluster characteristics
            st.subheader("Weather Pattern Characteristics")
            st.markdown("""
            This table shows the average values for each weather variable within each cluster.
            Compare these characteristics to understand the typical conditions associated with each pattern.
            """)
            cluster_means = cluster_data_with_labels.groupby('cluster').mean()
            st.write(cluster_means)
            
            # Fix calendar heatmap for daily data
            if 'date' in weather_data.columns:  # Only for daily data
                # Create a proper time series with cluster labels
                time_clusters = pd.DataFrame({
                    'date': cluster_data_with_labels.index,
                    'cluster': cluster_data_with_labels['cluster']
                })
                
                # Ensure date column is datetime type
                time_clusters['date'] = pd.to_datetime(time_clusters['date'])
                
                # Create calendar-like visualization safely
                if len(time_clusters) >= 14:  # Only show if we have enough data
                    # Add week and weekday columns for calendar visualization without using dt accessor directly
                    time_clusters['week'] = [d.isocalendar()[1] for d in time_clusters['date']]
                    time_clusters['weekday'] = [d.strftime('%A') for d in time_clusters['date']]
                    
                    # Create a pivot table for the calendar heatmap
                    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    
                    # Manual pivot to avoid issues
                    pivot_data = {}
                    for week in time_clusters['week'].unique():
                        pivot_data[week] = {}
                        for day in weekdays:
                            matches = time_clusters[(time_clusters['week'] == week) & (time_clusters['weekday'] == day)]
                            if len(matches) > 0:
                                pivot_data[week][day] = matches['cluster'].iloc[0]
                            else:
                                pivot_data[week][day] = np.nan
                                
                    # Convert to DataFrame
                    heatmap_data = pd.DataFrame(pivot_data).T
                    
                    # Create heatmap
                    fig = px.imshow(
                        heatmap_data,
                        labels=dict(x="Day of Week", y="Week Number", color="Pattern"),
                        title="Weekly Weather Pattern Distribution",
                        color_continuous_scale=px.colors.qualitative.Bold
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("""
                    This calendar view shows which weather pattern was dominant on each day of the week.
                    Look for recurring patterns across weekdays or weekends, and how patterns evolve over time.
                    """)
        else:
            st.warning("Clustering not available. Not enough data or features for clustering.")
    
    # Add data summary tab
    with tab5:
        st.header("Data Summary")
        st.markdown("""
        This tab provides a statistical overview of the weather data, helping you understand
        the range, central tendency, and variability of different weather parameters.
        """)
        
        # Display dataset info
        st.subheader("Dataset Information")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Number of Records", len(weather_data))
            
            
            min_date = weather_data.index.min()
            if hasattr(min_date, 'strftime'):
                min_date_str = min_date.strftime('%Y-%m-%d')
            else:
                # If it's not a datetime, convert to string directly
                min_date_str = str(min_date)
            st.metric("Start Date", min_date_str)
        
        with col2:
            st.metric("Number of Variables", len(weather_data.columns))
            
            max_date = weather_data.index.max()
            if hasattr(max_date, 'strftime'):
                max_date_str = max_date.strftime('%Y-%m-%d')
            else:
                max_date_str = str(max_date)
            st.metric("End Date", max_date_str)
        
        # Summary statistics
        st.subheader("Summary Statistics")
        summary_stats = weather_data.describe().T
        summary_stats = summary_stats.round(2)  # Round for better display
        
        # Add variable descriptions
        if 'hour' in weather_data.columns:  # hourly data
            key_vars = {
                'temperature_2m': 'Air temperature at 2m above ground',
                'relative_humidity_2m': 'Relative humidity at 2m above ground',
                'precipitation': 'Total precipitation (rain, showers, snow) in mm',
                'wind_speed_10m': 'Wind speed at 10m above ground'
            }
        else:  # daily data
            key_vars = {
                'temperature_2m_min': 'Minimum daily air temperature at 2m above ground',
                'temperature_2m_max': 'Maximum daily air temperature at 2m above ground',
                'precipitation_sum': 'Total daily precipitation in mm',
                'daylight_duration': 'Total daylight duration in seconds'
            }
        
        # Display key variables that exist in the data
        st.write("Key Variables:")
        available_keys = [k for k in key_vars.keys() if k in summary_stats.index]
        if available_keys:
            selected_stats = summary_stats.loc[available_keys]
            selected_stats.insert(0, 'Description', [key_vars[idx] for idx in available_keys])
            st.dataframe(selected_stats, use_container_width=True)
        else:
            st.write("No key variables available in this dataset.")
        
        # Show the full statistics with an expander
        with st.expander("See full statistics for all variables"):
            st.dataframe(summary_stats, use_container_width=True)
        
        # Show sample of raw data
        st.subheader("Raw Data Sample")
        st.dataframe(weather_data.head(10), use_container_width=True)
        
        # Option to download data
        csv = weather_data.to_csv()
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name=f"weather_data_{st.session_state.start_date_str}_to_{st.session_state.end_date_str}.csv",
            mime="text/csv",
        )

    st.markdown("---")
    
    st.markdown("### Citations")
    st.markdown("""
    - Zippenfenig, P. (2023). *Open-Meteo.com Weather API* [Computer software]. Zenodo. https://doi.org/10.5281/ZENODO.7970649
    - Hersbach, H., Bell, B., Berrisford, P., et al. (2023). *ERA5 hourly data on single levels from 1940 to present* [Data set]. ECMWF. https://doi.org/10.24381/cds.adbb2d47
    - Muñoz Sabater, J. (2019). *ERA5-Land hourly data from 2001 to present* [Data set]. ECMWF. https://doi.org/10.24381/CDS.E2161BAC
    """)

else:
    # Display initial information when app first loads
    st.info("Please select your parameters and click 'Analyze Weather Data' to begin.")
    
    # Show some sample visualizations or information about the app
    st.subheader("About This Tool")
    st.markdown("""
    This Weather Trend Analyzer helps you:
    
    - Visualize historical weather patterns
    - Compare temperature, precipitation, wind, and other variables
    - Identify seasonal trends using statistical analysis
    - Discover correlations between different weather parameters
    - Group similar weather patterns using machine learning
    
    The data is sourced from the Open-Meteo historical weather API, which provides accurate 
    historical weather information for any location worldwide.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://images.unsplash.com/photo-1592210454359-9043f067919b?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80", 
                caption="Analyze weather patterns with interactive visualizations.")
    with col2:
        st.image("https://images.unsplash.com/photo-1580193769210-b8d1c049a7d9?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80", 
                caption="Compare weather variables across different time periods.")

st.markdown("---")
st.markdown(
    "<div style='text-align: center;'>"
    "Made with ❤️ by <a href='https://github.com/Sanjeev-Kumar78' target='_blank'>Sanjeev Kumar</a>"
    "</div>",
    unsafe_allow_html=True
)
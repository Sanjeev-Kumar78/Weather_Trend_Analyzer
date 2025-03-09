# Weather Trend Analyzer

This dashboard enables you to analyze historical weather patterns using hourly and daily data from the Open-Meteo historical weather API.

## Features
- Visualize temperature, humidity, precipitation, wind, and cloud cover trends.
- Perform correlation and seasonal decomposition analyses.
- Cluster similar weather patterns.
- Interactive visualizations with Plotly and Streamlit.
- Integrated Q&A on weather data using a retrieval-augmented generation (RAG) module.

## Prerequisites
- Python 3.7 or later
- Required packages (see `requirements.txt`)
- A valid **Google Maps API Key** to resolve location names.
- Internet access to fetch weather data from the Open-Meteo API.

## Usage
1. Clone the repository `git clone https://www.github.com/Sanjeev-Kumar78/Weather_Trend_Analyzer`.
2. Install dependencies:
   ```bash
   cd Weather_Trend_Analyzer
   pip install -r requirements.txt
   ```
3. Set your Google Maps API key in the `st.secrets` or as an environment variable.
4. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```
5. On the sidebar:
   - Choose the data frequency (hourly or daily).
   - Select the date range. (For hourly data the range is limited to a maximum of 37 days.)
   - Input a location (as text or latitude/longitude).
6. Click **Analyze Weather Data** and explore the various visualizations and analyses provided.

## Data Sources & Citations
- Weather data is sourced from [Open-Meteo API](https://open-meteo.com/).
- Additional datasets: ERA5 hourly data, ERA5-Land.
- Refer to the citations section in the app for detailed credits.

## License
This project is open source under the MIT License.

---

Made with ❤️ by [Sanjeev Kumar](https://github.com/Sanjeev-Kumar78)

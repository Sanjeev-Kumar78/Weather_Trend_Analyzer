# 🌦️Weather Trend Analyzer (Work-in-progress)

<!-- Badges -->

## Overview
This project analyzes the weather data of a city, as per the user's choice, and shows the trend on various parameters like temperature, humidity, cloudiness, and wind speed over a period of time. It also shows the trend over the years, illustrating how the weather has been changing.

## Tools Used
- Python
- Jupyter Notebook
- Pandas
- Matplotlib
- GoogleMaps API (for geocoding)
- [Open-Meteo](https://open-meteo.com/)

## Usage
1. Clone the repository.
2. Install the required packages using the following command:
```bash
pip install -r requirements.txt
```
3. Create an `.env` file in the root directory and add the following variables:
   ```bash
   GOOGLE_MAPS_API_KEY=<Your Google Maps API Key> # Get the API Key from Google Cloud Platform
4. Run the `WeatherTrendAnalyzer.ipynb` file in Jupyter Notebook.
5. Select the Form of Analysis on data whether it is on Hourly or Daily type.
   1. Hourly: The data will be fetched on an hourly basis of each day from the start date till the end date.
      1. Hourly data is beneficial when you want to analyze the weather trend of a city on an hourly basis for a less number of days.
   2. Daily: The data will be fetched on a daily basis from the start date till the end date.
      1. Daily data is beneficial when you want to analyze the weather trend of a city on a daily basis for a large number of days or over a years of trend.
6. Enter the city name for which you want to analyze the weather trend.
7. Enter the Start and End dates for which you want to analyze the weather trend.
8. The weather trend will be displayed in the form of graphs.

## Future Improvements
- The project can be extended to show the weather trend of multiple cities.
- The project can be integrated with a chatbot to get the weather trend of a city by just asking the chatbot. Using the RAG model, the chatbot can be trained to answer questions related to weather.

## Citations
- Zippenfenig, P. (2023). *Open-Meteo.com Weather API* [Computer software]. Zenodo. https://doi.org/10.5281/ZENODO.7970649
- Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horányi, A., Muñoz Sabater, J., Nicolas, J., Peubey, C., Radu, R., Rozum, I., Schepers, D., Simmons, A., Soci, C., Dee, D., Thépaut, J-N. (2023). *ERA5 hourly data on single levels from 1940 to present* [Data set]. ECMWF. https://doi.org/10.24381/cds.adbb2d47
- Muñoz Sabater, J. (2019). *ERA5-Land hourly data from 2001 to present* [Data set]. ECMWF. https://doi.org/10.24381/CDS.E2161BAC
- Schimanke S., Ridal M., Le Moigne P., Berggren L., Undén P., Randriamampianina R., Andrea U., Bazile E., Bertelsen A., Brousseau P., Dahlgren P., Edvinsson L., El Said A., Glinton M., Hopsch S., Isaksson L., Mladek R., Olsson E., Verrelle A., Wang Z.Q. (2021). *CERRA sub-daily regional reanalysis data for Europe on single levels from 1984 to present* [Data set]. ECMWF. https://doi.org/10.24381/CDS.622A565A

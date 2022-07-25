# seattlecrimeforecast.com
 This is a private research project using crime history data collected from Seattle Open Data and weather history / forecast data from Visual Crossing. This data is fed into a TensorFlow neural network to predict crime potential rates (where a police report is filed) for the next several days in Seattle neighborhoods.

# Prerequisites: 
Uses a forked Folium. Install this fork with pip install -e git+https://github.com/justinjconroy/folium.git#egg=folium 

Weather data is downloaded from visualcrossings.com server which requires an app key. Register for an account and then create appids.py in root dir and set WEATHER = app key

# Training a new TensorFlow model
Delete or rename existing crime_model directory in root and then run crime_trainer.py

Training features:
#Feature 0  LAT
#Feature 1  LNG
#Feature 2  Day of Week (sin)
#Feature 3  Day of Week (cos)
#Feature 4  Day of year (sin)
#Feature 5  Day of year (cos)
#feature 6  year
#feature 7  Time of day (sin)
#feature 8  Time of day (cos)
#feature 9  Total Against Property Crime last 28 days for that police beat
#feature 10  Total Against Person Crime last 28 days for that police beat
#feature 11  Total Against Society Crime last 28 days for that police beat
#feature 12 Temperature
#feature 13 rain amount 
#feature 14 cloud cover %
#feature 15 snow

Labels (outputs):
#Label 1  Amount of against property crime
#Label 2 Amount of against person crome
#Label 3 Amount of against society


# Create crime forecast with existing model
Run crime_forecast.py
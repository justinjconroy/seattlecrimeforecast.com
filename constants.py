MYWEBPATH = "Html/Archive/"
RSSPATH = "Html/"

RECORDLIMIT = 50000				#50,000 is max number of records that Seattle Open API will return for a single request
DAYSTEP = 21       				#Number of crime days to store in one json file archive locally
COMPLETEMINDAYS = 60			#seattle police upload records still being uploaded in last week so need to redownload later

TIMEBLOCKNUM = 24 				#Number of time blocks per day
TIMEBLOCKLEN = 1 				#Length of time blocks (hr) 

WEEKDAYSMAPPING = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
TIMEDAYMAPPING = ("Late Night", "Early Morning", "Mid to Late Morning", "Afternoon", "Late Afternoon to Evening", "Night")

LATSEATTLE = "47.6062"
LNGSEATTLE = "-122.3321"

NUMFEATURES = 16			#Number of features (inputs) in the TensorFlow model
#Feature 0	LAT
#Feature 1	LNG
#Feature 2  Day of Week (sin)
#Feature 3  Day of Week (cos)
#Feature 4  Day of year (sin)
#Feature 5  Day of year (cos)
#feature 6  year
#feature 7  Time of day (sin)
#feature 8  Time of day (cos)
#feature 9  Total Against Property Crime last 28 days for that police beat
#feature 10  Total Against Person Crime last 28 days for that police beat
#feature 11	 Total Against Society Crime last 28 days for that police beat
#feature 12 Temperature
#feature 13 rain amount 
#feature 14 cloud cover %
#feature 15 snow

NUMLABELS = 3 			#Number of labels (outputs) in the TensorFlow model
#Label 1  Amount of against property crime
#Label 2 Amount of against person crome
#Label 3 Amount of against society

NUMDAYSFORPASTCRIMES = 27
NUMDAYSOFFSETCRIMES = 28 	

BEATSGEOSEATTLE = "MapData/SeattlePoliceBeatsFixedPoly.geojson"
BEATSNEIGHBORHOODSEATTLE = "MapData/Beat_to_Neighborhood.csv"

WEATHERTYPEDEFINITIONS = "WeatherDefinitions/visualcrossingtype.csv"
WEATHERICONS = "WeatherIcons/"

DROPBEATS = ["99", "H1", "H2", "H3"]     #Buffer beats around outer parameter. Typically no crime data here so drop them out 

URLPREDICTIONS = "Predictions/"
URLMODELNAME = "crime_model"

URLCRIMEBASE = 'https://data.seattle.gov/resource/tazs-3rd5.json?$limit=%s&$where=offense_start_datetime between %s and %s' 
URLWEATHERBASE = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/'

#!/usr/bin/python3.7
import constants
import appids #create appids.py in root dir and set WEATHER = app key obtained from visualcrossing.com account

from pathlib import Path

import json, requests

import os

import pandas as pd
import geopandas as gpd 

import branca.colormap as cm

from folium import Map, Icon, Html, Popup, Marker #forked Folium. First Install fork with pip install -e git+https://github.com/justinjconroy/folium.git#egg=folium
from folium.plugins import TimeSliderChoropleth

from datetime import date, timedelta, datetime

import numpy as np

import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import joblib

#procedure to download json file from data server and sve locally so it doesn't need to be downloaded next time
def getJsonArchive(urlDownload: str, fileArchive: str):
	if (not Path(fileArchive).is_file()) or (fileArchive[-15:]=="incomplete.json"):
		#download it from server if it wasn't downloaded before or was a recent download where all data might not have been collected yet
		print("Downloading json file from server...", fileArchive, end="\r")
		response = requests.get(urlDownload)
		response.raise_for_status()

		jsonData = json.loads(response.text)
		with open(fileArchive, "w") as fileOut:
			fileOut.write(response.text)
			fileOut.close()
	else:
		#otherwise open the file that was downloaded previously
		print("Reading archived json file..." + fileArchive, end="\r")
		with open(fileArchive, "r") as fileIn:
			jsonData = json.load(fileIn)
			fileIn.close()

	return (jsonData)

#procedure to load Seattle crime data over specific date range from json files located on Seattle Open Data server to Panda Dataframe
def crimeGetRange(dayBegin, dayEnd):

	dfReturn = pd.DataFrame()

	dayChunkBegin = date(dayBegin.year, 1, 1) + timedelta((dayBegin.timetuple().tm_yday // constants.DAYSTEP) * constants.DAYSTEP) 
	while dayChunkBegin <= dayEnd:
		queryChunkStart = dayChunkBegin.strftime("'%Y-%m-%dT00:00:00'")
		dayChunkEnd = dayChunkBegin + timedelta(constants.DAYSTEP - 1)
		if dayChunkEnd.year > dayChunkBegin.year: dayChunkEnd = date(dayChunkBegin.year, 12, 31)
		queryChunkEnd = dayChunkEnd.strftime("'%Y-%m-%dT23:59:59'")
		
		urlCrime = constants.URLCRIMEBASE % (constants.RECORDLIMIT, queryChunkStart, queryChunkEnd) #&cnt=3&APPID=%s' % (location, APPID)
		fileCrimeDownload = "CrimeDownload/" + dayChunkBegin.strftime("%m-%d-%y") + "to" + dayChunkEnd.strftime("%m-%d-%y")
		if dayChunkEnd >= (date.today() - timedelta(constants.COMPLETEMINDAYS)) :
			fileCrimeDownload = fileCrimeDownload + "incomplete"    #seattle police upload records still being uploaded in last week so need to redownload later
		fileCrimeDownload = fileCrimeDownload + ".json"
		#print ("Retrieving data from Seattle Open Data...")
		chunkData =  pd.DataFrame(getJsonArchive(urlCrime, fileCrimeDownload))
		if len(chunkData.index) > 0:	
			#Deleting Unknown Police Beat Crime Records
			chunkData.drop(chunkData[(chunkData.beat == 'UNKNOWN') | (chunkData.beat == '')].index, inplace=True)
					
			#Convert strings to dates
			chunkData['offense_start_datetime'] = pd.to_datetime(chunkData['offense_start_datetime'], errors = 'coerce')
			
			chunkData['offense_end_datetime'] = pd.to_datetime(chunkData['offense_end_datetime'], errors = 'coerce')
			chunkData['offense_end_datetime'].mask(chunkData['offense_end_datetime'].isnull(), chunkData['offense_start_datetime'], inplace=True)
			#chunkData['offense_middle_datetime'] = chunkData['offense_start_datetime'] + (chunkData['offense_end_datetime'] - chunkData['offense_start_datetime'])
			chunkData['offense_length_datetime'] = chunkData['offense_end_datetime'] - chunkData['offense_start_datetime']
			chunkData['hour_of_crime'] = chunkData['offense_start_datetime'].dt.hour
			dfReturn = pd.concat([dfReturn, chunkData])

		dayChunkBegin = dayChunkEnd + timedelta(1)

	return (dfReturn)

#procedure to count total number of crimes commited over specific date in a supplied Panda Dataframe 
def getNumCrimes(dfCrime, myStart, myEnd): 
	
	datetimeStart = datetime.combine(myStart, datetime.min.time()) 
	datetimeEnd = datetime.combine(myEnd, datetime.max.time()) 
	
	dfCrimeRange = (dfCrime[(dfCrime.offense_start_datetime >= datetimeStart) & 
		(dfCrime.offense_start_datetime <= datetimeEnd)])
	
	numCrime = np.zeros((constants.TIMEBLOCKNUM, 3))

	numCrime[:,0] = len((dfCrimeRange[dfCrimeRange.crime_against_category == 'PROPERTY']).index)
	numCrime[:,1] = len((dfCrimeRange[dfCrimeRange.crime_against_category == 'PERSON']).index)
	numCrime[:,2] = len((dfCrimeRange[dfCrimeRange.crime_against_category == 'SOCIETY']).index)
	return(numCrime)

#procedure to render crime map and timeslider choropleth
def getCrimeMap(dfBeatData, strCrimeName, dfBeatGeo, cmColorName, cmColorOpacity, minColor, maxColor):

	lstWeatherTxt = dfBeatData[dfBeatData['Index']==0]['WeatherLabel'].tolist()

	lstBeatsDef = dfBeatGeo['beat']
	beat_gdf = gpd.GeoDataFrame(dfBeatGeo[['geometry']])
	beat_gdf = beat_gdf.drop_duplicates().reset_index()

	#maxColor = max(dfBeatData[strCrimeName]) * 100
	#minColor = min(dfBeatData[strCrimeName]) * 100
	cmapCrime = cmColorName.scale(minColor, maxColor)
	dfBeatData[strCrimeName + 'Color'] = (dfBeatData[strCrimeName] * 100).map(cmapCrime)

	style_dict = {}
	for beatID, beatDef in enumerate(lstBeatsDef):
		result = dfBeatData[dfBeatData['Index'] == beatID]
		inner_dict = {}
		for c, r in result.iterrows():
			inner_dict[str(r['TimeOfDayUTC'])] = {'color': r[strCrimeName + 'Color'], 'opacity': cmColorOpacity}


		style_dict[str(beatID)] = inner_dict


	slideCrime = TimeSliderChoropleth(data=beat_gdf.to_json(), styledict=style_dict, customlbl=lstWeatherTxt, overlay=True, name=strCrimeName)

	mapForecast = Map(location=[constants.LATSEATTLE, constants.LNGSEATTLE], zoom_start = 11)
	mapForecast.add_child(slideCrime)
	cmapCrime.caption = "% Forecasted " + strCrimeName + " Potential in " + str(constants.TIMEBLOCKLEN) + " hour steps"
	cmapCrime.add_to(mapForecast)
	


	for beatID, beatDef in enumerate(lstBeatsDef):
		result = dfBeatData[dfBeatData['Index'] == beatID]
		currentbeatCrime = result[strCrimeName].sum(axis=0)
		if beatID==0:
			worstBeat = currentbeatCrime
			bestBeat = currentbeatCrime
		else:
			if currentbeatCrime > worstBeat:
				worstBeat = currentbeatCrime
			elif currentbeatCrime < bestBeat:
				bestBeat = currentbeatCrime

	lstWorstBeats = []
	lstBestBeats = []

	for beatID, beatDef in enumerate(lstBeatsDef):
		beatLat = dfBeatGeo.iloc[beatID]["y"]
		beatLng = dfBeatGeo.iloc[beatID]["x"]
		result = dfBeatData[dfBeatData['Index'] == beatID]
		result.sort_values(by=['TimeOfDayUTC'], inplace=True, ascending=True)
		
		currentbeatCrime = result[strCrimeName].sum(axis=0)
		if currentbeatCrime <= 1.2 * bestBeat:
			lstBestBeats.append(beatDef)
			iconX = Icon(icon="check", color='green', prefix='fa')
		elif currentbeatCrime  <= 0.80 * worstBeat:
			iconX = Icon(icon="minus", color='orange', prefix='fa')	
		else:
			iconX = Icon(icon="exclamation", color='red', prefix='fa')
			lstWorstBeats.append(beatDef)

		html=("""<div STYLE="font-size:25px; font-weight: bold; text-align: center; font-family: Garamond;">""" + beatDef + """ Police Beat Info</div>
		    <iframe src="BeatPopup.html?crime=""" + result[strCrimeName].to_json(orient="records", double_precision=4) + "&temp=" + result[strCrimeName + ':Change Due to Temp'].to_json(orient="records", double_precision=4) +
		    "&conditions=" + result[strCrimeName + ':Change Due to Conditions'].to_json(orient="records", double_precision=4) + """"\" scrolling="no" frameborder="0" width="230" height="170"></iframe>
		   	""")
		
		myhtml = Html(html, script=True)
		popupX = Popup(myhtml, max_width = 330, max_height = 210)
		Marker([beatLat, beatLng], popup=popupX, icon=iconX).add_to(mapForecast)

	return([mapForecast, lstBestBeats, lstWorstBeats])

#procedure to get crime predictions 
def getCrimePredictions(dictdfCrimeByBeats, dateBegin, dfBeatGeo, defWeatherType, fieldsOverwrite=None):
	now = datetime.now()

	lstBeatsDef = dfBeatGeo['beat']

	testData = np.zeros((len(lstBeatsDef) * constants.TIMEBLOCKNUM, constants.NUMFEATURES))

	dfBeatData = pd.DataFrame(columns=['Index', 'TimeOfDayIndex', 'TimeOfDayUTC', 'WeatherTemp', 'WeatherPrecip', 'WeatherCloud', 'WeatherSnow', 'WeatherDescription'])

	appendRowStart = 0
	appendRowEnd = 0
	
	
	print("Producing forcast input data for date: " + dateBegin.strftime("'%m-%d-%Y'"), end="\r")
	datetimeStart = datetime.combine(dateBegin, datetime.min.time())
	datetimeEnd = datetime.combine(dateBegin, datetime.max.time())
	

	weatherElements = getWeather(dateBegin, recordWhen=True)
	weathertempElement = weatherElements['temp']
	weatherprecipElement = weatherElements['precip']
	weathercloudElement = weatherElements['cloud']
	weathersnowElement = weatherElements['snow']
	weatherdescElement = weatherElements['desc']
		
		
	lstWeatherTxt = []
	for hrBlock in range(constants.TIMEBLOCKNUM):
		myDescription = defWeatherType.loc[defWeatherType['IdIndex'] == weatherdescElement[hrBlock], 'TypeDescription'].item()		
		if ((hrBlock*constants.TIMEBLOCKLEN) > 6) and ((hrBlock*constants.TIMEBLOCKLEN) <= 18):
			myIcon = constants.WEATHERICONS + defWeatherType.loc[defWeatherType['IdIndex'] == weatherdescElement[hrBlock], 'DayIcon'].item()
		else:
			myIcon = constants.WEATHERICONS + defWeatherType.loc[defWeatherType['IdIndex'] == weatherdescElement[hrBlock], 'NightIcon'].item()

		fileIcon = open(myIcon, "r")
		dataIcon = fileIcon.read()
		#dataIcon = urlEncode(dataIcon) 	#https://stackoverflow.com/questions/30733607/svg-data-image-not-working-as-a-background-image-in-a-pseudo-element
		dataIcon=dataIcon.replace("<svg", "<svg width='36px' height='22px'")
		fileIcon.close()
		lstWeatherTxt.append("<strong> - Weather: </strong>" + myDescription + 
			" <body>" + dataIcon + "</body><strong> - Temperature: </strong>" + str(round(convertC(weathertempElement[hrBlock]), 1)) + " °F")


		
	for beatID, beatDef in enumerate(lstBeatsDef):
		beatPreviousNumCrimes = getNumCrimes(dictdfCrimeByBeats[beatDef], (dateBegin - timedelta(constants.NUMDAYSFORPASTCRIMES+constants.NUMDAYSOFFSETCRIMES)), (dateBegin - timedelta(constants.NUMDAYSOFFSETCRIMES)))	
		
		dataElement = np.zeros((constants.TIMEBLOCKNUM, constants.NUMFEATURES))
		weekdayNorm = 2 * np.pi * dateBegin.weekday() / 6
		dayyearNorm = 2 * np.pi * dateBegin.timetuple().tm_yday / 365
		timeBlockNorm = 2 * np.pi * np.linspace(0, constants.TIMEBLOCKNUM-1, constants.TIMEBLOCKNUM) / (constants.TIMEBLOCKNUM-1)
		
		dataElement[:, 0] = dfBeatGeo.iloc[beatID]["y"]  							#Feature 0	LAT
		dataElement[:, 1] = dfBeatGeo.iloc[beatID]["x"] 							#Feature 1	LNG
		dataElement[:, 2] = np.sin(weekdayNorm)	 									#Feature 2  Day of Week (sin)
		dataElement[:, 3] = np.cos(weekdayNorm)										#Feature 3  Day of Week (cos)
		dataElement[:, 4] = np.sin(dayyearNorm)										#Feature 4  Day of year (sin)
		dataElement[:, 5] = np.cos(dayyearNorm)										#Feature 5  Day of year (cos)
		dataElement[:, 6] = dateBegin.year  								    	#feature 6  year
		dataElement[:, 7] = np.sin(timeBlockNorm)									#feature 7  Time of day (sin)
		dataElement[:, 8] = np.cos(timeBlockNorm)									#feature 8  Time of day (cos)
		dataElement[:, 9] = beatPreviousNumCrimes[:, 0]								#feature 9  Total Against Property Crime last 28 days for that police beat
		dataElement[:, 10] = beatPreviousNumCrimes[:, 1] 							#feature 10  Total Against Person Crime last 28 days for that police beat
		dataElement[:, 11] = beatPreviousNumCrimes[:, 2] 							#feature 11	 Total Against Society Crime last 28 days for that police beat
		dataElement[:, 12] = weathertempElement										#feature 12 Temperature
		dataElement[:, 13] = weatherprecipElement									#feature 13 rain
		dataElement[:, 14] = weathercloudElement									#feature 14 clouds
		dataElement[:, 15] = weathersnowElement										#feature 15 snow

		#this is for calls to the function which want some fields to be used for comparisons
		if fieldsOverwrite != None:
			for field in fieldsOverwrite:
				dataElement[:, field] = fieldsOverwrite[field]


		appendRowStart = appendRowEnd 
		appendRowEnd += constants.TIMEBLOCKNUM

		testData[appendRowStart:appendRowEnd, :] = dataElement

		utcTimeOfDay = [(datetimeStart.timestamp() + (x * constants.TIMEBLOCKLEN * 3600)) for x in range(constants.TIMEBLOCKNUM)]
		beatIndex = [beatID] * constants.TIMEBLOCKNUM
		
		dfBeatData = pd.concat([dfBeatData, pd.DataFrame({'Index': beatIndex, 'TimeOfDayIndex': range(constants.TIMEBLOCKNUM), 
			'TimeOfDayUTC': utcTimeOfDay, 'WeatherTemp': dataElement[:, 12], 'WeatherPrecip': dataElement[:, 13],
			'WeatherCloud': dataElement[:, 14], 'WeatherSnow': dataElement[:, 15], 'WeatherDescription': weatherdescElement, 
			'WeatherLabel': lstWeatherTxt})])

	normScaler = joblib.load(constants.URLMODELNAME + '/scaler.gz')	
	testData = normScaler.transform(testData)
	modelForecast = keras.models.load_model(constants.URLMODELNAME)

	crimePredictions = modelForecast.predict(testData)

	dfBeatData['Property Crime'] = crimePredictions[:, 0]
	dfBeatData['Person Crime'] = crimePredictions[:, 1]
	dfBeatData['Society Crime'] = crimePredictions[:, 2]
	dfBeatData['Total Crime'] = dfBeatData['Property Crime'] + dfBeatData['Person Crime'] + dfBeatData['Society Crime']

	return(dfBeatData)

#procedure to compare two diffrent prediction datasets
def comparePredictions(crimePredictions1, crimePredictions2):
	crimeAvgTemp1 = crimePredictions1['WeatherTemp'].mean(axis=0)
	crimeSumTotal1 = crimePredictions1['Total Crime'].sum(axis=0) 
	crimeProperty1 = crimePredictions1['Property Crime'].sum(axis=0) 
	crimePerson1 = crimePredictions1['Person Crime'].sum(axis=0) 
	crimeSociety1 = crimePredictions1['Society Crime'].sum(axis=0)

	crimeAvgTemp2 = crimePredictions2['WeatherTemp'].mean(axis=0)
	crimeSumTotal2 = crimePredictions2['Total Crime'].sum(axis=0) 
	crimeProperty2 = crimePredictions2['Property Crime'].sum(axis=0) 
	crimePerson2 = crimePredictions2['Person Crime'].sum(axis=0) 
	crimeSociety2 = crimePredictions2['Society Crime'].sum(axis=0)



	difTemp = crimeAvgTemp1 - crimeAvgTemp2
	perTemp = (crimeAvgTemp1 - crimeAvgTemp2) * 100 / crimeAvgTemp2

	difCrime = crimeSumTotal1 - crimeSumTotal2
	perCrime = (crimeSumTotal1 - crimeSumTotal2) * 100 / crimeSumTotal2
	
	difProperty = crimeProperty1 - crimeProperty2
	perProperty = (crimeProperty1 - crimeProperty2) * 100 / crimeProperty2
	
	difPerson = crimePerson1 - crimePerson2
	perPerson = (crimePerson1 - crimePerson2) * 100 / crimePerson2
	
	difSociety = crimeSociety1  - crimeSociety2
	perSociety = (crimeSociety1  - crimeSociety2) * 100 / crimeSociety2

	return ({'difTemp': difTemp, 'perTemp': perTemp, 
		'difCrime': difCrime, 'perCrime': perCrime, 
		'difProperty': difProperty, 'perProperty': perProperty, 
		'difPerson': difPerson, 'perPerson': perPerson, 
		'difSociety': difSociety, 'perSociety': perSociety})

#Get weather history for that day
def getWeather(dateChunk, recordWhen=False):
	
	urlWeatherCurrent = (constants.URLWEATHERBASE + constants.LATSEATTLE + "%2C%20" + constants.LNGSEATTLE +
			"/" + dateChunk.strftime("%Y-%m-%d") + "/" + dateChunk.strftime("%Y-%m-%d") +
			"?unitGroup=metric&include=hours&lang=id&key=" + appids.WEATHER + "&contentType=json")
	fileWeatherDownload = "WeatherDownload/" + "weatherdate" + dateChunk.strftime("%m-%d-%y") 
	if recordWhen == True:
		fileWeatherDownload = fileWeatherDownload +"obtained" + date.today().strftime("%m-%d-%y-") + str(datetime.now().hour) + "hr"
	fileWeatherDownload = fileWeatherDownload + ".json"

	weatherData = getJsonArchive(urlWeatherCurrent, fileWeatherDownload)

	w = weatherData['days'][0]['hours']


	tempElement = np.zeros(constants.TIMEBLOCKNUM)
	precipElement = np.zeros(constants.TIMEBLOCKNUM)
	cloudElement = np.zeros(constants.TIMEBLOCKNUM)
	snowElement = np.zeros(constants.TIMEBLOCKNUM)
	descElement = np.zeros(constants.TIMEBLOCKNUM)

	for hrBlock in range(constants.TIMEBLOCKNUM):
		if hrBlock*constants.TIMEBLOCKLEN <= len(w) - 1:
			currentWeather = w[int(hrBlock*constants.TIMEBLOCKLEN)]
			if currentWeather['temp'] is not None:							
				tempElement[hrBlock] = float(currentWeather['temp'])
			else:
				tempElement[hrBlock] = 0
			
			if currentWeather['precip'] is not None:
				precipElement[hrBlock] = float(currentWeather['precip'])
			else:
				precipElement[hrBlock] = 0
			
			if currentWeather['cloudcover'] is not None:
				cloudElement[hrBlock] = float(currentWeather['cloudcover'])
			else:
				cloudElement[hrBlock] = 0
			
			if currentWeather['snow'] is not None:
				snowElement[hrBlock] = float(currentWeather['snow'])
			else:
				snowElement[hrBlock] = 0

			lstDesc = (currentWeather['conditions']).split(',')
			descElement[hrBlock] = int(lstDesc[0][5:])
		else:   #incomplete missing last hr in visual crossings dataElement. Assign previous hr's data
		  	print ("Missing Visual Crossing's Hr")
		  	tempElement[hrBlock] = tempElement[hrBlock-1]
		  	precipElement[hrBlock] = precipElement[hrBlock-1]
		  	cloudElement[hrBlock] = cloudElement[hrBlock-1]
		  	snowElement[hrBlock] = snowElement[hrBlock-1]
		  	descElement[hrBlock] = descElement[hrBlock-1]

	
	return ({'temp': tempElement, 'precip': precipElement, 'cloud': cloudElement, 'snow': snowElement, 'desc': descElement})	

def getWeatherHistoryStats(dateBegin):
	urlWeatherCurrent = (constants.URLWEATHERBASE + constants.LATSEATTLE + "%2C%20" + constants.LNGSEATTLE +
		"/" + dateBegin.strftime("%Y-%m-%d") + "/" + dateBegin.strftime("%Y-%m-%d") +
		"?unitGroup=metric&include=stats&lang=id&key=" + appids.WEATHER + "&contentType=json")
	fileWeatherDownload = "WeatherDownload/" + "statsdate" + dateBegin.strftime("%m-%d-%y") + "obtained" + date.today().strftime("%m-%d-%y") + ".json"	
	statsData = getJsonArchive(urlWeatherCurrent, fileWeatherDownload)
	s = statsData['days'][0]
	avgTemp = s['temp']
	maxTemp = s['tempmax']						
	minTemp = s['tempmin']
	cloudCover = s['cloudcover']
	totalPrecip = s['precip']
	
	if s['snow'] is not None:
		totalSnow = s['snow']
	else:
		totalSnow = 0

	return ({'avgTemp': avgTemp, "maxTemp": maxTemp, "minTemp": minTemp, 'cloudcover': cloudCover, "totalprecip": totalPrecip, "totalsnow": totalSnow})

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def normalizeNum(nums, newMin, newMax):
	
	oldMin = min(nums)
	oldMax = max(nums)

	ratio = (newMax - newMin) / (oldMax - oldMin)

	return (newMin + (nums - oldMin) * ratio)

def convertC(myCelsius):
	myFahrenheit = (myCelsius * 1.8) + 32
	return(myFahrenheit)

def urlEncode(myString):
	
	processString = myString

	dictCodes = {"%": "%25",
	"<": "%3C",
	">": "%3E",
	" ": "%20",
	"!": "%21",
	"*": "%2A",
	"'": "%27",
	'"': "%22",
	"(": "%28",
	")": "%29",
	";": "%3B",
	":": "%3A",
	"@": "%40",
	"&": "%26",
	"=": "%3D",
	"+": "%2B",
	"$": "%24",
	",": "%2C",
	"/": "%2F",
	"?": "%3F",
	"#": "%23",
	"[": "%5B",
	"]": "%5D"}
    
	for urlCode in dictCodes:
		processString = processString.replace(urlCode, dictCodes[urlCode])

	return (processString)

#take time interger 0-23 and convert to string with 00:00 AM/PM format
def convertTimeBlock(myTimeBlock):

	currentHour = (myTimeBlock * constants.TIMEBLOCKLEN) % 24
	if (currentHour <= 12):
		if (currentHour == 12):
			currentAMPM = "PM"
		else:
			currentAMPM = "AM"
			if (currentHour == 0):
				currentHour = 12

	else:
		currentHour = (myTimeBlock * constants.TIMEBLOCKLEN) % 12
		currentAMPM = "PM"

	return(str(currentHour) + ":00 " + currentAMPM)
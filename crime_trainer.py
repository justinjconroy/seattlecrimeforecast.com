#!/usr/bin/python3.7
print("Loading...")

import constants

import sys
from pathlib import Path

from crime_util import crimeGetRange, getNumCrimes, getWeather

import json, requests

from datetime import date, timedelta, datetime, timezone

import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import RandomNormal, Constant

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

import pandas as pd
import geopandas as gpd

dateEnd = date(2022, 5, 31) 
dateBegin = date(2012, 5, 31)

URLTRAINDATA = "Training/traindata" + dateBegin.strftime("%m-%d-%y") + "to" + dateEnd.strftime("%m-%d-%y") + ".csv"
URLLABELSDATA = "Training/labelsdata" + dateBegin.strftime("%m-%d-%y") + "to" + dateEnd.strftime("%m-%d-%y") + ".csv"


#Load Police Beat Definitions 
dfBeatGeo = gpd.read_file(constants.BEATSGEOSEATTLE)
#drop unused police beats around parameter and calculate lat and lng center pointers for remaining beats. 
#At some point should save these changes to geojson so it doesn't need to be calculated each run time 
for dropBeat in constants.DROPBEATS:
	dfBeatGeo.drop(dfBeatGeo.index[dfBeatGeo['beat'] == dropBeat], inplace=True)
lstBeatsDef = dfBeatGeo['beat']
print ("Number of beats: ", len(lstBeatsDef))

beat_gdf = gpd.GeoDataFrame(dfBeatGeo[['geometry']])
dfBeatGeo['x'] = beat_gdf.centroid.x
dfBeatGeo['y'] = beat_gdf.centroid.y
beat_gdf = beat_gdf.drop_duplicates().reset_index()

if not (Path(URLTRAINDATA).is_file() or Path(URLLABELSDATA ).is_file()):

	print ("Loading Crime Data...")
	dfCrime = crimeGetRange(dateBegin - timedelta(constants.NUMDAYSFORPASTCRIMES + constants.NUMDAYSOFFSETCRIMES), dateEnd)
	
	dictdfCrimeByBeats = {}
	for beatDef in lstBeatsDef:
		dictdfCrimeByBeats[beatDef]  =  dfCrime[dfCrime.beat == beatDef]
	print("\nCompleted crime aquisition")

	dateChunk = dateBegin

	trainData = np.zeros(((int((dateEnd - dateBegin).days) + 1) * len(lstBeatsDef) * constants.TIMEBLOCKNUM, constants.NUMFEATURES))
	trainLabels = np.zeros(((int((dateEnd - dateBegin).days)  + 1) * len(lstBeatsDef) * constants.TIMEBLOCKNUM, constants.NUMLABELS))
	
	appendRowStart = 0
	appendRowEnd = 0
	while dateChunk <= dateEnd:
		print("Processing data for date: " + dateChunk.strftime("'%m-%d-%Y'"), end="\r")
		
		datetimeStart = datetime.combine(dateChunk, datetime.min.time())
		datetimeEnd = datetime.combine(dateChunk, datetime.max.time())
		
		weatherElements = getWeather(dateChunk, recordWhen=False)
		weathertempElement = weatherElements['temp']
		weatherprecipElement = weatherElements['precip']
		weathercloudElement = weatherElements['cloud']
		weathersnowElement = weatherElements['snow']
		weatherdescElement = weatherElements['desc']
		#Create training and label data
		
		for beatID, beatDef in enumerate(lstBeatsDef):
			
			dfCrimeToday = dictdfCrimeByBeats[beatDef]
			dfCrimeToday = (dfCrimeToday[(dfCrimeToday.offense_start_datetime >= datetimeStart) & 
				(dfCrimeToday.offense_start_datetime <= datetimeEnd)])

			dataElement = np.zeros((constants.TIMEBLOCKNUM, constants.NUMFEATURES))
			weekdayNorm = 2 * np.pi * dateChunk.weekday() / 6
			dayyearNorm = 2 * np.pi * dateChunk.timetuple().tm_yday / 365
			timeBlockNorm = 2 * np.pi * np.linspace(0, constants.TIMEBLOCKNUM-1, constants.TIMEBLOCKNUM) / (constants.TIMEBLOCKNUM-1)


			beatPreviousNumCrimes = getNumCrimes(dictdfCrimeByBeats[beatDef], dateChunk - timedelta(constants.NUMDAYSFORPASTCRIMES+constants.NUMDAYSOFFSETCRIMES), dateChunk - timedelta(constants.NUMDAYSOFFSETCRIMES))	
			
			dataElement[:, 0] = dfBeatGeo.iloc[beatID]["y"]  							#Feature 0	LAT
			dataElement[:, 1] = dfBeatGeo.iloc[beatID]["x"] 							#Feature 1	LNG
			dataElement[:, 2] = np.sin(weekdayNorm)	 									#Feature 2  Day of Week (sin)
			dataElement[:, 3] = np.cos(weekdayNorm)										#Feature 3  Day of Week (cos)
			dataElement[:, 4] = np.sin(dayyearNorm)										#Feature 4  Day of year (sin)
			dataElement[:, 5] = np.cos(dayyearNorm)										#Feature 5  Day of year (cos)
			dataElement[:, 6] = dateChunk.year  								    	#feature 6  year
			dataElement[:, 7] = np.sin(timeBlockNorm)									#feature 7  Time of day (sin)
			dataElement[:, 8] = np.cos(timeBlockNorm)									#feature 8  Time of day (cos)
			dataElement[:, 9] = beatPreviousNumCrimes[:, 0]								#feature 9  Total Against Property Crime last 28 days for that police beat
			dataElement[:, 10] = beatPreviousNumCrimes[:, 1]							#feature 10  Total Against Person Crime last 28 days for that police beat
			dataElement[:, 11] = beatPreviousNumCrimes[:, 2]							#feature 11	 Total Against Society Crime last 28 days for that police beat
			dataElement[:, 12] = weathertempElement										#feature 12 Temperature
			dataElement[:, 13] = weatherprecipElement									#feature 13 rain
			dataElement[:, 14] = weathercloudElement									#feature 14 clouds
			dataElement[:, 15] = weathersnowElement										#feature 15 snow

			labelElement = np.zeros((constants.TIMEBLOCKNUM, constants.NUMLABELS))
			 
			if len(dfCrimeToday.index) > 0:	
				for crime in dfCrimeToday.itertuples():
					dtCurrent = crime.offense_start_datetime
					timeIndex = int(dtCurrent.hour / constants.TIMEBLOCKLEN)
					if crime.crime_against_category=='PROPERTY':
						labelElement[timeIndex, 0] += 1
					elif crime.crime_against_category=='PERSON':
						labelElement[timeIndex, 1] += 1
					elif crime.crime_against_category=='SOCIETY':
						labelElement[timeIndex, 2] += 1
									
			else:
				pass
				#no crimes in that police beat today
			
			appendRowStart = appendRowEnd 
			appendRowEnd += constants.TIMEBLOCKNUM

			trainData[appendRowStart:appendRowEnd, :] = dataElement
			trainLabels[appendRowStart:appendRowEnd, :] = labelElement
		
		dateChunk = dateChunk + timedelta(1)

	print("Saving Training Data...")
	np.savetxt(URLTRAINDATA, trainData, delimiter=",")
	np.savetxt(URLLABELSDATA , trainLabels, delimiter=",")
else:
	print("Loading Existing Training Data...")
	trainData = np.genfromtxt(URLTRAINDATA, delimiter=',')
	trainLabels = np.genfromtxt(URLLABELSDATA , delimiter=',')



#Create TensorFlow model 
if not Path(constants.URLMODELNAME).is_dir():
	# Split the data
	splitTrainData, splitValidateData, splitTrainLbl, splitValidateLbl = train_test_split(trainData, trainLabels, test_size=0.2, shuffle=True)

	normScaler = MinMaxScaler()
	normScaler.fit(splitTrainData)			#normalize data
	
	splitTrainData = normScaler.transform(splitTrainData)
	splitValidateData = normScaler.transform(splitValidateData)

	model = keras.Sequential([
		keras.layers.Input(shape=(splitTrainData.shape[1],)),                  
	   	keras.layers.Dense(96, activation = 'relu'),				#model seems to have difficulty fully fitting day night cycle in models <64
	   	keras.layers.Dense(96, activation = 'relu'),
	   	keras.layers.Dense(96, activation = 'relu'),
	   	keras.layers.Dense(3, activation = 'softplus')				#softplus activations so we don't get negative crime predictions
		], name="MLP_model")

	optimizer = keras.optimizers.Adamax(learning_rate=0.005)

	model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'msle'])

	model.summary()
	EPOCHS = 10
	# Store training stats
	log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
	es = EarlyStopping(monitor='val_loss', mode='min', patience=1, verbose=1)
	history = model.fit(splitTrainData, splitTrainLbl, epochs=EPOCHS, validation_data=(splitValidateData, splitValidateLbl), callbacks=[es, tensorboard_callback])
	model.save(constants.URLMODELNAME)
	joblib.dump(normScaler, constants.URLMODELNAME + '/scaler.gz')
	print("Tensor Flow Model complete!")
else:
	print("Model already trained on most recent data!")
	



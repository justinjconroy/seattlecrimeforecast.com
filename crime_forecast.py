#!/usr/bin/python3.7

#This module will run crime forecastes for multiple days: plot the folium maps, write a prediction summary html file, and also write it to an RSS feed

import constants
from crime_util import crimeGetRange, getNumCrimes, getCrimeMap, getCrimePredictions, getWeatherHistoryStats, normalizeNum, comparePredictions, convertTimeBlock, convertC

from datetime import date, timedelta, datetime

import json

import numpy as np

import pandas as pd
import geopandas as gpd

import branca.colormap as cm

import random

random.seed()

dateBegin = date.today()
dateEnd = dateBegin + timedelta(5)
#Load weather type definitions. Dataframe to convert ID to description
defWeatherType = pd.read_csv(constants.WEATHERTYPEDEFINITIONS)

#Load Police Beat Definitions and neieghborhood lookup for each
dfBeatGeo = gpd.read_file(constants.BEATSGEOSEATTLE)
dfBeatNeighborhood = pd.read_csv(constants.BEATSNEIGHBORHOODSEATTLE)

#drop unused police beats around parameter and calculate lat and lng center pointers for remaining beats. 
#At some point should save these changes to geojson so it doesn't need to be calculated each run time 
for dropBeat in constants.DROPBEATS:
	dfBeatGeo.drop(dfBeatGeo.index[dfBeatGeo['beat'] == dropBeat], inplace=True)

lstBeatsDef = dfBeatGeo['beat']


beat_gdf = gpd.GeoDataFrame(dfBeatGeo[['geometry']])
dfBeatGeo['x'] = beat_gdf.centroid.x
dfBeatGeo['y'] = beat_gdf.centroid.y
beat_gdf = beat_gdf.drop_duplicates().reset_index()


dfCrime = crimeGetRange(dateBegin - timedelta(constants.NUMDAYSFORPASTCRIMES + constants.NUMDAYSOFFSETCRIMES), dateEnd)

dictdfCrimeByBeats = {}
for beatDef in lstBeatsDef:
	dictdfCrimeByBeats[beatDef]  =  dfCrime[dfCrime.beat == beatDef]

#create rss feed channel header info
rss="""<?xml version="1.0" encoding="UTF-8" ?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom" xmlns:dc="http://purl.org/dc/elements/1.1/>

<channel>
  <title>Seattle Crime Forecast</title>
  <link>https://www.seattlecrimeforecast.com</link>
  <description>Seattle Crime Forecasting using TensorFlow Neural Networks</description>
  <category>Technology and Crime News</category>
  <copyright>2022 Seattle Crime Forecast. All rights reserved.</copyright>
  <language>en-us</language>
  <image>
    <url>https://www.seattlecrimeforecast.com/searching-robot.png</url>
    <title>Seattle Crime Forecast</title>
    <link>https://www.seattlecrimeforecast.com</link>
  </image>
  <atom:link href="https://www.seattlecrimeforecast.com/SeattleCrimeRSS.xml" rel="self" type="application/rss+xml" />
  """
lstPhotos = []
dateCurrent = dateBegin

while dateCurrent <= dateEnd:
  #get the crime predictions for this date (datecurrent)
  dfBeatData = getCrimePredictions(dictdfCrimeByBeats, dateCurrent, dfBeatGeo, defWeatherType)
  todayAvgTemp = dfBeatData['WeatherTemp'].mean(axis=0)
  todayAvgCrime = dfBeatData['Total Crime'].sum(axis=0)
  todayPropertyCrime = dfBeatData['Property Crime'].sum(axis=0)
  todayPersonCrime = dfBeatData['Person Crime'].sum(axis=0)
  todaySocietyCrime = dfBeatData['Society Crime'].sum(axis=0)

  #get weather history stats for what average weather on this date is
  weatherHistoryStats = getWeatherHistoryStats(dateCurrent)
  tempHistory = normalizeNum(dfBeatData['WeatherTemp'].iloc[:constants.TIMEBLOCKNUM],  weatherHistoryStats['minTemp'], weatherHistoryStats['maxTemp'])

  #get comparison predictions using average temperature instead
  dfHistoryData = getCrimePredictions(dictdfCrimeByBeats, dateCurrent, dfBeatGeo, defWeatherType, fieldsOverwrite={12: tempHistory})
  dfBeatData['Property Crime:Change Due to Temp'] = (dfBeatData['Property Crime'] - dfHistoryData['Property Crime'])
  dfBeatData['Person Crime:Change Due to Temp'] = (dfBeatData['Person Crime'] - dfHistoryData['Person Crime'])
  dfBeatData['Society Crime:Change Due to Temp'] = (dfBeatData['Society Crime'] - dfHistoryData['Society Crime'])
  dfBeatData['Total Crime:Change Due to Temp'] = (dfBeatData['Total Crime'] - dfHistoryData['Total Crime'])

  compareStatsTemp= comparePredictions(dfBeatData, dfHistoryData)

  difTemp = compareStatsTemp['difTemp']

  if difTemp < -0.1:
  	strTemp = str(abs(round(difTemp*1.8, 1))) + " &#176;F below average"
  elif difTemp > 0.1:
  	strTemp = str(abs(round(difTemp*1.8, 1))) + " &#176;F above average"
  else:
  	strTemp = "the same as average."

  perCrimeTemp = compareStatsTemp['perCrime']

  if perCrimeTemp < -0.1:
    strCrimeTemp = str(abs(round(perCrimeTemp, 2))) + "% below average."
  elif perCrimeTemp > 0.1:
    strCrimeTemp = str(abs(round(perCrimeTemp, 2))) + "% above average."
  else:
    strCrimeTemp = "the same as average."


  currentWeatherPrecip = dfBeatData['WeatherPrecip'].mean()
  currentWeatherCloud = dfBeatData['WeatherCloud'].mean()
  currentWeatherSnow = dfBeatData['WeatherSnow'].mean()
  print("Cloud Average today: ", currentWeatherCloud)
  print("Precip Average today: ", currentWeatherPrecip)
  print("Snow Average today: ", currentWeatherSnow)

  if (currentWeatherCloud > 50) and (currentWeatherPrecip < 1.27) and (currentWeatherSnow < 1.27): # (currentWeatherDescTxt == "Overcast") | (currentWeatherDescTxt == "Partially cloudy"):
    if currentWeatherCloud <= 75:
      currentWeatherDescTxt = str(round(currentWeatherCloud, 0)) + "% Partly cloudy"
      currentPhoto = "PartlyCloudy"
    else:
      currentWeatherDescTxt = "Overcast"
      currentPhoto = "Overcast"

  elif (currentWeatherCloud <= 50) and (currentWeatherPrecip < 1.27) and (currentWeatherSnow < 1.27):
    if (currentWeatherCloud <= 25):
      currentWeatherDescTxt = "Sunny"
      currentPhoto = "Sunny"
    else:
      currentWeatherDescTxt = str(round(100 - currentWeatherCloud, 0)) + "% Partly Sunny"
      currentPhoto = "PartlySunny"
  elif (currentWeatherSnow >= 1.27):
    currentWeatherDescTxt = "Snow"
    currentPhoto = "Snow"
  elif (currentWeatherPrecip >= 1.27):
    currentWeatherDescTxt = "Rain"
    currentPhoto = "Rain"
  else:
    currentWeatherDescType = dfBeatData['WeatherDescription'].mode()[0]
    currentWeatherDescTxt = defWeatherType[defWeatherType['IdIndex'] == int(currentWeatherDescType)]['TypeDescription'].iloc[0]
    currentPhoto = "Overcast"
    

  compareCloud = weatherHistoryStats['cloudcover']
  compareRain = weatherHistoryStats['totalprecip'] / constants.TIMEBLOCKNUM
  compareSnow = weatherHistoryStats['totalsnow'] / constants.TIMEBLOCKNUM
  
  if weatherHistoryStats['cloudcover'] > 50:
    if weatherHistoryStats['cloudcover'] <= 75:
      compareWeatherDesc = "Partly Cloudy"
    else:
      compareWeatherDesc = "Overcast"
  else:
    if weatherHistoryStats['cloudcover'] <= 25:
      compareWeatherDesc = "Sunny"
    else:
      compareWeatherDesc = "Partly Sunny"

  if weatherHistoryStats['totalprecip'] > 1.27:
    compareWeatherDesc = compareWeatherDesc + " and " + str(round(weatherHistoryStats['totalprecip'] / 25.4, 1)) + " total inches of precipitation"
  
  if weatherHistoryStats['totalsnow'] > 1.27:
    compareWeatherDesc = compareWeatherDesc + " and " + str(round(weatherHistoryStats['totalsnow'] / 25.4, 1)) + " total inches of snow"

  #get comparison predictions using average cloud cover and precipitation (and in rare cases: snow) instead
  dfCompareWeather = getCrimePredictions(dictdfCrimeByBeats, dateCurrent, dfBeatGeo, defWeatherType, fieldsOverwrite={13: compareRain, 14: compareCloud, 15: compareSnow})
  dfBeatData['Property Crime:Change Due to Conditions'] = (dfBeatData['Property Crime'] - dfCompareWeather['Property Crime'])
  dfBeatData['Person Crime:Change Due to Conditions'] = (dfBeatData['Person Crime'] - dfCompareWeather['Person Crime'])
  dfBeatData['Society Crime:Change Due to Conditions'] = (dfBeatData['Society Crime'] - dfCompareWeather['Society Crime'])
  dfBeatData['Total Crime:Change Due to Conditions'] = (dfBeatData['Total Crime'] - dfCompareWeather['Total Crime'])

  compareStatsWeather = comparePredictions(dfBeatData, dfCompareWeather)

  perCrimeWeather = compareStatsWeather['perCrime']
  if perCrimeWeather < -0.1:
    strCrimeWeather = "a decrease of " + str(abs(round(perCrimeWeather, 2))) + "%"
  elif perCrimeWeather > 0.1:
    strCrimeWeather = "an increase of " + str(abs(round(perCrimeWeather, 2))) + "%"
  else:
    strCrimeWeather = "about the same as average"

  if (perCrimeTemp / perCrimeWeather) < 0:
    strCrimeTemp = strCrimeTemp + " However,"
  else:
    strCrimeTemp = strCrimeTemp + " In addition,"

  dfBeatWeek = pd.DataFrame(columns=['Property Crime', 'Person Crime', 'Society Crime', 'Total Crime'])
  #get comparison predictions using for other days of the week instead
  for dayWeek in range(7):
    if dayWeek != dateCurrent.weekday():  #check to make sure prediction day of week isn't the one already done 
      weekdayNorm = 2 * np.pi * dayWeek / 6
      dfBeatDay = getCrimePredictions(dictdfCrimeByBeats, dateCurrent, dfBeatGeo, defWeatherType, fieldsOverwrite={2: np.sin(dayWeek), 3: np.cos(dayWeek)})
      if dfBeatWeek.empty:
        dfBeatWeek = dfBeatDay
      else:
        dfBeatWeek['Property Crime'] = dfBeatWeek['Property Crime'] + dfBeatDay['Property Crime']
        dfBeatWeek['Person Crime'] = dfBeatWeek['Person Crime'] + dfBeatDay['Person Crime']
        dfBeatWeek['Society Crime'] = dfBeatWeek['Society Crime'] + dfBeatDay['Society Crime']
        dfBeatWeek['Total Crime'] = dfBeatWeek['Total Crime'] + dfBeatDay['Total Crime']

  dfBeatWeek['Property Crime'] = dfBeatWeek['Property Crime'] / 6
  dfBeatWeek['Person Crime'] = dfBeatWeek['Person Crime'] / 6
  dfBeatWeek['Society Crime'] = dfBeatWeek['Society Crime'] / 6
  dfBeatWeek['Total Crime'] = dfBeatWeek['Total Crime'] / 6

  compareStatsWeekday = comparePredictions(dfBeatData, dfBeatWeek)
  perDayWeek = compareStatsWeekday['perCrime']
  if perDayWeek < -0.1:
    strDay = "a decrease of " + str(abs(round(perDayWeek, 2))) + "%"
  elif perDayWeek > 0.1:
    strDay = "an increase of " + str(abs(round(perDayWeek, 2))) + "%"
  else:
    strDay = "about the same as average"

  perTotalChange = perDayWeek + perCrimeWeather + perCrimeTemp
  if perTotalChange < -0.1:
    strTotalChange = str(abs(round(perTotalChange, 2))) + "% below average"
  elif perTotalChange > 0.1:
    strTotalChange = str(abs(round(perTotalChange, 2))) + "% above average"
  else:
    strTotalChange = "about the same as average"

  #render the crime maps
  mapForecast = getCrimeMap(dfBeatData, "Property Crime", dfBeatGeo, cm.linear.Oranges_09, 0.75, 0, 35)
  mapForecast[0].save(constants.MYWEBPATH + "PropertyCrime" + dateCurrent.strftime("-%m-%d-%Y") + ".html")
  beatBestProperty = mapForecast[1]
  beatWorstProperty = mapForecast[2]

  mapForecast = getCrimeMap(dfBeatData, "Person Crime", dfBeatGeo, cm.linear.Reds_09, 0.75, 0, 7)
  mapForecast[0].save(constants.MYWEBPATH + "PersonCrime" + dateCurrent.strftime("-%m-%d-%Y") + ".html")
  beatBestPerson = mapForecast[1]
  beatWorstPerson = mapForecast[2]

  mapForecast = getCrimeMap(dfBeatData, "Society Crime", dfBeatGeo, cm.linear.Purples_09, 0.75, 0, 5)
  mapForecast[0].save(constants.MYWEBPATH + "SocietyCrime" + dateCurrent.strftime("-%m-%d-%Y") + ".html")
  beatBestSociety = mapForecast[1]
  beatWorstSociety = mapForecast[2]

  mapForecast = getCrimeMap(dfBeatData, "Total Crime", dfBeatGeo, cm.linear.YlOrRd_09, 0.75, 0, 40)
  mapForecast[0].save(constants.MYWEBPATH + "TotalCrime" + dateCurrent.strftime("-%m-%d-%Y") + ".html")
  beatBestCrime = mapForecast[1]
  nBestCrime = []
  nWorstCrime = []
  for b in beatBestCrime:
    nBestCrime.append(dfBeatNeighborhood.loc[dfBeatNeighborhood['Beat'] == b, 'Neighborhood'].item())
  nBestCrime = list(dict.fromkeys(nBestCrime)) #remove duplicates

  beatWorstCrime = mapForecast[2]
  for b in beatWorstCrime:
    nWorstCrime.append(dfBeatNeighborhood.loc[dfBeatNeighborhood['Beat'] == b, 'Neighborhood'].item())
  nWorstCrime = list(dict.fromkeys(nWorstCrime)) #remove duplicates

  for hr in range(0, constants.TIMEBLOCKNUM, constants.TIMEBLOCKLEN):
    crimeHrBlock = dfBeatData[dfBeatData['TimeOfDayIndex']==hr]["Total Crime"].sum(axis=0)
    if hr==0:
      iTimeBest = 0
      iTimeWorst = 0
      countBest = crimeHrBlock
      countWorst = crimeHrBlock
    else:
      if crimeHrBlock < countBest:
        countBest = crimeHrBlock
        iTimeBest = hr
      if crimeHrBlock > countWorst:
        countWorst = crimeHrBlock
        iTimeWorst = hr

  strTimeBest = convertTimeBlock(iTimeBest)
  strTimeWorst = convertTimeBlock(iTimeWorst)

  #render html summary page for this date
  strInro = ""

  if dateCurrent.weekday() == 0:
    strIntro = (["The Seattle work week starts out on Monday with",
      "The Seattle work week begins on Monday being",
      "On Monday, the Seattle work week revs up with"][random.randint(0,2)])
  elif dateCurrent.weekday() ==  1:
    strIntro = (["On Tuesday, the Seattle work week continues with",
      "The Seattle work week carries on to Tuesday with",
      "The Seattle work week pushes on into Tuesday with"][random.randint(0,2)])
  elif dateCurrent.weekday() ==  2:  
    strIntro = (["The Seattle work week continues into the downhill stretch on Wednesday with",
      "The Seattle work week reaches hump day Wednesday with",
      "On Wednesday, The Seattle work week reaches the mid point with"][random.randint(0,2)])
  elif dateCurrent.weekday() ==  3:  
    strIntro = (["The Seattle work week approaches its end on Thursday with",
      "The Seattle work week creeps closer to the weekend on Thursday with",
      "On Thursday, the weekend is right around the corner with"][random.randint(0,2)])
  elif dateCurrent.weekday() ==  4:  
    strIntro = (["The Seattle weekend starts on Friday with",
      "Friday begins the Seattle weekend with",
      "Friday ushers in the Seattle weekend with"][random.randint(0,2)])
  elif dateCurrent.weekday() ==  5:  
    strIntro = (["The Seattle weekend is in full swing on Saturday with",
      "Saturday anchors the Seattle weekend with",
      "The Seattle weekend's first full day on Saturday will be"][random.randint(0,2)])
  elif dateCurrent.weekday() == 6:
    strIntro = (["The Seattle weekend ends on Sunday with",
      "The Seattle weekend comes to a close on Sunday with",
      "Sundays says goodbye to the Seattle weekend with"][random.randint(0,2)])

  
  strIntro = strIntro + " " + currentWeatherDescTxt.lower() + " weather conditions and crime " + strTotalChange + "."

  matchPhoto = False
  firstPass = True
  #ensure that the weather photoes don't repeat
  while firstPass or matchPhoto:
    firstPass = False
    strPhoto = "Photos/" + currentPhoto + str(random.randint(1,5)) + ".jpg" 
    matchPhoto = [s for s in lstPhotos if strPhoto in s]

  lstPhotos.append(strPhoto)

  f = open(constants.MYWEBPATH + 'Summary' + dateCurrent.strftime("-%m-%d-%Y") + '.html','w')

  summary = """<!DOCTYPE html>
  <html>
  <head>
  <style>
  table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
  }
  th, td {
    padding: 12px;
  }
  </style>
  </head>
  <body  style="text-align:center">
  <p style="text-align:left">%s First, for %s the neural network model is forcasting %s crime reporting potential compared to other days of the week for this time of year. Then there is the weather situation.
  The average forecasted temperature for the day is %.2f &#176;F which is %s.
   As a result, the crime prediction model is forecasting total crime at %s the forecasted weather conditions are %s.
   Compared to normal %s conditions, the model is predicting %s. All combined this leads to crime %s.</p>
   
  <table style="margin-left: auto; margin-right: auto;", border=1>
    <tr>
      <th>Crime Type</th>
      <th>Forecasted Number</th>
      <th>Change Due to Day of Week</th>
      <th>Change Due to Temperature</th>
      <th>Change Due to Weather Conditions</th>
      <th>Total Change</th>       
    </tr>
    <tr>
      <td>Against Property (Burglary, Car Prowls)</td>
      <td>%d</td>
      <td>%+.2f%%</td>
      <td>%+.2f%%</td>
      <td>%+.2f%%</td>
      <td>%+.2f%%</td>
    </tr>
    <tr>
      <td>Against People (Assaults, Threats)</td>
      <td>%d</td>
      <td>%+.2f%%</td>
      <td>%+.2f%%</td>
      <td>%+.2f%%</td>
      <td>%+.2f%%</td>
    </tr>
    <tr>
      <td>Against Society (DUI, Drugs)</td>
      <td>%d</td>
      <td>%+.2f%%</td>
      <td>%+.2f%%</td>
      <td>%+.2f%%</td>
      <td>%+.2f%%</td>
    </tr>
    <tr>
      <th>Crime Total</th>
      <th>%d</th>
      <th>%+.2f%%</th>
      <th>%+.2f%%</th>
      <th>%+.2f%%</th>
      <th>%+.2f%%</th>
    </tr>
  </table>
  <p style="text-align:left">The least amount of crime is forecasted at %s while the peak is at %s. The Seattle neighborhoods making the naughty and nice lists are: </p>
  <span>
   <ul style="list-style-type: none; display: inline-block; vertical-align: top;">
      <h4>Worst</h4>
      <li>%s</li>
   </ul>
   
   <ul style="list-style-type: none; display: inline-block; vertical-align: top;">
      <h4>Best</h4>
      <li>%s</li>
   </ul>
  <span><br>
   <img src="%s" style="width: 600px; height: auto;" alt="Seattle">
  <p><em>Forecast last updated on %s (PST)</em></p>
  </body>
  </html>""" % (strIntro, constants.WEEKDAYSMAPPING[dateCurrent.weekday()], strDay, convertC(todayAvgTemp), strTemp, strCrimeTemp,
    currentWeatherDescTxt.lower(), compareWeatherDesc.lower(), strCrimeWeather, strTotalChange,  
    round(todayPropertyCrime, 0), compareStatsWeekday['perProperty'], compareStatsTemp['perProperty'], compareStatsWeather['perProperty'], compareStatsWeekday['perProperty'] + compareStatsTemp['perProperty'] + compareStatsWeather['perProperty'], 
  	round(todayPersonCrime, 0), compareStatsWeekday['perPerson'], compareStatsTemp['perPerson'], compareStatsWeather['perPerson'], compareStatsWeekday['perPerson'] + compareStatsTemp['perPerson'] + compareStatsWeather['perPerson'], 
  	round(todaySocietyCrime, 0), compareStatsWeekday['perSociety'], compareStatsTemp['perSociety'], compareStatsWeather['perSociety'], compareStatsWeekday['perSociety'] + compareStatsTemp['perSociety'] + compareStatsWeather['perSociety'],
  	round(todayAvgCrime, 0), compareStatsWeekday['perCrime'], compareStatsTemp['perCrime'], compareStatsWeather['perCrime'], compareStatsWeekday['perCrime'] + compareStatsTemp['perCrime'] + compareStatsWeather['perCrime'],
    strTimeBest, strTimeWorst, 
    "</li><li>".join(nWorstCrime), "</li><li>".join(nBestCrime), "https://www.seattlecrimeforecast.com/Archive/" + strPhoto,
    datetime.now().strftime("%m/%d/%Y at %I:%M:%S %p"))

  f.write(summary)
  f.close()

  dfBeatArchive = pd.DataFrame()
  dfBeatArchive['Index'] = dfBeatData['Index']
  dfBeatArchive['TimeOfDayUTC'] = dfBeatData['TimeOfDayUTC']
  dfBeatArchive['Property Crime'] = dfBeatData['Property Crime']
  dfBeatArchive['Person Crime'] = dfBeatData['Person Crime']
  dfBeatArchive['Society Crime'] = dfBeatData['Society Crime']
  dfBeatArchive['Total Crime'] = dfBeatData['Total Crime']
  #save predictions for later comparisons
  dfBeatArchive.to_csv(constants.URLPREDICTIONS + "ForecastFor" + dateCurrent.strftime("-%m-%d-%Y") +
    "MadeOn" + date.today().strftime("%m-%d-%y-") + str(datetime.now().hour) + ".csv")


  rss = rss + """<item>
    <title>Seattle Crime Forecast for %s</title>
    <link>https://www.seattlecrimeforecast.com/?date=%s</link>
    <description><![CDATA[%s]]></description>
  <guid>https://www.seattlecrimeforecast.com/?date=%s</guid>
  <pubDate>%s</pubDate>
  <dc:creator>Seattle Crime Forecast</dc:creator>
  <enclosure url="%s" type="image/jpeg"/>
  <category>Technology and Crime News</category>
  </item>""" % (dateCurrent.strftime("%B %d, %Y"), dateCurrent.strftime("%m-%d-%Y"), summary,
    dateCurrent.strftime("%m-%d-%Y"), datetime.now().strftime("%Y-%m-%dT%H:%M:%S-08:00:00"), "https://www.seattlecrimeforecast.com/Archive/" + strPhoto)

  dateCurrent = dateCurrent + timedelta(1)

rss = rss + """</channel>
</rss>"""
f = open(constants.RSSPATH + 'SeattleCrimeRSS.xml','w')
f.write(rss)
f.close()





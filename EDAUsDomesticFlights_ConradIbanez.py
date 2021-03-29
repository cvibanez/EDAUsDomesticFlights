#Conrad Ibanez
#DSC530 Winter 2019
#Term Project 
#Exploratory Data Analysis on the following data set- https://www.kaggle.com/usdot/flight-delays

"""This file contains code used in "Think Stats",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import math

import thinkplot
import thinkstats2
import hypothesis
import regression


#import thinkplot
#import thinkstats2

pd.options.display.max_columns = None
pd.options.display.max_rows = None

FORMATS = ['png']

def ReadFlightData():
    """Reads flight data

    returns: DataFrame
    """
    flights = pd.read_csv('flights.csv')
    
    # Convert the date columns to one variable
    flights['DATE_TIME']= pd.to_datetime(flights[['YEAR', 'MONTH', 'DAY']].rename(columns={'YEAR': 'year', 'MONTH': 'month', 'DAY': 'day'}))
    flights['DATE'] = flights['DATE_TIME'].dt.strftime("%Y/%m/%d")
    
    # The following is necessary to have the AIRLINE as a categoricalvariable converted to a numeric variable for analysis
    #  Alaska Airlines will have flights.AIRLINE == 'AS' and flights.AIRLINE_CODE == 0
    flights['AIRLINE_CODE'] = pd.factorize(flights.AIRLINE)[0]

#    flights['DATE'] = flights['DATE'].astype('datetime64')
    return flights

def ReadAirlineData():
    """Reads flight data

    returns: DataFrame
    """
    airlines = pd.read_csv('airlines.csv')
    return airlines

def ReadAirportData():
    """Reads flight data

    returns: DataFrame
    """
    airports = pd.read_csv('airports.csv')
    return airports

def createHistograms(flights, airlines, airports):
    """Create histograms

    """
    histFlights = thinkstats2.Hist(flights.DAY_OF_WEEK, label='DAY_OF_WEEK')
#    print(histFlights.GetDict())
#    print(pd.DataFrame(histFlights.GetDict(), index=[0]))
#    histFlights.Print()
    thinkplot.Hist(histFlights)
    thinkplot.Save(root='FlightDayHistogram', xlabel='day', ylabel='frequency')
#    thinkplot.Show(xlabel='day', ylabel='frequency')
    
    print("Mode for DAY_OF_WEEK: ", flights.DAY_OF_WEEK.mode())
    
    histDates = thinkstats2.Hist(flights.DATE, label='DATE')
#    print(histDates.GetDict())
#    print(pd.DataFrame(histDates.GetDict(), index=[0]))
#    histDates.Print()
    thinkplot.Hist(histDates)
    thinkplot.Save(root='FlightDateHistogram', xlabel='date', ylabel='frequency')
#    thinkplot.Show(xlabel='date', ylabel='frequency', figsize=(25, 20))
    
    print("Mode for DATE: ", flights.DATE.mode())
    
    histAirlines = thinkstats2.Hist(flights.AIRLINE, label='AIRLINE')
#    print(histAirlines.GetDict())
#    histAirlines.Print()
    thinkplot.Hist(histAirlines)
    thinkplot.Save(root='FlightAirlineHistogram', xlabel='airline', ylabel='frequency')
#    thinkplot.Show(xlabel='airline', ylabel='frequency')
    
    print("Mode for AIRLINE: ", flights.AIRLINE.mode())
    
    histOrigin = thinkstats2.Hist(flights.ORIGIN_AIRPORT, label='ORIGIN_AIRPORT')
#    print(histOrigin.GetDict())
#    print(pd.DataFrame(histOrigin.GetDict(), index=[0]))
#    histOrigin.Print()
    thinkplot.Hist(histOrigin)
    thinkplot.Save(root='FlightOriginAirportHistogram', xlabel='origin airport', ylabel='frequency')
#    thinkplot.Show(xlabel='origin airport', ylabel='frequency')
    
    print("Mode for ORIGIN_AIRPORT: ", flights.ORIGIN_AIRPORT.mode())
    
    histDest = thinkstats2.Hist(flights.DESTINATION_AIRPORT, label='DESTINATION_AIRPORT')
#    print(histDest.GetDict())
#    print(pd.DataFrame(histDest.GetDict(), index=[0]))
#    histDest.Print()
    thinkplot.Hist(histDest)
    thinkplot.Save(root='FlightDestinationHistogram', xlabel='destination airport', ylabel='frequency')
#    thinkplot.Show(xlabel='destination airport', ylabel='frequency')
    
    print("Mode for DESTINATION_AIRPORT: ", flights.DESTINATION_AIRPORT.mode())
    
    delayedDepart = flights[flights.DEPARTURE_DELAY > 0]
    histDelayedDepart = thinkstats2.Hist(delayedDepart.DEPARTURE_DELAY, label='DEPARTURE_DELAY')
    thinkplot.Hist(histDelayedDepart)
    thinkplot.Save(root='FlightDepartureDelayHistogram', xlabel='departure delay', ylabel='frequency')
#    thinkplot.Show(xlabel='departure delay in min', ylabel='frequency')
    
    print("Mode for DEPARTURE_DELAY: ", flights.DEPARTURE_DELAY.mode())
    print("Mean for DEPARTURE_DELAY: ", flights.DEPARTURE_DELAY.mean())
    print("Var for DEPARTURE_DELAY: ", flights.DEPARTURE_DELAY.var())
    print("STD for DEPARTURE_DELAY: ", flights.DEPARTURE_DELAY.std())
    
    delayedArrival = flights[flights.ARRIVAL_DELAY > 0]
    histDelayedArrival = thinkstats2.Hist(delayedArrival.ARRIVAL_DELAY, label='ARRIVAL_DELAY')
    thinkplot.Hist(histDelayedArrival)
    thinkplot.Save(root='FlightArrivalDelayHistogram', xlabel='arrival delay', ylabel='frequency')
#    thinkplot.Show(xlabel='arrival delay in min', ylabel='frequency')
    
    print("Mode for DEPARTURE_DELAY: ", flights.ARRIVAL_DELAY.mode())
    print("Mean for DEPARTURE_DELAY: ", flights.ARRIVAL_DELAY.mean())
    print("Var for DEPARTURE_DELAY: ", flights.ARRIVAL_DELAY.var())
    print("STD for DEPARTURE_DELAY: ", flights.ARRIVAL_DELAY.std())
    
  #https://www.kaggle.com/smiller933/fixing-airport-codes
  #https://www.kaggle.com/usdot/flight-delays/discussion/29308
  
def compareAlaskaAirlinesPmf(alaska, others):
    """Create PMF to compare Alaska Airlines versus other airlines
       Per JD Power: Among traditional carriers, Alaska Airlines ranks highest for the 12th consecutive year
       https://www.jdpower.com/business/press-releases/2019-north-america-airline-satisfaction-study 

    """
    alaska_pmf = thinkstats2.Pmf(alaska.ARRIVAL_DELAY, label='Alaska Airlines')
    other_pmf = thinkstats2.Pmf(others.ARRIVAL_DELAY, label='other')
    width = 0.45

    thinkplot.PrePlot(2, cols=2)
    thinkplot.Hist(alaska_pmf, align='right', width=width)
    thinkplot.Hist(other_pmf, align='left', width=width)
    thinkplot.Save(root='-10to10ArrivalDelayBarPMF', title='-10 to 10 min Arrival Delay', xlabel='arrival delay',
                     ylabel='probability -10 to 10 mins',
                     axis=[-10, 10, 0, 0.032])
    
    thinkplot.PrePlot(2)
    thinkplot.SubPlot(2)
    thinkplot.Pmfs([alaska_pmf, other_pmf])
    thinkplot.Save(root='-10to10ArrivalDelayStepPMF', title='-10 to 10 min Arrival Delay', xlabel='arrival delay',
                     ylabel='probability -10 to 10 mins',
                     axis=[-10, 10, 0, 0.032])
    
    
    thinkplot.PrePlot(2, cols=2)
    thinkplot.Hist(alaska_pmf, align='right', width=width)
    thinkplot.Hist(other_pmf, align='left', width=width)
    thinkplot.Save(root='-20to0ArrivalDelayBarPMF',title='-20 to 0 min Arrival Delay', xlabel='arrival delay -20 to 0 mins',
                     ylabel='probability',
                     axis=[-20, 0, 0, 0.032])

    thinkplot.PrePlot(2)
    thinkplot.SubPlot(2)
    thinkplot.Pmfs([alaska_pmf, other_pmf])
    thinkplot.Save(root='-20to0ArrivalDelayStepPMF', xlabel='arrival delay -20 to 0 mins',
                     ylabel='probability',
                     axis=[-20, 0, 0, 0.032])
    
    
    thinkplot.PrePlot(2, cols=2)
    thinkplot.Hist(alaska_pmf, align='right', width=width)
    thinkplot.Hist(other_pmf, align='left', width=width)
    thinkplot.Save(root='0to20ArrivalDelayBarPMF', title='0 to 20 min Arrival Delay', xlabel='arrival delay 0 to 20 mins',
                     ylabel='probability',
                     axis=[0, 20, 0, 0.032])
    
    thinkplot.PrePlot(2)
    thinkplot.SubPlot(2)
    thinkplot.Pmfs([alaska_pmf, other_pmf])
    thinkplot.Save(root='0to20ArrivalDelayStepPMF', title='0 to 20 min Arrival Delay', xlabel='arrival delay 0 to 20 mins',
                     ylabel='probability',
                     axis=[0, 20, 0, 0.032])
    
    
    thinkplot.PrePlot(2, cols=2)
    thinkplot.Hist(alaska_pmf, align='right', width=width)
    thinkplot.Hist(other_pmf, align='left', width=width)
    thinkplot.Save(root='-40to20ArrivalDelayBarPMF', title='-40 to -20 min Arrival Delay', xlabel='arrival delay -40 to -20 mins',
                     ylabel='probability',
                     axis=[-40, -20, 0, 0.032])
    thinkplot.Show()
    thinkplot.PrePlot(2)
    thinkplot.SubPlot(2)
    thinkplot.Pmfs([alaska_pmf, other_pmf])
    thinkplot.Save(root='-40to-20ArrivalDelayStepPMF',title='-40 to -20 min Arrival Delay', xlabel='arrival delay -40 to -20 mins',
                     ylabel='probability',
                     axis=[-40, -20, 0, 0.032])
  
    
    thinkplot.PrePlot(2, cols=2)
    thinkplot.Hist(alaska_pmf, align='right', width=width)
    thinkplot.Hist(other_pmf, align='left', width=width)
    thinkplot.Save(root='20to40ArrivalDelayBarPMF',title='20 to 40 min Arrival Delay', xlabel='arrival delay 20 to 40 mins',
                     ylabel='probability',
                     axis=[20, 40, 0, 0.032])

    thinkplot.PrePlot(2)
    thinkplot.SubPlot(2)
    thinkplot.Pmfs([alaska_pmf, other_pmf])
    thinkplot.Save(root='20to40ArrivalDelayStepPMF', title='20 to 40 min Arrival Delay', xlabel='arrival delay 20 to 40 mins',
                     ylabel='probability',
                     axis=[20, 40, 0, 0.032])
    
"""
    thinkplot.PrePlot(2)
    thinkplot.SubPlot(2)
    thinkplot.Pmfs([first_pmf, other_pmf])
    thinkplot.Save(root='probability_nsfg_pmf',
                   xlabel='weeks',
                   axis=[27, 46, 0, 0.6])
"""

def compareAlaskaAirlinesCdf(alaska, others):
    """Create CDF to compare Alaska Airlines versus other airlines
       Per JD Power: Among traditional carriers, Alaska Airlines ranks highest for the 12th consecutive year
       https://www.jdpower.com/business/press-releases/2019-north-america-airline-satisfaction-study 

    """
    # plot CDFs of arrival delays for alaska airlines and others
    alaska_cdf = thinkstats2.Cdf(alaska.ARRIVAL_DELAY, label='Alaska Airlines')
    other_cdf = thinkstats2.Cdf(others.ARRIVAL_DELAY, label='other')

    thinkplot.PrePlot(2)
    thinkplot.Cdfs([alaska_cdf, other_cdf])
#    thinkplot.Show(xlabel='arrival delay (min)', ylabel='CDF', axis=[-20, 40, 0, 1])
    thinkplot.Save(root='AlaskaAirlines_ArrivalDelay_cdf',
                   title='Arrival delay',
                   xlabel='arrival delay (min)',
                   ylabel='CDF',
                   axis=[-20, 40, 0, 1]
                   )
    
    
   

def compareDetroitAirport(flights):
    """Create PMF to compare Atlanta airport versus other airports
       Per JD Power: Detroit Metropolitan Wayne County Airport ranks highest in passenger satisfaction among mega airports with a score of 786. 
       https://www.jdpower.com/business/press-releases/2019-north-america-airport-satisfaction-study 

    """
    detroit = flights[flights.DESTINATION_AIRPORT == 'DTW']
    others = flights[flights.AIRLINE != 'DTW']
    detroit_pmf = thinkstats2.Pmf(detroit.ARRIVAL_DELAY, label='Detroit Metro Arrival Delay')
    other_pmf = thinkstats2.Pmf(others.ARRIVAL_DELAY, label='other')
    width = 0.45

    thinkplot.PrePlot(2, cols=2)
    thinkplot.Hist(detroit_pmf, align='right', width=width)
    thinkplot.Hist(other_pmf, align='left', width=width)
    thinkplot.Save(root='-100to100DetroitDelayBarPMF', title='-100 to 100 min Arrival Delay', xlabel='detroit metro arrival delay',
                     ylabel='probability -100 to 100 mins',
                     axis=[-100, 100, 0, 0.032])
   
    thinkplot.PrePlot(2)
    thinkplot.SubPlot(2)
    thinkplot.Pmfs([detroit_pmf, other_pmf])
    thinkplot.Save(root='-100to100DetroitDelayStepPMF', title='-100 to 100 min Arrival Delay', xlabel='detroit metro arrival delay',
                     ylabel='probability -100 to 100 mins',
                     axis=[-100, 100, 0, 0.032])
    
    thinkplot.PrePlot(2, cols=2)
    thinkplot.Hist(detroit_pmf, align='right', width=width)
    thinkplot.Hist(other_pmf, align='left', width=width)
    thinkplot.Save(root='-30to30DetroitDelayBarPMF', title='-30 to 30 min Arrival Delay', xlabel='detroit metro arrival delay',
                     ylabel='probability -30 to 30 mins',
                     axis=[-30, 30, 0, 0.032])
    
    thinkplot.PrePlot(2)
    thinkplot.SubPlot(2)
    thinkplot.Pmfs([detroit_pmf, other_pmf])
    thinkplot.Save(root='-30to30DetroitDelayStepPMF', title='-30 to 30 min Arrival Delay', xlabel='detroit metro arrival delay',
                     ylabel='probability -30 to 30 mins',
                     axis=[-30, 30, 0, 0.032])
    
    
    thinkplot.PrePlot(2, cols=2)
    thinkplot.Hist(detroit_pmf, align='right', width=width)
    thinkplot.Hist(other_pmf, align='left', width=width)
    thinkplot.Save(root='-60to0DetroitDelayBarPMF', title='-60 to 0 min Arrival Delay', xlabel='detroit metro arrival delay',
                     ylabel='probability -60 to 0 mins',
                     axis=[-60, 0, 0, 0.032])
    
    thinkplot.PrePlot(2)
    thinkplot.SubPlot(2)
    thinkplot.Pmfs([detroit_pmf, other_pmf])
    thinkplot.Save(root='-60to0DetroitDelayStepPMF', title='-60 to 0 min Arrival Delay', xlabel='detroit metro arrival delay',
                     ylabel='probability -60 to 0 mins',
                     axis=[-60, 0, 0, 0.032])
    
    thinkplot.PrePlot(2, cols=2)
    thinkplot.Hist(detroit_pmf, align='right', width=width)
    thinkplot.Hist(other_pmf, align='left', width=width)
    thinkplot.Save(root='0to60DetroitDelayBarPMF', title='0 to 60 min Arrival Delay', xlabel='detroit metro arrival delay',
                     ylabel='probability 0 to 60 mins',
                     axis=[0, 60, 0, 0.032])
    
    thinkplot.PrePlot(2)
    thinkplot.SubPlot(2)
    thinkplot.Pmfs([detroit_pmf, other_pmf])
    thinkplot.Save(root='0to60DetroitDelayStepPMF',title='0 to 60 min Arrival Delay', xlabel='detroit metro arrival delay',
                     ylabel='probability 0 to 60 mins',
                     axis=[0, 60, 0, 0.032])
    


def compareDay4(flights):
    """Create PMF to compare Day 4 (Thursday) with other days.
       I chose Day 4 (Thursday) because it showed the most flights for that day in the scatterplot

    """
    labelString = "Day 4 Arrival Delay"
    xLabelString = "day 4 arrival delay"
    day = flights[flights.DAY == 4]
    others = flights[flights.DAY != 4]
    day_pmf = thinkstats2.Pmf(day.ARRIVAL_DELAY, label=labelString)
    other_pmf = thinkstats2.Pmf(others.ARRIVAL_DELAY, label='other')
    width = 0.45

    thinkplot.PrePlot(2, cols=2)
    thinkplot.Hist(day_pmf, align='right', width=width)
    thinkplot.Hist(other_pmf, align='left', width=width)
    thinkplot.Save(root='Thursday-100to100ArrivalDelayBarPMF', title='-100 to 100 min Arrival Delay', xlabel=xLabelString,
                     ylabel='probability -100 to 100 mins',
                     axis=[-100, 100, 0, 0.032])

    thinkplot.PrePlot(2)
    thinkplot.SubPlot(2)
    thinkplot.Pmfs([day_pmf, other_pmf])
    thinkplot.Save(root='Thursday-100to100ArrivalDelayStepPMF', title='-100 to 100 min Arrival Delay', xlabel=xLabelString,
                     ylabel='probability -100 to 100 mins',
                     axis=[-100, 100, 0, 0.032])
    
    
    thinkplot.PrePlot(2, cols=2)
    thinkplot.Hist(day_pmf, align='right', width=width)
    thinkplot.Hist(other_pmf, align='left', width=width)
    thinkplot.Save(root='Thursday-30to30ArrivalDelayBarPMF', title='-30 to 30 min Arrival Delay', xlabel=xLabelString,
                     ylabel='probability -30 to 30 mins',
                     axis=[-30, 30, 0, 0.032])

    thinkplot.PrePlot(2)
    thinkplot.SubPlot(2)
    thinkplot.Pmfs([day_pmf, other_pmf])
    thinkplot.Save(root='Thursday-30to30ArrivalDelayStepPMF',title='-30 to 30 min Arrival Delay', xlabel=xLabelString,
                     ylabel='probability -30 to 30 mins',
                     axis=[-30, 30, 0, 0.032])
  
    
    thinkplot.PrePlot(2, cols=2)
    thinkplot.Hist(day_pmf, align='right', width=width)
    thinkplot.Hist(other_pmf, align='left', width=width)
    thinkplot.Save(root='Thursday-60to0ArrivalDelayBarPMF', title='-60 to 0 min Arrival Delay', xlabel=xLabelString,
                     ylabel='probability -60 to 0 mins',
                     axis=[-60, 0, 0, 0.032])
    
    thinkplot.PrePlot(2)
    thinkplot.SubPlot(2)
    thinkplot.Pmfs([day_pmf, other_pmf])
    thinkplot.Save(root='Thursday-60to0ArrivalDelayStepPMF', title='-60 to 0 min Arrival Delay', xlabel=xLabelString,
                     ylabel='probability -60 to 0 mins',
                     axis=[-60, 0, 0, 0.032])
 
    
    thinkplot.PrePlot(2, cols=2)
    thinkplot.Hist(day_pmf, align='right', width=width)
    thinkplot.Hist(other_pmf, align='left', width=width)
    thinkplot.Save(root='Thursday0to60ArrivalDelayBarPMF', title='0 to 60 min Arrival Delay', xlabel=xLabelString,
                     ylabel='probability 0 to 60 mins',
                     axis=[0, 60, 0, 0.032])
 
    thinkplot.PrePlot(2)
    thinkplot.SubPlot(2)
    thinkplot.Pmfs([day_pmf, other_pmf])
    thinkplot.Save(root='Thursday0to60ArrivalDelayStepPMF', title='0 to 60 min Arrival Delay', xlabel=xLabelString,
                     ylabel='probability 0 to 60 mins',
                     axis=[0, 60, 0, 0.032])

def MakeNormalModel(arrivalDelays):
    """Plot the CDF of arrival delays with a normal model.
       This is a modified copy from analytic.py
    """
    
    # estimate parameters: trimming outliers yields a better fit
    mu, var = thinkstats2.TrimmedMeanVar(arrivalDelays, p=0.01)
    print('Mean, Var', mu, var)
    
    # plot the model
    sigma = math.sqrt(var)
    print('Sigma', sigma)
    xs, ps = thinkstats2.RenderNormalCdf(mu, sigma, low=0, high=12.5)

    thinkplot.Plot(xs, ps, label='model', color='0.8')

    # plot the data
    cdf = thinkstats2.Cdf(arrivalDelays, label='data')

    thinkplot.PrePlot(1)
    thinkplot.Cdf(cdf) 
    thinkplot.Save(root='NormalModel_arrivaldelay_model',
                   title='Arrival Delays',
                   xlabel='arrival delays (min)',
                   ylabel='CDF')

def MakeNormalPlot(arrivalDelays):
    """Generate the normal probability plot for the arrival delays.
       This is a modified copy from analytic.py
    """
    
    mean = arrivalDelays.mean();
    std = arrivalDelays.std()

    xs = [-4, 4]
    fxs, fys = thinkstats2.FitLine(xs, mean, std)
    thinkplot.Plot(fxs, fys, linewidth=4, color='0.8')

    thinkplot.PrePlot(2) 
    xs, ys = thinkstats2.NormalProbability(arrivalDelays)
    thinkplot.Plot(xs, ys, label='arrival delays (min)')

    thinkplot.Save(root='NormalModel_arrivaldelay_normalplot',
                   title='Normal probability plot',
                   xlabel='Standard deviations from mean',
                   ylabel='Arrival Delays (min)')
    

def MakeAirlineArrivalDelayScatterPlots(flights):
    """Make scatterplots.
    """
    sample = thinkstats2.SampleRows(flights, 10000)

    # simple scatter plot
    thinkplot.PrePlot(cols=2)
#    airports, arrivalDelays = GetAirlineArrivalDelay(sample)
    airlines = sample.AIRLINE_CODE
    arrivalDelays = sample.ARRIVAL_DELAY
#    ScatterPlot(airports, arrivalDelays)

    # scatter plot with jitter
#    thinkplot.SubPlot(2)
#    heights, weights = GetAirlineArrivalDelay(sample, hjitter=1.3, wjitter=0.5)
#    ScatterPlot(heights, weights)
    
    thinkplot.Scatter(airlines, arrivalDelays, alpha=1)
    thinkplot.Config(xlabel='airline',
                     ylabel='arrival delay (min)',
#                     axis=[-20, 20, 20, 200],
                     legend=False)

    thinkplot.Save(root='AirlineArrivalDelayScatterplot')
    
def GetArrivalDepartureDelay(flights, hjitter=0.0, wjitter=0.0):
    """Get sequences of airports and arrival delays.

    df: 
    hjitter: float magnitude of random noise added to heights
    wjitter: float magnitude of random noise added to weights

    returns: tuple of sequences (airport, arrivaldelays)
    """
    arrivalDelays = flights.ARRIVAL_DELAY
    if hjitter:
        arrivalDelays = thinkstats2.Jitter(arrivalDelays, hjitter)

    departureDelays = flights.DEPARTURE_DELAY
    if wjitter:
        departureDelays = thinkstats2.Jitter(departureDelays, wjitter)

    return arrivalDelays, departureDelays




def MakeArrivalDepartureDelayScatterPlots(flights):
    """Make scatterplots.
    """
    sample = thinkstats2.SampleRows(flights, 10000)

    # simple scatter plot
    thinkplot.PrePlot(cols=2)
#    departureDelays, arrivalDelays = GetArrivalDepartureDelay(sample)
#    airports = sample.AIRLINE
#   arrivalDelays = sample.ARRIVAL_DELAY
#    ScatterPlot(airports, arrivalDelays)

    # scatter plot with jitter
#    thinkplot.SubPlot(2)
    departureDelays, arrivalDelays = GetArrivalDepartureDelay(sample, hjitter=1.3, wjitter=0.5)
       
    thinkplot.Scatter(arrivalDelays, departureDelays, alpha=1)
    thinkplot.Config(xlabel='arrival delay (min)',
                     ylabel='departure delay (min)',
#                     axis=[-20, 20, 20, 200],
                     legend=False)

    thinkplot.Save(root='ArrivalDepartureDelayScatterplot')

def ComputeArrivalDepartureDelayCorrelations(flights):
    """Compute the different correlations.
        This is similar to Correlations() in scatter.py
    """
    flights = flights.dropna(subset=['ARRIVAL_DELAY', 'DEPARTURE_DELAY'])
    print('pandas cov', flights.ARRIVAL_DELAY.cov(flights.DEPARTURE_DELAY))
    print('thinkstats2 Cov', thinkstats2.Cov(flights.ARRIVAL_DELAY, flights.DEPARTURE_DELAY))
    print()

    print('pandas corr Pearson', flights.ARRIVAL_DELAY.corr(flights.DEPARTURE_DELAY))
    print('thinkstats2 Corr Pearson', thinkstats2.Corr(flights.ARRIVAL_DELAY, flights.DEPARTURE_DELAY))
    print()

    print('pandas corr spearman', flights.ARRIVAL_DELAY.corr(flights.DEPARTURE_DELAY, method='spearman'))
    print('thinkstats2 SpearmanCorr', 
          thinkstats2.SpearmanCorr(flights.ARRIVAL_DELAY, flights.DEPARTURE_DELAY))
    print()
    
def ComputeAirlineArrivalDelayCorrelations(flights):
    """Compute the different correlations.
        This is similar to Correlations() in scatter.py
    """
    flights = flights.dropna(subset=['AIRLINE', 'ARRIVAL_DELAY'])
    print('pandas cov', flights.AIRLINE_CODE.cov(flights.ARRIVAL_DELAY))
    print('thinkstats2 Cov', thinkstats2.Cov(flights.AIRLINE_CODE, flights.ARRIVAL_DELAY))
    print()

    print('pandas corr Pearson', flights.AIRLINE_CODE.corr(flights.ARRIVAL_DELAY))
    print('thinkstats2 Corr Pearson', thinkstats2.Corr(flights.AIRLINE_CODE, flights.ARRIVAL_DELAY))
    print()

    print('pandas corr spearman', flights.AIRLINE_CODE.corr(flights.ARRIVAL_DELAY, method='spearman'))
    print('thinkstats2 SpearmanCorr', 
          thinkstats2.SpearmanCorr(flights.AIRLINE_CODE, flights.ARRIVAL_DELAY))
    print()

def RunAlaskaTests(data, iters=1000):
    """Test the difference in means between Alaska Airlines and other airlines
    """
 
    # test the difference in means
    ht = hypothesis.DiffMeansPermute(data)
    p_value = ht.PValue(iters=iters)
    print('\nmeans permute two-sided')
    hypothesis.PrintTest(p_value, ht)

    ht.PlotCdf()
    thinkplot.Save(root='hypothesis1 Alaska',
                   title='Permutation test',
                   xlabel='difference in means (min)',
                   ylabel='CDF',
                   legend=False) 
    
    # test the difference in means one-sided
    ht = hypothesis.DiffMeansOneSided(data)
    p_value = ht.PValue(iters=iters)
    print('\nmeans permute one-sided')
    hypothesis.PrintTest(p_value, ht)

    # test the difference in std
    ht = hypothesis.DiffStdPermute(data)
    p_value = ht.PValue(iters=iters)
    print('\nstd permute one-sided')
    hypothesis.PrintTest(p_value, ht)
    
def PlotAirlineArrivalDelayFit(flights):
    """Plots a scatter plot and fitted curve.

    live: DataFrame
    """
    
    sample = thinkstats2.SampleRows(flights, 1000)
    airlineCodes = sample.AIRLINE_CODE
    arrivalDelays = sample.ARRIVAL_DELAY
    inter, slope = thinkstats2.LeastSquares(airlineCodes, arrivalDelays)
    fit_xs, fit_ys = thinkstats2.FitLine(airlineCodes, inter, slope)

    thinkplot.Scatter(airlineCodes, arrivalDelays, color='gray', alpha=0.1)
    thinkplot.Plot(fit_xs, fit_ys, color='white', linewidth=3)
    thinkplot.Plot(fit_xs, fit_ys, color='blue', linewidth=2)
    thinkplot.Save(root='AirlineArrivalDelayFit_linear1',
                   xlabel='airline',
                   ylabel='arrival delay (min)',
#                   axis=[10, 45, 0, 15],
                   legend=False)
    
    formula = 'ARRIVAL_DELAY ~ AIRLINE_CODE'
    model = smf.ols(formula, data=sample)
    results = model.fit()
    regression.SummarizeResults(results)
    
def PlotArrivalDepartureDelayFit(flights):
    """Plots a scatter plot and fitted curve.

    live: DataFrame
    """
    
    sample = thinkstats2.SampleRows(flights, 1000)
    arrivalDelays = sample.ARRIVAL_DELAY
    departureDelays = sample.DEPARTURE_DELAY
    inter, slope = thinkstats2.LeastSquares(arrivalDelays, departureDelays)
    fit_xs, fit_ys = thinkstats2.FitLine(arrivalDelays, inter, slope)

    thinkplot.Scatter(arrivalDelays, departureDelays, color='gray', alpha=0.1)
    thinkplot.Plot(fit_xs, fit_ys, color='white', linewidth=3)
    thinkplot.Plot(fit_xs, fit_ys, color='blue', linewidth=2)
    thinkplot.Save(root='ArrivalDepartureDelayFit_linear1',
                   xlabel='arrival delay (min)',
                   ylabel='departure delay (min)',
#                   axis=[10, 45, 0, 15],
                   legend=False)
    
    formula = 'DEPARTURE_DELAY ~ ARRIVAL_DELAY'
    model = smf.ols(formula, data=sample)
    results = model.fit()
    regression.SummarizeResults(results)
    
def main():
    
    thinkstats2.RandomSeed(17)
    
    flights = ReadFlightData()
#    print(flights.head())
    
#    print(flights.DESTINATION_AIRPORT.to_string(index=False))
    
    airlines = ReadAirlineData()
#    print(airlines.head())
    
    airports = ReadAirportData()
#    print(airports.head())

    """ A minimum of 5 variables in your dataset used during your analysis (for help with selecting, the author made his selection on page 6 of your book). Consider what you think could have an impact on your question – remember this is never perfect, so don’t be worried if you miss one (Chapter 1).
    Describe what the 5 variables mean in the dataset (Chapter 1).
    DAY_OF_WEEK - Integer 1 - 7 corresponding to the day of the week.  1 is Monday and 7 is Sunday.
    AIRLINE - Letter code corresponding to the airline for the flight.
    ORIGIN_AIRPORT - Airport code corresponding to the flight's origin airport.
    DESTINATION_AIRPORT - Airport code corresponding to the flight's destination airport.
    DEPARTURE_DELAY - Integer value corresponding to the departure delay for the flight. Computed from SCHEDULED_DEPARTURE and DEPARTURE_TIME.
    ARRIVAL_DELAY - Integer value corresponding to the arrival delay for the flight.  Computed from SCHEDULED_ARRIVAL and ARRIVAL_TIME.
    """
    
    """Include a histogram of each of the 5 variables – in your summary and analysis, identify any outliers and explain the reasoning for them being outliers and how you believe they should be handled (Chapter 2).  
       Include the other descriptive characteristics about the variables: Mean, Mode, Spread, and Tails (Chapter 2).
    """   
    createHistograms(flights, airlines, airports)
    
    alaska = flights[flights.AIRLINE == 'AS']
#    print(alaska.head())
    notAlaska = flights[flights.AIRLINE != 'AS']
#    print(notAlaska.head())


    """Using pg. 29 of your text as an example, compare two scenarios in your data using a PMF. 
       Reminder, this isn’t comparing two variables against each other – it is the same variable, 
       but a different scenario. Almost like a filter. The example in the book is first babies compared 
       to all other babies, it is still the same variable, but breaking the data out based on criteria 
       we are exploring (Chapter 3).
    """
    compareAlaskaAirlinesPmf(alaska, notAlaska)
    compareDetroitAirport(flights)
    compareDay4(flights)


    """ Create 1 CDF with one of your variables, using page 41-44 as your guide, what does this tell you 
        about your variable and how does it address the question you are trying to answer (Chapter 4).
    
    """    
    compareAlaskaAirlinesCdf(alaska, notAlaska)
    
    arrivalDelays = flights.ARRIVAL_DELAY.dropna()


    """ Plot 1 analytical distribution and provide your analysis on how it applies to the dataset you have chosen (Chapter 5).
    
    """
    MakeNormalModel(arrivalDelays)
    MakeNormalPlot(arrivalDelays)
  
    
    """ Create two scatter plots comparing two variables and provide your analysis on correlation and causation. 
        Remember, covariance, Pearson’s correlation, and Non-Linear Relationships should also be considered during 
        your analysis (Chapter 7).
    """   
    
    MakeAirlineArrivalDelayScatterPlots(flights)
    MakeArrivalDepartureDelayScatterPlots(flights)
    ComputeArrivalDepartureDelayCorrelations(flights)
    ComputeAirlineArrivalDelayCorrelations(flights)
    
    # Remove data with missing arrival delay
    # It seems most of the rows in the set with missing arrival delay is also missing values for other attributes
    # I do not feel this will have an impact for this analysis.
    """ Conduct a test on your hypothesis using one of the methods covered in Chapter 9.
    """
    hypothesisTestData= alaska.ARRIVAL_DELAY.dropna().values, notAlaska.ARRIVAL_DELAY.dropna().values
    RunAlaskaTests(hypothesisTestData)

    
    """ For this project, conduct a regression analysis on either one dependent and one explanatory variable, 
        or multiple explanatory variables (Chapter 10 & 11).  
    """
    PlotAirlineArrivalDelayFit(flights)
    PlotArrivalDepartureDelayFit(flights)
    
if __name__ == '__main__':
    main()

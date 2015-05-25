
# coding: utf-8

# #Inmar - predicting flight delay
# This Python code is written to be run on Spark.  The model predicts whether a flight will be delayed by 15 minutes or more.  The independent variables are the max and min temperature reported on the day of the flight.
# 
# A logistic regression algorithm was used because the outcome is binary.  I also thought about running standard regression model that would predict the amount of time delayed and then count the number of flights delayed past 15 minutes.

# ##Plan for Deliverables
# The request from Inmar is to spin up three nodes using Horton Works's solution, create a model that predicts flights delayed by 15 minutes or greater, and write the code in Scala.
# 
# As with most software projects the ideal can take some time.  Therefore, building on an Agile Methodology process, I broke the project down into deliverables.  The first is to write the code in Python.  This would allow working through the parsing of the data in a language in which I have delivered more often.  The follow-on deliverables are 1- write the code in Scala and 2- spin up three nodes for the code to run on the full datasets.

# ##Sources
# In my research of logistic regression in MLlib, I came across [this blog](https://samarthbhargav.wordpress.com/2014/04/22/logistic-regression-in-apache-spark/) and [this Spark documentation](https://spark.apache.org/docs/1.3.0/mllib-linear-methods.html).  I borrowed their code and repurposed based on the data.

# ##Training and Test Data
# I used the 2007 sets as the training sets and the 2008 sets as the test sets.  A possible fourth deliverable could be to combine the sets and then create a 70/30 split.

# ##Code

from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
from numpy import array

# number of nodes - used for number of partitions
nodes = 13

# ###Read data in from airline

def air_parse(x):
    '''
    air_parse takes in the line and parses the date and the delayed data point.
    The date is created in the format of YYYYMMDD, which is consistent with the date in the weather data.
    The delayed data point is 1 if greater than or equal to 15 and 0 otherwise. 
    '''
    air_date = x[0]
    # add leading zero if month is single digit
    if len(x[1]) == 1:
        air_date = air_date + '0'
    air_date = air_date + x[1]
        # add leading zero if month is single digit
    if len(x[2]) == 1:
        air_date = air_date + '0'
    air_date = air_date + x[2]

    # if delayed more than 15 minutes
    if int(x[15]) >= 15:
        air_delay_15 = 1
    else:
        air_delay_15 = 0

    return([air_date, air_delay_15])

train_air = sc.textFile('gs://donorbureaudata/inmar/airline/2007.csv', (nodes * 4))

# filter out header row
# train_air = train_air.filter(lambda i: 'Year' not in i)
# filter on rows that have 2007 - removes header and extra rows
train_air = train_air.filter(lambda i: '2007' in i)

# filter out canceled flights
train_air = train_air.filter(lambda i: i.split(',')[21] == '0')
    
train_air = train_air.map(lambda i: i.split(',')).map(air_parse)
#                           l.split(',')).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[3]))))

train_air.take(10)

# ###Read data from weather

sc.textFile('gs://donorbureaudata/full_data_rand/*', (nodes * 4))
train_tempmax = sc.textFile('gs://donorbureaudata/inmar/weather/2007.csv', (nodes * 4))

# filter on TMAX and USW00094846
train_tempmax = train_tempmax.filter(lambda i: 'TMAX' in i)
train_tempmax = train_tempmax.filter(lambda i: 'USW00094846' in i)

train_tempmax = train_tempmax.map(lambda i: i.split(',')).map(lambda i: [i[1], int(i[3])])

train_tempmax.collect()

train_tempmin = sc.textFile('gs://donorbureaudata/inmar/weather/2007.csv', (nodes * 4))

# filter on TMAX and USW00094846
train_tempmin = train_tempmin.filter(lambda i: 'TMIN' in i)
train_tempmin = train_tempmin.filter(lambda i: 'USW00094846' in i)

train_tempmin = train_tempmin.map(lambda i: i.split(',')).map(lambda i: [i[1], int(i[3])])

# ###Join the two data sets

train_join_data = train_air.join(train_tempmax.join(train_tempmin))

# ###Create the LabeledPoints for building the model

def mapper(line):
    """
    mapper converts the joined datasets into a LabeledPoint - Label and Features
    """    
    #### ***feats = line.strip().split(",") 
    # labels must be at the beginning for LRSGD, it's in the end in our data, so 
    # putting it in the right place
    label = line[1][0]
    feats = [line[1][1][0], line[1][1][1]]
    feats.insert(0,label)
    features = [ float(feature) for feature in feats ] # need floats
    return(LabeledPoint(features[0], features[1:]))

train_model_data = train_join_data.map(mapper)

train_model_data.persist()


# ###Train the data on 2007 data

# Train model
model = LogisticRegressionWithSGD.train(train_model_data)


# ###Parse 2008 data for testing the model

# ###Airline data

test_air = sc.textFile('gs://donorbureaudata/inmar/airline/2008.csv', (nodes * 4))

# filter out header row
# filter on rows that have 2007 - removes header and extra rows
test_air = test_air.filter(lambda i: '2008' in i)

# filter out canceled flights
test_air = test_air.filter(lambda i: i.split(',')[21] == '0')
    
test_air = test_air.map(lambda i: i.split(',')).map(air_parse)
#                           l.split(',')).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[3]))))

test_air.take(10)

# ###Weather Data

test_tempmax = sc.textFile('gs://donorbureaudata/inmar/weather/2008.csv', (nodes * 4))

# filter on TMAX and USW00094846
test_tempmax = test_tempmax.filter(lambda i: 'TMAX' in i)
test_tempmax = test_tempmax.filter(lambda i: 'USW00094846' in i)

test_tempmax = test_tempmax.map(lambda i: i.split(',')).map(lambda i: [i[1], int(i[3])])

test_tempmin = sc.textFile('gs://donorbureaudata/inmar/weather/2008.csv', (nodes * 4))

# filter on TMAX and USW00094846
test_tempmin = test_tempmin.filter(lambda i: 'TMIN' in i)
test_tempmin = test_tempmin.filter(lambda i: 'USW00094846' in i)

test_tempmin = test_tempmin.map(lambda i: i.split(',')).map(lambda i: [i[1], int(i[3])])

test_join_data = test_air.join(test_tempmax.join(test_tempmin))

test_join_data.take(10)

# ###Create LabeledPoints for the calculation of error

test_model_data = test_join_data.map(mapper)


# ##Evaluate Model

# Evaluating the model on test data
labels_preds = test_model_data.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labels_preds.filter(lambda (v, p): v != p).count() / float(test_model_data.count())
print("Training Error = " + str(trainErr))

import warnings
warnings.filterwarnings('ignore')
import os
import boto3
import io
import sagemaker
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
%matplotlib inline

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression

import pickle,gzip,urllib,json,csv

from sklearn import preprocessing

from sagemaker import get_execution_role
role = get_execution_role()

s3 = boto3.resource('s3')
bucket_name = 'aws-machinelearning-chez'
object_key = 'weatherhistory2.csv'

s3_client = boto3.client('s3')
response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
response_body = response["Body"].read()
weather = pd.read_csv(io.BytesIO(response_body), header=0, delimiter=",", low_memory=False)

weather.head()

weather.columns

weather.shape

weather.describe()

weather.info()

weather.isnull().any()

weather.isnull().all()

round(100*(weather.isnull().sum()/len(weather.index)),2)

weather['Precip Type'].value_counts()

weather.loc[weather['Precip Type'].isnull(),'Precip Type']='rain'

round(100*(weather.isnull().sum()/len(weather.index)),2)

#Input binary values in type column
weather.loc[weather['Precip Type']=='rain','Precip Type']=1
weather.loc[weather['Precip Type']=='snow','Precip Type']=0

weather_num=weather[list(weather.dtypes[weather.dtypes!='odject'].index)]

weather_y = weather_num.pop('Temperature (C)')
weather_x = weather_num

train_x,test_x,train_y,test_y = train_test_split(weather_x,weather_y,test_size = 0.2,random_state=4)

train_x.head()

test_x.head()

to_drop = ['Formatted Date','Summary','Daily Summary']
weather.drop(to_drop, inplace=True, axis=1)
train_x.drop(to_drop, inplace=True, axis=1)
test_x.drop(to_drop, inplace=True, axis=1)

train_x.head()

test_x.head()

train_x.head()

##Using Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(train_x,train_y)

prediction3 = regressor.predict(test_x)
np.mean((prediction3-test_y)**2)

pd.DataFrame({'actual':test_y,
             'prediction':prediction3,
             'diff':(test_y-prediction3)})

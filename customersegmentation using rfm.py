
"""
Created on Mon Mar 22 12:49:41 2021

@author: simran lopes
"""
#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# reading the dataset
dataset = pd.read_csv('OnlineRetail.csv')
dataset.head(6)
# we observe unique key as customer and common column as counrty
customerdatabycountry=dataset[['Country','CustomerID']].drop_duplicates()
#  hence we group by coutry and observe 
customerdatabycountry.groupby(['Country'])['CustomerID'].agg('count').reset_index().sort_values('CustomerID',ascending=False).head(11)

# WE OBSERVE FROM THE CODE THAT MAJORITYOF THE CUSTOMERS ARE FROM UK HENCE
# HENCE  WWEILL ANALYSE THE DATA FOR THAT COUNTRY AND FILTER OUT THE OTHERS 
dataset = dataset.query("Country=='United Kingdom'").reset_index(drop=True)
# BEFORE WE START THE ANALYSES WE PERFOM DATA CLEANING STEPS
dataset.isnull().sum(axis=0)
dataset = dataset[pd.notnull(dataset['CustomerID'])]
dataset['StockCode'] = dataset['StockCode'].dropna()
dataset.Quantity.min()
dataset.UnitPrice.min()
dataset = dataset[(dataset['Quantity']>0)]
dataset['InvoiceDate'] = pd.to_datetime(dataset['InvoiceDate'])
dataset['TotalAmount'] = dataset['Quantity']*dataset['UnitPrice']


dataset.shape
dataset.head()

#CALCULATING SCORES OF EACH CUSTOMER BASED ON RFM
import datetime as dt
latestdate =dt.datetime(2011, 12, 10)
#  FORMULA FOR RFM
RFMScores = dataset.groupby('CustomerID').agg({'InvoiceDate': lambda x: (latestdate - x.max()).days,
                                               'InvoiceNo': lambda x: len(x), 'TotalAmount': lambda x: x.sum()})
RFMScores['InvoiceDate'] = RFMScores['InvoiceDate'].astype(int)
# MAKING RFM TABLE
RFMScores.rename(columns={'CustomerID':'CustomerID',
                          'InvoiceDate': 'Recency', 
                          'InvoiceNo': 'Frequency', 
                        'TotalAmount': 'Monetary'}, inplace=True)
RFMScores.reset_index().head()





# DESCRIBTION EACH VARIABLE
RFMScores.Recency.describe()
RFMScores.Frequency.describe()
RFMScores.Monetary.describe()

# ASSIGNING quantiles 
quantiles = RFMScores.quantile(q=[0.2,0.4,0.6,0.8])
quantiles = quantiles.to_dict()
quantiles

def Recency_Score(x,p,d):
    if x <= d[p][0.2]:
        return 5
    elif x <= d[p][0.4]:
        return 4
    elif x <= d[p][0.6]: 
        return 3
    elif x <= d[p][0.8]: 
        return 2
    else:
        return 1
    
def Frequency_AND_Monetory_Score(x,p,d):
    if x <= d[p][0.2]:
        return 1
    elif x <= d[p][0.4]:
        return 2
    elif x <= d[p][0.6]: 
        return 3
    elif x <= d[p][0.8]: 
        return 4
    else:
        return 5

RFMScores['R'] = RFMScores['Recency'].apply(Recency_Score, args=('Recency',quantiles,))
RFMScores['F'] = RFMScores['Frequency'].apply(Frequency_AND_Monetory_Score, args=('Frequency',quantiles,))
RFMScores['M'] = RFMScores['Monetary'].apply(Frequency_AND_Monetory_Score, args=('Monetary',quantiles,))


RFMScores.head()

RFMScores['RFMGroup'] = RFMScores.R.map(str) + RFMScores.F.map(str) + RFMScores.M.map(str)
RFMScores['RFMScore'] = RFMScores[['R', 'F', 'M']].sum(axis = 1)

# DESCRIBING THEM BASED ON THIERS SCORES N ASSIGNING THEM CATEGORIES
segt_map= {
  r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at risk',
    r'[1-2]5': 'can\'t loose',
    r'3[1-2]': 'about to sleep',
    r'33': 'need attention',
    r'[3-4][4-5]': 'loyal customers',
    r'41': 'promising',
    r'51': 'new customers',
    r'[4-5][2-3]': 'potential loyalists',
    r'5[4-5]': 'champions'
}

    

RFMScores['Segment'] = RFMScores['R'].map(str) + RFMScores['F'].map(str) 
RFMScores['Segment'] = RFMScores['Segment'].replace(segt_map, regex=True)
RFMScores.head()

# CODE FOR THE top customers  who spends the most among all
RFMScores[RFMScores['RFMGroup']=='555'].sort_values('Monetary', ascending=False).reset_index().head()
RFMScores.sort_values('Monetary', ascending=False).reset_index().head(1)

# ASSIGNING THEM DEALS BASED ON RFM SCORE AND CATEGORY TO ATTRACT THEM TO THE SHOP
deals_map = {
    

        r'[1-2][1-2]': '35% DISCOUNT',
    r'[1-2][3-4]': '30% DISCOUNT',
    r'[1-2]5': 'BUY1GET1',
    r'3[1-2]': 'BUY2GET1',
    r'33': 'BUY3GET1',
    r'[3-4][4-5]': 'free giftwith every purchase',
    r'41': '25% on all products',
    r'51': '20% off all products',
    r'[4-5][2-3]': '35% LIMITED TIME OFFER',
    r'5[4-5]': '30% LIMITED TIME OFFER '
    }


RFMScores['deals'] = RFMScores['R'].map(str) + RFMScores['F'].map(str) 
RFMScores['deals'] = RFMScores['deals'].replace(deals_map, regex=True)
RFMScores.head()




    







































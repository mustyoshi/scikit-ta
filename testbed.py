import mods
import pandas as pd 
import numpy as np
from sklearn.base import TransformerMixin
from scipy.signal import argrelextrema

class InflectionPointTransformer(TransformerMixin):
    def __init__(self, columns=[],outputnames=[],distance=10):
        self.columns = columns
        self.cols = columns
        self.outputnames = outputnames
        self.distance = distance
    def fit(self, x, y=None):
        return self
    def transform(self, dataframe):
        vals = dataframe[self.columns].min(axis=1).values
        indexes = argrelextrema(vals,np.less)
        dataframe[self.outputnames[0]] = 0
        dataframe[self.outputnames[1]] = 0
        for ind in indexes:
            dataframe.set_value(ind,'Support_3Wide',vals[ind])
        vals = dataframe[self.columns].max(axis=1).values
        indexes = argrelextrema(vals,np.greater)
        for ind in indexes:
            dataframe.set_value(ind,'Resist_3Wide',vals[ind])
        #dataframe[self.outputnames[0]] = (dataframe[self.columns].min(axis=1)== dataframe[self.columns].min(axis=1).rolling(window=self.distance,min_periods=0,center=True).min()).astype(int)*dataframe[self.columns].min(axis=1)
        #dataframe[self.outputnames[1]] = (dataframe[self.columns].max(axis=1)== dataframe[self.columns].max(axis=1).rolling(window=self.distance,min_periods=0,center=True).max()).astype(int)*dataframe[self.columns].max(axis=1)
        return dataframe

df = pd.read_csv('bac.us.txt')
std = InflectionPointTransformer(columns=["Close","Open"],outputnames=["Support_3Wide","Resist_3Wide"],distance=3)

std.transform(df).to_csv('out.csv',columns=['Date','Support_3Wide','Resist_3Wide','Open','High','Low','Close','Volume'])
from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd 
class RSITransformer(TransformerMixin):
    def __init__(self, column,outputname,period=14):
        self.column = column
        self.cols = [column]
        self.outputname = outputname
        self.period = period

    def fit(self, x, y=None):
        return self
#Stolen from https://stackoverflow.com/questions/20526414/relative-strength-index-in-python-pandas
    def transform(self, dataframe):
        delta = dataframe[self.column].diff()
        dUp, dDown = delta.copy(), delta.copy()
        dUp[dUp < 0] = 0
        dDown[dDown > 0] = 0

        RolUp = dUp.rolling(window=self.period,center=False).sum()
        #pd.rolling_mean(dUp, self.period)
        RolDown = abs(dDown.rolling(window=self.period,center=False).sum()) #pd.rolling_mean(dDown, self.period).abs()
        RS = (RolUp / RolDown)
        dataframe[self.outputname]= (1-(1/(1+RS)))
        return dataframe[[self.outputname]]
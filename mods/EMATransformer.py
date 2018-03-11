from sklearn.base import TransformerMixin
import numpy as np

class EMATransformer(TransformerMixin):
    def __init__(self, column,outputname,period=4):
        self.column = column
        self.cols = [column]
        self.outputname = outputname
        self.period = period

    def fit(self, x, y=None):
        return self

    def transform(self, dataframe):
        dataframe[self.outputname]= dataframe[self.column].ewm(span=self.period,min_periods=self.period,adjust=True,ignore_na=False).mean()
        return dataframe[[self.outputname]]
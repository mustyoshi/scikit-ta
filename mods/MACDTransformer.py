from sklearn.base import TransformerMixin
import numpy as np

class MACDTransformer(TransformerMixin):
    def __init__(self, macdcolumns=[],outputnames=[],period=9):
        self.macdcolumns = macdcolumns
        self.cols = macdcolumns
        self.outputnames = outputnames
        self.period = period

    def fit(self, x, y=None):
        return self

    def transform(self, dataframe):
        dataframe[self.outputnames[0]]=(dataframe[self.macdcolumns[0]] - dataframe[self.macdcolumns[1]])
        dataframe[self.outputnames[1]] = dataframe[self.outputnames[0]].ewm(span=self.period,min_periods=self.period,adjust=False).mean()
        return dataframe[self.outputnames]
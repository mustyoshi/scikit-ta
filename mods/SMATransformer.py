from sklearn.base import TransformerMixin
import numpy as np

class SMATransformer(TransformerMixin):
    def __init__(self, column,outputname,period=4):
        self.column = column
        self.cols = [column]
        self.outputname = outputname
        self.period = period

    def fit(self, x, y=None):
        return self

    def transform(self, dataframe):
        dataframe[self.outputname]=np.sum([dataframe[self.column].shift(n).astype(float) for n in range(0,self.period)],axis=0)/self.period
        return dataframe[[self.outputname]]
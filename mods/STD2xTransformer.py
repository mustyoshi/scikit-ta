from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd
class STD2xTransformer(TransformerMixin):
    def __init__(self, column="",outputname="",period=0):
        self.column = column
        self.cols=[column]
        self.period = period
        self.outputname = outputname

    def fit(self, x, y=None):
        return self

    def transform(self, dataframe):
        dataframe[self.outputname] = 2*dataframe[self.column].rolling(window=self.period,center=False).std()
        return dataframe[[self.outputname]]

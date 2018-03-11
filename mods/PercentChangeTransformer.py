from sklearn.base import TransformerMixin

class PercentChangeTransformer(TransformerMixin):
    def __init__(self, column,outputname="Pct_Change",lookback=1):
        self.column = column
        self.cols = [column]
        self.outputname = outputname
        self.lookback = lookback

    def fit(self, x, y=None):
        return self

    def transform(self, dataframe):
        
        dataframe[self.outputname]= (dataframe[self.column].astype(float))/(dataframe[self.column].shift(self.lookback).astype(float))
        return dataframe[[self.outputname]]
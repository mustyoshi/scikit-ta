from sklearn.base import TransformerMixin

class MonthTransformer(TransformerMixin):
    def __init__(self):
        self.cols = ['Date']
        pass
    def fit(self, x, y=None):
        return self

    def transform(self, dataframe):
        
        dataframe["Date_Month"]= (dataframe['Date'].apply(lambda x: int(x[5:7])))
        return dataframe[["Date_Month"]]

class DayTransformer(TransformerMixin):
    def __init__(self):
        self.cols = ['Date']
        pass
    def fit(self, x, y=None):
        return self

    def transform(self, dataframe):
        
        dataframe["Date_Day"]= (dataframe['Date'].apply(lambda x: int(x[8:])))
        return dataframe[["Date_Day"]]

class HourTransformer(TransformerMixin):
    def __init__(self):
        self.cols = ['Time']
        pass
    def fit(self, x, y=None):
        return self

    def transform(self, dataframe):
        
        dataframe["Date_Hour"]= (dataframe['Time'].apply(lambda x: int(x[:2])))
        return dataframe[["Date_Hour"]]

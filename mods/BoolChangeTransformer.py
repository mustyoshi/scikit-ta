from sklearn.base import TransformerMixin

class BoolChangeTransformer(TransformerMixin):
    def __init__(self, column="",outputname="",period=1):
        self.column = column
        self.cols = [column]
        self.outputname = outputname
        self.period = period
        
    def fit(self, x, y=None):
        return self

    def transform(self, dataframe):
        dataframe[self.outputname] = dataframe[self.column] != dataframe[self.column].shift(-self.period)
        return dataframe[[self.outputname]]
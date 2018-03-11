from sklearn.base import TransformerMixin

class GreaterThanTransformer(TransformerMixin):
    def __init__(self, columns=[],outputname="",constant=None):
        self.columns = columns
        self.outputname = outputname
        self.constant = constant

    def fit(self, x, y=None):
        return self

    def transform(self, dataframe):
        if(self.constant != None):
            dataframe[self.outputname] = (dataframe[self.columns[0]].astype(float) > self.constant)
        else:
            dataframe[self.outputname] = (dataframe[self.columns[0]].astype(float) > dataframe[self.columns[1]].astype(float))
        #dataframe[self.outputname] = int(dataframe[self.outputname].astype(bool))
        return dataframe[[self.outputname]]
from sklearn.base import TransformerMixin

class BollingerBandTransform(TransformerMixin):
    def __init__(self, smacolumn="",stdcolumn="",outputname="BBand",deviations=1):
        self.smacolumn = smacolumn
        self.stdcolumn = stdcolumn
        self.cols = [smacolumn,stdcolumn]
        self.outputname = outputname
        self.deviations = deviations

    def fit(self, x, y=None):
        return self

    def transform(self, dataframe):
        
        dataframe[self.outputname+"_Bot"]= dataframe[self.smacolumn].astype(float)-(dataframe[self.stdcolumn].astype(float)*self.deviations)
        dataframe[self.outputname+"_Top"]= dataframe[self.smacolumn].astype(float)+(dataframe[self.stdcolumn].astype(float)*self.deviations)

        return dataframe[[self.outputname+"_Bot",self.outputname+"_Top"]]
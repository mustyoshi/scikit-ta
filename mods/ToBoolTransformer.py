from sklearn.base import TransformerMixin

class ToBoolTransformer(TransformerMixin):
    def __init__(self, columns=[]):
        self.columns = columns
        self.cols = columns

    def fit(self, x, y=None):
        return self

    def transform(self, dataframe):
        for c in self.columns:
            dataframe[c] = dataframe[c].astype(int)
        return dataframe[self.columns]
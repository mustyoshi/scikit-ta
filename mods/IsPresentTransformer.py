from sklearn.base import TransformerMixin

class IsPresentTransformer(TransformerMixin):
    def __init__(self, columns=[],prefix="IsPresent_"):
        self.columns = columns
        self.cols = columns
        self.prefix = prefix

    def fit(self, x, y=None):
        return self

    def transform(self, dataframe):
        for c in self.columns:
            dataframe[c] = dataframe[c].astype(int)
        return dataframe[self.columns]
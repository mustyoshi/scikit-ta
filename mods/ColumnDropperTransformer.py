from sklearn.base import TransformerMixin

class ColumnDropperTransformer(TransformerMixin):
    def __init__(self, columns=[]):
        self.columns = columns

    def fit(self, x, y=None):
        return self

    def transform(self, dataframe):
        #print('Dropping',self.columns,'from',dataframe.columns)
        return dataframe.drop(self.columns,errors="ignore",axis=1)
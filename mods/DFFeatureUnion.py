from sklearn.pipeline import FeatureUnion
from sklearn.externals.joblib import Parallel, delayed, Memory
import pandas as pd

def _transform_one(transformer, weight, X):
    res = transformer.transform(X)
    # if we have a weight for this transformer, multiply output
    if weight is None:
        return res
    return res * weight
class DFFeatureUnion(FeatureUnion):
    def __init__(self,**kw):
        FeatureUnion.__init__(self,**kw)

    def transform(self, X):
        print("Executing",len(self.transformer_list))
        """Transform X separately by each transformer, concatenate results.
        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.
        Returns
        -------
        X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, weight, X[trans.cols if hasattr(trans, 'cols') else X.columns])
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        return pd.concat(Xs,axis=1)


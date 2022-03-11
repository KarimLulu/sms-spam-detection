import numpy as np
import pandas as pd
import re
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import binarize


def unsquash(X):
    """ (n,) -> (n,1) """
    if len(X.shape) == 1 or X.shape[0] == 1:
        return np.asarray(X).reshape((len(X), 1))
    else:
        return X


def squash(X):
    """ (n,1) -> (n,) """
    return np.squeeze(np.asarray(X))


def is_lower(tokens):
    return any(token.islower() for token in tokens)


def is_upper(tokens):
    return any(token.isupper() for token in tokens)


class Transformer(TransformerMixin):
    """Base class for pure transformers"""

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        return X

    def get_params(self, deep=True):
        return dict()


class Squash(Transformer):
    def transform(self, X, **kwargs):
        return squash(X)


class Unsquash(Transformer):
    def transform(self, X, **kwargs):
        return unsquash(X)


class ModelTransformer(TransformerMixin):
    """ Use model predictions as transformer """
    def __init__(self, model, probs=True):
        self.model = model
        self.probs = probs

    def get_params(self, deep=True):
        return dict(model=self.model, probs=self.probs)

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X, **transform_params):
        if self.probs:
            pred = self.model.predict_proba(X)[:, 1]
        else:
            pred = self.model.predict(X)
        return unsquash(pred)


class Converter(Transformer):

    def transform(self, X, **kwargs):
        if isinstance(X, np.ndarray):
            return X
        elif isinstance(X, pd.Series):
            return X.values
        elif isinstance(X, str):
            return np.array([X])
        else:
            return X


class Length(Transformer):
    def __init__(self, use_tfidf=True):
        self.use_tfidf = use_tfidf

    def get_params(self, deep=True):
        return {"use_tfidf": self.use_tfidf}

    def transform(self, X, **kwargs):
        if self.use_tfidf:
            res = (X > 0).sum(axis=1)
        else:
            res = np.vectorize(len)(X)
        return unsquash(res)


class TfIdfLen(Transformer):
    def __init__(self, add_len=True, **tfidf_params):
        self.add_len = add_len
        self.tfidf_params = tfidf_params.copy()
        self.vectorizer = None

    def get_params(self, deep=True):
        output = self.tfidf_params
        output.update({"add_len": self.add_len})
        return output

    def set_params(self, **params):
        self.tfidf_params.update(**params)

    def fit(self, X, y=None):
        self.add_len = self.tfidf_params.pop("add_len", self.add_len)
        self.vectorizer = TfidfVectorizer(**self.tfidf_params)
        self.vectorizer.fit(X)
        return self

    def transform(self, X, **kwargs):
        res = self.vectorizer.transform(X)
        if self.add_len:
            lens = (res > 0).sum(axis=1)
            res = sparse.hstack([res, lens]).tocsr()
        return res


class MatchPattern(Transformer):

    def __init__(self, pattern, is_len, flags=re.U,
                 lib="re"):
        self.pattern = pattern
        self.is_len = is_len
        self.flags = flags
        self._lib = lib

    def get_params(self, deep=True):
        return dict(pattern=self.pattern, is_len=self.is_len, flags=self.flags)

    def transform(self, X, **kwargs):
        if self.is_len:
            func = lambda text: len(eval(self._lib).findall(self.pattern, text, self.flags))
        else:
            func = lambda text: bool(eval(self._lib).search(self.pattern, text, self.flags))
        rez = np.vectorize(func)(X).astype(int)
        return unsquash(rez)


class TokenFeatures(Transformer):

    def __init__(self, features=None):
        if features is None:
            features = []
        self.features = features

    def get_params(self, deep=True):
        return dict()

    def _get_features(self, tokens):
        output = []
        for f in self.features:
            output.append(eval(f)(tokens))
        return np.array(output)

    def transform(self, X, **kwargs):
        rez = []
        for record in X:
            temp = self._get_features(record)
            rez.append(temp)
        return np.array(rez)


class Select(Transformer):
    """  Extract specified columns from a pandas df or numpy array """

    def __init__(self, columns=0, to_np=True):
        self.columns = columns
        self.to_np = to_np

    def get_params(self, deep=True):
        return dict(columns=self.columns, to_np=self.to_np)

    def transform(self, X, **transform_params):
        if isinstance(X, pd.DataFrame):
            allint = isinstance(self.columns, int) or (isinstance(self.columns, list) and all([isinstance(x, int) for x in self.columns]))
            if allint:
                res = X.ix[:, self.columns]
            elif all([isinstance(x, str) for x in self.columns]):
                if len(self.columns) == 1:
                    res = X[self.columns[0]]
                else:
                    res = X[self.columns]
            else:
                print("Select error: mixed or wrong column type.")
                res = X

            if self.to_np:
                res = unsquash(res.values)
        else:
            res = unsquash(X[:, self.columns])

        return res


class EnsembleBinaryClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):

    def __init__(self, mode, weights=None):
        self.mode = mode
        self.weights = weights

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        """ Predict (weighted) probabilities """
        probs = np.average(X, axis=1, weights=self.weights)
        return np.column_stack((1-probs, probs))

    def predict(self, X):
        """ Predict class labels. """
        if self.mode == 'average':
            return binarize(self.predict_proba(X)[:, [1]], threshold=.5)
        else:
            res = binarize(X, threshold=.5)
            return np.apply_along_axis(lambda x: np.bincount(x.astype(int), self.weights).argmax(), axis=1, arr=res)

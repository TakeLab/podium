"""Module contains classes related to creating tfidf vectors from examples."""
import array
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer


class TfIdfVectorizer:
    def __init__(self, dataset, field, vocab=None, norm='l2', use_idf=True,
                 smooth_idf=True, sublinear_tf=False):
        self._vocab = field.vocab if vocab is None else vocab
        self._dataset = dataset
        self._field = field
        self._tfidf = TfidfTransformer(norm=norm, use_idf=use_idf,
                                       smooth_idf=smooth_idf,
                                       sublinear_tf=sublinear_tf)


    def _get_example_values(self, example):
        

    def _build_count_matrix(self, data):
        j_indices = []
        indptr = []
        values = array.array("i")
        indptr.append(0)

        for example in data:
            feature_counter = {}
            example_values = self._get_example_values(example)
            for feature in example_values:
                try:
                    feature_idx = self._vocab[feature]
                    if feature_idx not in feature_counter:
                        feature_counter[feature_idx] = 1
                    else:
                        feature_counter[feature_idx] += 1
                except KeyError:
                    # Ignore out-of-vocabulary items
                    continue

            j_indices.extend(feature_counter.keys())
            values.extend(feature_counter.values())
            indptr.append(len(j_indices))

        j_indices = np.asarray(j_indices, dtype=np.int64)
        indptr = np.asarray(indptr, dtype=np.int64)
        values = np.frombuffer(values, dtype=np.intc)

        X = sp.csr_matrix((values, j_indices, indptr),
                          shape=(len(indptr) - 1, len(self._vocab)),
                          dtype=np.int64)
        X.sort_indices()
        return X

    def fit(self):
        count_matrix = self._build_count_matrix(self._dataset)
        self._tfidf.fit(X)

    def transform(self, examples):
        count_matrix = self._build_count_matrix(examples)
        return self._tfidf.transform(X, copy=False)

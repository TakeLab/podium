"""Module contains classes related to creating tfidf vectors from examples."""
import array
from functools import partial
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer


class TfIdfVectorizer:
    """Class converts data from one field in examples to matrix of tf-idf features.
    It is equivalent to scikit-learn TfidfVectorizer available at
    https://scikit-learn.org.
    Class is dependant on TfidfTransformer defined in scikit-learn library.
    """
    def __init__(self, vocab, norm='l2', use_idf=True,
                 smooth_idf=True, sublinear_tf=False):
        """Constructor that initializes tfidf vectorizer. Parameters besides vocab
        are passed to TfidfTransformer, for further details on these parameters see
        scikit-learn documentation.

        Parameters
        ----------
        vocab : Vocab
            vocabulary instance that can be given as field.vocab or as vocab
            from other source
        norm : 'l1', 'l2' or None, optional (default='l2')
            Each output row will have unit norm, either:
            * 'l2': Sum of squares of vector elements is 1. The cosine
            similarity between two vectors is their dot product when l2 norm has
            been applied.
            * 'l1': Sum of absolute values of vector elements is 1.
        use_idf : boolean (default=True)
            Enable inverse-document-frequency reweighting.
        smooth_idf : boolean (default=True)
            Smooth idf weights by adding one to document frequencies, as if an
            extra document was seen containing every term in the collection
            exactly once. Prevents zero divisions.
        sublinear_tf : boolean (default=False)
            Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
        """
        self._vocab = vocab
        self._tfidf = TfidfTransformer(norm=norm, use_idf=use_idf,
                                       smooth_idf=smooth_idf,
                                       sublinear_tf=sublinear_tf)

    def _get_tensor_values(self, data):
        """Function obtains data for example in numericalized matrix. This function
        is used when transforming data with tfidf and in general pipeline it should
        be in moment when vectorization of numericalized batch happens

        Parameters
        ----------
        data : array like
            array containing numericalized tokens for one example (document)

        Returns
        -------
        data : array like
            numericalized tokens array
        """
        return data

    def _get_example_values(self, example, field):
        """Function obtains data for given field in example. This function is used
        when fitting tfidf vectorizer with dataset. Fields that are not numericalized
        but are eager will be numericalized.

        Parameters
        ----------
        example : Example
            example instance from dataset
        field : Field
            field instance, required to access data in example and to numericalize
            data if necessary

        Returns
        -------
        values : array like
            numericalized tokens array
        """
        values = None
        cached_numericalization = getattr(example, f"{field.name}_")
        if cached_numericalization is not None:
            values = cached_numericalization
        else:
            values = field.numericalize(getattr(example, field.name))
        return values

    def _build_count_matrix(self, data, unpack_data):
        """Function builds sparse count feature matrix. It is equivalent with using
        CountVectorizer in scikit-learn.

        Parameters
        ----------
        data : Dataset or array-like
            data source used for creating feature matrix
        unpack_data : callable
            callable that can transform one instance from data to numericalized
            tokens array

        """
        j_indices = []
        indptr = []
        values = array.array("i")
        indptr.append(0)

        for example in data:
            feature_counter = {}
            example_values = unpack_data(example)
            for feature_idx in example_values:
                try:
                    if feature_idx not in feature_counter:
                        feature_counter[feature_idx] = 1
                    else:
                        feature_counter[feature_idx] += 1
                except ValueError:
                    # Ignore out-of-vocabulary items
                    continue

            j_indices.extend(feature_counter.keys())
            values.extend(feature_counter.values())
            indptr.append(len(j_indices))

        j_indices = np.asarray(j_indices, dtype=np.int64)
        indptr = np.asarray(indptr, dtype=np.int64)
        values = np.frombuffer(values, dtype=np.intc)

        count_matrix = sp.csr_matrix((values, j_indices, indptr),
                                     shape=(len(indptr) - 1, len(self._vocab)),
                                     dtype=np.int64)
        count_matrix.sort_indices()
        return count_matrix

    def fit(self, dataset, field):
        """Learn idf from dataset on data in given field.

        Parameters
        ----------
        dataset : Dataset
            dataset instance cointaining data on which to build idf matrix
        field : Field
            which field in dataset to use for tfidf

        Returns
        -------
        self : TfIdfVectorizer

        """
        count_matrix = self._build_count_matrix(
            data=dataset, unpack_data=partial(self._get_example_values,
                                              field=field))
        self._tfidf.fit(count_matrix)
        return self

    def transform(self, examples):
        """Transforms examples to example-term matrix. Uses vocabulary that is given
        in constructor.

        Parameters
        ----------
        example : iterable
            an iterable which yields array with numericalized tokens

        Returns
        -------
        X : sparse matrix, [n_samples, n_features]
            Tf-idf weighted document-term matrix
        """
        count_matrix = self._build_count_matrix(data=examples,
                                                unpack_data=self._get_tensor_values)
        return self._tfidf.transform(count_matrix, copy=False)

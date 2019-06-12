"""Module contains classes related to creating tfidf vectors from examples."""
import array
from collections import Counter
import logging
from functools import partial
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from takepod.storage.vocab import SpecialVocabSymbols

_LOGGER = logging.getLogger(__name__)


class TfIdfVectorizer:
    """Class converts data from one field in examples to matrix of tf-idf features.
    It is equivalent to scikit-learn TfidfVectorizer available at
    https://scikit-learn.org.
    Class is dependant on TfidfTransformer defined in scikit-learn library.
    """
    def __init__(self, vocab=None, norm='l2', use_idf=True,
                 smooth_idf=True, sublinear_tf=False):
        """Constructor that initializes tfidf vectorizer. Parameters besides vocab
        are passed to TfidfTransformer, for further details on these parameters see
        scikit-learn documentation.

        Parameters
        ----------
        vocab : Vocab, optional
            vocabulary instance that can be given as field.vocab or as vocab
            from other source. If None, it will be initialized during fit from field.
        norm
            see scikit tfidf transformer documentation
        use_idf
            see scikit tfidf transformer documentation
        smooth_idf
            see scikit tfidf transformer documentation
        sublinear_tf
            see scikit tfidf transformer documentation
        """
        self._vocab = vocab
        self._tfidf = TfidfTransformer(norm=norm, use_idf=use_idf,
                                       smooth_idf=smooth_idf,
                                       sublinear_tf=sublinear_tf)
        self._fitted = False

    def _check_vocab(self):
        """Method checks if the vocab is valid before fitting. Vocab mustn't be None.
        Also user is warned if the vocab contains unknown special symbol because
        unknown symbol is also part of tfidf matrix (different from scikit).
        """
        if self._vocab is None:
            error_msg = "TfIdf can't fit without vocab, given vocab is None."
            _LOGGER.error(error_msg)
            raise ValueError(error_msg)
        if SpecialVocabSymbols.UNK in self._vocab.stoi:
            warning_msg = "Vocab contains unknown special symbol. Tf-idf for all "\
                          "non-vocabulary words will be counted towards the unknown"\
                          " symbol"
            _LOGGER.warning(warning_msg)

    def _get_tensor_values(self, data):
        """Method obtains data for example in numericalized matrix. This method
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
        """Method obtains data for given field in example. This method is used
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
        values = field.get_numericalization_for_example(example)
        return values

    def _build_count_matrix(self, data, unpack_data):
        """Method builds sparse count feature matrix. It is equivalent with using
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
            feature_counter = Counter()
            example_values = unpack_data(example)
            for feature_idx in example_values:
                feature_counter[feature_idx] += 1

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
        if dataset is None or field is None:
            error_msg = f"dataset or field mustn't be None, given dataset: "\
                        f"{str(dataset)}, field: {str(field)}"
            _LOGGER.error(error_msg)
            raise ValueError(error_msg)
        if field.name not in dataset.field_dict:
            error_msg = f"invalid field, given field: {str(field)}"
            _LOGGER.error(error_msg)
            raise ValueError(error_msg)
        if self._vocab is None:
            self._vocab = field.vocab
        self._check_vocab()
        count_matrix = self._build_count_matrix(
            data=dataset, unpack_data=partial(self._get_example_values,
                                              field=field))
        self._tfidf.fit(count_matrix)
        self._fitted = True
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
        if not self._fitted:
            error_msg = "Vectorizer must be fitted before transforming."
            _LOGGER.error(error_msg)
            raise RuntimeError(error_msg)
        if examples is None:
            error_msg = "examples mustn't be None"
            _LOGGER.error(error_msg)
            raise ValueError(error_msg)
        count_matrix = self._build_count_matrix(data=examples,
                                                unpack_data=self._get_tensor_values)
        return self._tfidf.transform(count_matrix, copy=False)
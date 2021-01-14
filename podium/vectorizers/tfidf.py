"""
Module contains classes related to creating tfidf vectors from examples.
"""
import array
from collections import Counter
from functools import partial

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer


class CountVectorizer:
    """
    Class converts data from one field in examples to matrix of bag of words
    features.

    It is equivalent to scikit-learn CountVectorizer available at
    https://scikit-learn.org.
    """

    def __init__(self, vocab=None, specials=None):
        """
        Method initializes count vectorizer.

        Parameters
        ----------
        vocab : Vocab, optional
            vocabulary instance that can be given as field.vocab or as vocab
            from other source. If None, it will be initialized during fit from field.
        specials : list(str), optional
            list of tokens for which tfidf is not calculated,
            if None vocab specials are used
        """
        self._vocab = vocab
        self._specials = specials
        self._special_indexes = None
        self._fitted = False

    def _init_special_indexes(self):
        """
        Initializes set of special symbol indexes in vocabulary.

        Used to skip special symbols while calculating count matrix.
        """
        special_symbols = self._vocab.specials if not self._specials else self._specials
        self._special_indexes = set([self._vocab.stoi[s] for s in special_symbols])

    def _build_count_matrix(self, data, unpack_data):
        """
        Method builds sparse count feature matrix. It is equivalent with using
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

        count_matrix = sp.csr_matrix(
            (values, j_indices, indptr),
            shape=(len(indptr) - 1, len(self._vocab)),
            dtype=np.int64,
        )
        count_matrix.sort_indices()
        if self._special_indexes:
            keep_columns = list(set(range(count_matrix.shape[1])) - self._special_indexes)
            count_matrix = count_matrix[:, keep_columns]
        return count_matrix

    def _get_tensor_values(self, data):
        """
        Method obtains data for example in numericalized matrix. This method is
        used when transforming data with vectorizer and in general pipeline it
        should be in moment when vectorization of numericalized batch happens.

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
        """
        Method obtains data for given field in example. This method is used when
        fitting vectorizer with dataset. Fields that are not numericalized but
        are eager will be numericalized.

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

    def _check_fitted(self):
        """
        Method checks if the current vectorizer is fitted.

        Raises
        ------
        RuntimeError
            If the vectorizer is not fitted before transforming.
        """
        if not self._fitted:
            raise RuntimeError("Vectorizer has not been fitted.")

    def fit(self, dataset, field):
        """
        Method initializes count vectorizer.

        Parameters
        ----------
        dataset : Dataset, optional
            dataset instance which contains field
        field : Field, optional
            which field in dataset to use for vocab, if None vocab given in constructor is
            used

        Returns
        -------
        self : CountVectorizer

        Raises
        ------
        ValueError
            If the vocab or fields vocab are None
        """
        if self._vocab is None and (field is None or field.vocab is None):
            raise ValueError(
                "Vocab is not defined. User should define vocab in constructor "
                "or by providing field with a non-empty vocab property."
            )

        if field and field._allow_missing_data:
            raise ValueError(
                "CountVectorizer doesn't support fields that "
                f"contain missing data: {dataset}, field: {field}"
            )

        self._vocab = field.vocab if self._vocab is None else self._vocab
        self._init_special_indexes()
        self._fitted = True

    def transform(self, examples, **kwargs):
        """
        Method transforms given examples to count matrix where rows are examples
        and columns represent token counts.

        Parameters
        ----------
        examples : iterable
            an iterable which yields array with numericalized tokens or list of examples
        tokens_tensor : bool, optional
            if True method expects for examples to be a tensor of numericalized values,
            otherwise it expects to receive list of examples(which can be in fact dataset)
            and a field for numericalization
        field : Field, optional
            if tokens_tensor is False, method expects reference to field that is used for
            numericalization

        Raises
        ------
        ValueError
            If user has given invalid arguments - if examples are None or the field is not
            provided and given examples are not in token tensor format.
        """
        self._check_fitted()
        is_tokens_tensor = (
            kwargs["is_tokens_tensor"] if "is_tokens_tensor" in kwargs else True
        )
        field = kwargs["field"] if "field" in kwargs else None

        if examples is None:
            raise ValueError("Examples mustn't be None.")
        if not is_tokens_tensor and field is not None:
            return self._build_count_matrix(
                data=examples, unpack_data=partial(self._get_example_values, field=field)
            )
        elif is_tokens_tensor:
            return self._build_count_matrix(
                data=examples, unpack_data=self._get_tensor_values
            )
        raise ValueError(
            "Invalid method arguments. Method expects tensors of numericalized "
            "tokens as examples or dataset as collection of examples from which "
            " with given field to extract data."
        )


class TfIdfVectorizer(CountVectorizer):
    """
    Class converts data from one field in examples to matrix of tf-idf features.

    It is equivalent to scikit-learn TfidfVectorizer available at
    https://scikit-learn.org. Class is dependant on TfidfTransformer defined in
    scikit-learn library.
    """

    def __init__(
        self,
        vocab=None,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
        specials=None,
    ):
        """
        Constructor that initializes tfidf vectorizer. Parameters besides vocab
        are passed to TfidfTransformer, for further details on these parameters
        see scikit-learn documentation.

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
        specials : list(str), optional
            list of tokens for which tfidf is not calculated,
            if None vocab specials are used
        """
        super(TfIdfVectorizer, self).__init__(**{"vocab": vocab, "specials": specials})
        self._tfidf = TfidfTransformer(
            norm=norm, use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf
        )
        self._fitted = False

    def fit(self, dataset, field):
        """
        Learn idf from dataset on data in given field.

        Parameters
        ----------
        dataset : Dataset
            dataset instance cointaining data on which to build idf matrix
        field : Field
            which field in dataset to use for tfidf

        Returns
        -------
        self : TfIdfVectorizer

        Raises
        ------
        ValueError
            If dataset or field are None and if name of given field is not in dataset.
        """
        super(TfIdfVectorizer, self).fit(dataset=dataset, field=field)
        if dataset is None or field is None:
            raise ValueError(
                "dataset or field mustn't be None, given dataset: "
                f"{dataset}, field: {field}"
            )
        if field.name not in dataset.field_dict:
            raise ValueError(f"invalid field, given field: {field}")
        count_matrix = super(TfIdfVectorizer, self).transform(
            **{"examples": dataset, "is_tokens_tensor": False, "field": field}
        )
        self._tfidf.fit(count_matrix)
        self._fitted = True

    def transform(self, examples, **kwargs):
        """
        Transforms examples to example-term matrix. Uses vocabulary that is
        given in constructor.

        Parameters
        ----------
        example : iterable
            an iterable which yields array with numericalized tokens

        Returns
        -------
        X : sparse matrix, [n_samples, n_features]
            Tf-idf weighted document-term matrix

        Raises
        ------
        ValueError
            If examples are None.
        RuntimeError
            If vectorizer is not fitted yet.
        """
        self._check_fitted()
        if examples is None:
            raise ValueError("examples mustn't be None")
        count_matrix = super(TfIdfVectorizer, self).transform(
            **{"examples": examples, "is_tokens_tensor": True, "field": None}
        )
        return self._tfidf.transform(count_matrix, copy=False)

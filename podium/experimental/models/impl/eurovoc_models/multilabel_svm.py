"""
Multilabel SVM classifier for EuroVoc dataset.

Example
-------

    model_path = ... Path where the trained model will be stored

    dataset_path = ... Path where the dilled instance of EuroVoc dataset \
        will be stored to and/or loaded from

    LargeResource.BASE_RESOURCE_DIR = ... Directory where the EuroVoc downloaded raw \
                                          dataset is stored or where it should be \
                                          downloaded

    # this creates and dills the dataset and it should be done only once
    # the created instance of the dataset can be reused for training the model
    dill_dataset(dataset_path)

    p_grid = {"C": [1, 10, 100]}
    train_multilabel_svm(dataset_path=dataset_path, n_jobs=20, param_grid=p_grid, \
                         cut_off=2)

    with open(model_path, "wb") as output_file:
        dill.dump(obj=clf, file=output_file)
"""
import warnings

import dill
import numpy as np
from sklearn import model_selection, svm
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

from podium.datasets.iterator import Iterator
from podium.experimental.models import AbstractSupervisedModel
from podium.experimental.validation.validation import KFold
from podium.vectorizers.tfidf import TfIdfVectorizer


class MultilabelSVM(AbstractSupervisedModel):
    """
    Multilabel SVM with hyperparameter optimization via grid search using K-fold
    cross-validation.

    Multilabel SVM is implemented as a set of binary SVM classifiers, one for
    each class in dataset (one vs. rest).
    """

    def __init__(self):
        """
        Creates and instance of MultilabelSVM.
        """
        self._models = None

    def fit(
        self,
        X,
        y,
        parameter_grid,
        n_splits=3,
        max_iter=10000,
        cutoff=1,
        scoring="f1",
        n_jobs=1,
    ):
        """
        Fits the model on given data.

        For each class present in y (for each column of the y matrix), a separate SVM
        model is trained. If there are no positive training instances for some label
        (the entire column is filled with zeros), no model is trained. Upon calling the
        predict function, a zero vector is returned for that class. The indexes of the
        columns containing such labels are stored and can be retrieved using the
        get_indexes_of_missing_models method.

        Parameters
        ----------
        X : np.array
            input data
        y : np.array
            data labels, 2D array (number of examples, number of labels)
        parameter_grid : dict or list(dict)
            Dictionary with parameters names (string) as keys and lists of parameter
            settings to try as values, or a list of such dictionaries, in which case the
            grids spanned by each dictionary in the list are explored. This enables
            searching over any sequence of parameter settings. For more information,
            refer to
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
            The parameter_grid may contain any of the parameters used to train an
            instance of the LinearSVC model, most notably penalty parameter 'C' and
            regularization penalty 'penalty' that can be set to 'l1' or 'l2'.
            For more information, please refer to:
            https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
        n_splits : int
            Number of splits for K-fold cross-validation
        max_iter : int
            Maximum number of iterations for training a single SVM within the model.
        cutoff : int >= 1
            If the number of positive training examples for a class is less than the
            cut-off, no model is trained for such class and the index of the label is
            added in the missing model indexes.
        scoring : string, callable, list/tuple, dict or None
            Indicates what scoring function to use in order to determine the best
            hyperparameters via grid search. For more details, view
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
        n_jobs : int
            Number of threads to be used.

        Raises
        ------
        ValueError
            If cutoff is not a positive integer >= 1.
            If n_jobs is not a positive integer or -1.
            If n_jobs is not a positive integer >= 1.
            If max_iter is not a positive integer >= 1.
        """
        if cutoff < 1:
            raise ValueError(
                "cutoff must be a positive integer >= 1, but " f"{cutoff} was given"
            )
        if n_jobs < -1 or n_jobs == 0:
            raise ValueError(
                "n_jobs must be a postivive integer or -1, but " f"{n_jobs} was given"
            )
        if n_splits < 1:
            raise ValueError(
                "n_splits must be a positive integer >= 1, but " f"{n_splits} was given"
            )
        if max_iter < 1:
            raise ValueError(
                "max_iter must be a positive integer >= 1, but " f"{max_iter} was given"
            )

        y = np.ndarray.transpose(y)  # Returns a transposed view of the y matrix

        # storing indexes of missing models (models are not trained when the number of
        # positive training examples is less that the cut off given)
        self._missing_indexes = set()

        self._models = []
        for i, y_i in enumerate(y):
            num_examples = np.count_nonzero(y_i)
            if num_examples < cutoff:
                # Skipping label due to cut-off.
                self._models.append(None)
                self._missing_indexes.add(i)

                warnings.warn(
                    f"Label at index {i} doesn't have enough instances in the "
                    "train set, a model won't be trained for this label. "
                    f"Number of instances: {num_examples}, cutoff {cutoff}",
                    RuntimeWarning,
                )
                continue

            # using KFold from sklearn as model_selection.KFold
            inner_cv = model_selection.KFold(
                n_splits=n_splits, shuffle=True, random_state=0
            )
            classifier = svm.LinearSVC(max_iter=max_iter)
            clf = GridSearchCV(
                estimator=classifier,
                param_grid=parameter_grid,
                cv=inner_cv,
                scoring=scoring,
                error_score=0.0,
                n_jobs=n_jobs,
            )

            clf.fit(X, y_i)
            self._models.append(clf)

    def predict(self, X):
        """
        Predict labels for given data.

        If no model has been trained for some class (because the was not enough examples
        for this label in the train set), a zero column is returned. If one wishes to
        exclude such labels from the evaluation, their indexes can be retrieved through
        the get_indexes_of_missing_models method.

        Parameters
        ----------
        X : np.array
            input data

        Returns
        -------
        result : 2D np.array (number of examples, number of classes)
            Predictions of the model for the given examples.

        Raises
        ------
        RuntimeError
            If the model instance is not fitted.
        """
        if not self._models:
            raise RuntimeError("Trying to predict using an unfitted model instance.")

        Y = np.empty(shape=(len(self._models), X.shape[0]), dtype=np.int32)
        for i, model in enumerate(self._models):
            if model is None:
                Y[i] = [0] * X.shape[0]
                warnings.warn(
                    f"No model trained for label at index {i}, returning a zero "
                    "vector instead of model prediction.",
                    RuntimeWarning,
                )
            else:
                Y[i] = model.predict(X)
        return {AbstractSupervisedModel.PREDICTION_KEY: Y.transpose()}

    def get_indexes_of_missing_models(self):
        """
        Returns the indexes of classes for which the models have not been
        trained due to the lack of positive training examples.

        Returns
        -------
        result : list(int)
            Indexes of models that were not trained.

        Raises
        ------
        RuntimeError
            If the model instance is not fitted.
        """
        if self._models is None:
            raise RuntimeError(
                "Trying to get missing model indices on an unfitted model instance."
            )
        return self._missing_indexes

    def reset(self, **kwargs):
        raise NotImplementedError("Reset not yet implemented for this model.")


def get_label_matrix(Y):
    """
    Takes the target fields returned by the EuroVoc iterator and returns the
    EuroVoc label matrix.

    Parameters
    ----------
    Y : dict
        Target returned by the EuroVoc dataset iterator.

    Returns
    -------
    np.array : matrix of labels for each example (number of examples, number of classes)
    """
    return np.array(Y.eurovoc_labels, dtype=np.float32)


def train_multilabel_svm(
    dataset_path,
    param_grid,
    cutoff,
    n_outer_splits=5,
    n_inner_splits=3,
    n_jobs=1,
    is_verbose=True,
    include_classes_with_no_train_examples=False,
    include_classes_with_no_test_examples=False,
):
    """
    Trains the multilabel SVM model on a given instance of dataset.

    Parameters
    ----------
    dataset_path : str
        Path to the instance of EuroVoc dataset stored as a dill file.
    param_grid : dict or list(dict)
            Dictionary with parameters names (string) as keys and lists of parameter
            settings to try as values, or a list of such dictionaries, in which case the
            grids spanned by each dictionary in the list are explored. This enables
            searching over any sequence of parameter settings. For more information,
            refer to
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    cutoff : int
        If the number of positive training examples for a class is less than the
        cut-off, no model is trained for such class and the index of the label is
        added in the missing model indexes.
    n_outer_splits : int
        Number of splits in an outer loop of a nested cross validation.
    n_inner_splits : int
        Number of splits in an inner loop of a nested cross validation.
    n_jobs : int
        Number of threads to be used.
    is_verbose : boolean
        If set to true, scores on test set are printed for each fold of the
        outer loop in the nested cross validation.
    include_classes_with_no_train_examples : boolean
        If True, scores of the classes witn an unsufficient number of training examples
        (less than the specified cut-off) are included when calculating general scores.
        Note that this makes sense if cut-off=1 because that means classes with no train
        examples will be taken into consideration.
    include_classes_with_no_test_examples : boolean
        If True, scores for classes with no positive instances in the test set are
        included in the general score.
    """
    dataset = None
    with open(dataset_path, "rb") as input_file:
        dataset = dill.load(input_file)

    vectorizer = TfIdfVectorizer()
    vectorizer.fit(dataset, dataset.field_dict["text"])

    outer_cv = KFold(n_splits=n_outer_splits, shuffle=True, random_state=0)

    micro_P = []
    micro_R = []
    micro_F1 = []
    macro_P = []
    macro_R = []
    macro_F1 = []

    for train, test in outer_cv.split(dataset):
        train_iter = Iterator(dataset=train, batch_size=len(train))
        clf = MultilabelSVM()
        for X, Y in train_iter:
            X = vectorizer.transform(X.text)
            Y = get_label_matrix(Y)

            clf.fit(X, Y, parameter_grid=param_grid, cutoff=cutoff, n_jobs=n_jobs)

        test_iter = Iterator(dataset=test, batch_size=len(test))
        for X, Y in test_iter:
            X = vectorizer.transform(X.text)
            Y = get_label_matrix(Y)
            prediction_dict = clf.predict(X)
            Y_pred = prediction_dict[AbstractSupervisedModel.PREDICTION_KEY]

            if not include_classes_with_no_train_examples:
                Y_pred = np.delete(
                    Y_pred, list(clf.get_indexes_of_missing_models()), axis=1
                )
                Y = np.delete(Y, list(clf.get_indexes_of_missing_models()), axis=1)

            # deletes all zero columns (all labels which don't have any positive exaples
            # in the current test set)
            if not include_classes_with_no_test_examples:
                cols = ~(Y == 0).all(axis=0)
                Y = Y[:, cols]
                Y_pred = Y_pred[:, cols]

            micro_P.append(precision_score(Y, Y_pred, average="micro"))
            micro_R.append(recall_score(Y, Y_pred, average="micro"))
            micro_F1.append(f1_score(Y, Y_pred, average="micro"))

            macro_P.append(precision_score(Y, Y_pred, average="macro"))
            macro_R.append(recall_score(Y, Y_pred, average="macro"))
            macro_F1.append(f1_score(Y, Y_pred, average="macro"))

            if is_verbose:
                print("Scores on test set:")
                print("micro P", micro_P[-1])
                print("micro R", micro_R[-1])
                print("micro F1", micro_F1[-1])
                print("macro P", macro_P[-1])
                print("macro R", macro_R[-1])
                print("macro F1", macro_F1[-1])

    print("Average scores on test sets:")

    print("average micro P", np.average(micro_P))
    print("average micro R", np.average(micro_R))
    print("average micro F1", np.average(micro_F1))

    print("average macro P", np.average(macro_P))
    print("average macro R", np.average(macro_R))
    print("average macro F1", np.average(macro_F1))

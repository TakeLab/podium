import pytest
import numpy as np
from collections import namedtuple

from numpy.testing import assert_array_equal

from takepod.models.impl.eurovoc_models import multilabel_svm as ms

import warnings
from sklearn.exceptions import FitFailedWarning, ConvergenceWarning
from sklearn.exceptions import UndefinedMetricWarning
# Happens when model predicts all zeros and the F1 score can't be calcualted.
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
# Due to small sample dataset, it often happens that some folds of GridSearchCV end up
# with no positive or no negative training examples for some label. In this case,
# FitFailedWarinig occurs.
warnings.filterwarnings("ignore", category=FitFailedWarning)
# The maximum number of iterations in test is set to 1. The model fails to converge in
# 1 iteration and issues a warning.
warnings.filterwarnings("ignore", category=ConvergenceWarning)


X = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
Y = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]])

Y_missing = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]])

Target = namedtuple('Target', 'eurovoc_labels')
target = Target(eurovoc_labels=Y)


def test_get_label_matrix():
    Y = ms.get_label_matrix(target)
    np.testing.assert_array_equal(Y, target.eurovoc_labels)


def test_fitting_multilable_svm():
    clf = ms.MultilabelSVM()
    parameter_grid = {"C": [1]}
    n_splits = 2
    max_iter = 1
    cutoff = 1
    scoring = 'f1'
    n_jobs = 1

    clf.fit(X=X, y=Y, parameter_grid=parameter_grid, n_splits=n_splits,
            max_iter=max_iter, cutoff=cutoff, scoring=scoring, n_jobs=n_jobs)
    prediction_dict = clf.predict(X)
    Y_pred = prediction_dict[ms.MultilabelSVM.PREDICTION_KEY]
    assert Y_pred.shape == Y.shape


def test_invalid_cutoff():
    clf = ms.MultilabelSVM()
    parameter_grid = {"C": [1]}
    n_splits = 2
    max_iter = 1
    cutoff = 0
    scoring = 'f1'
    n_jobs = 1

    with pytest.raises(ValueError):
        clf.fit(X=X, y=Y, parameter_grid=parameter_grid, n_splits=n_splits,
                max_iter=max_iter, cutoff=cutoff, scoring=scoring, n_jobs=n_jobs)


def test_invalid_n_jobs():
    clf = ms.MultilabelSVM()
    parameter_grid = {"C": [1]}
    n_splits = 2
    max_iter = 1
    cutoff = 1
    scoring = 'f1'
    n_jobs = -2

    with pytest.raises(ValueError):
        clf.fit(X=X, y=Y, parameter_grid=parameter_grid, n_splits=n_splits,
                max_iter=max_iter, cutoff=cutoff, scoring=scoring, n_jobs=n_jobs)


def test_invalid_n_splits():
    clf = ms.MultilabelSVM()
    parameter_grid = {"C": [1]}
    n_splits = 0
    max_iter = 1
    cutoff = 1
    scoring = 'f1'
    n_jobs = 1

    with pytest.raises(ValueError):
        clf.fit(X=X, y=Y, parameter_grid=parameter_grid, n_splits=n_splits,
                max_iter=max_iter, cutoff=cutoff, scoring=scoring, n_jobs=n_jobs)


def test_invalid_max_iter():
    clf = ms.MultilabelSVM()
    parameter_grid = {"C": [1]}
    n_splits = 2
    max_iter = 0
    cutoff = 1
    scoring = 'f1'
    n_jobs = 1

    with pytest.raises(ValueError):
        clf.fit(X=X, y=Y, parameter_grid=parameter_grid, n_splits=n_splits,
                max_iter=max_iter, cutoff=cutoff, scoring=scoring, n_jobs=n_jobs)


def test_missing_indexes():
    clf = ms.MultilabelSVM()
    parameter_grid = {"C": [1]}
    n_splits = 2
    max_iter = 1
    cutoff = 1
    scoring = 'f1'
    n_jobs = 1

    clf.fit(X=X, y=Y_missing, parameter_grid=parameter_grid, n_splits=n_splits,
            max_iter=max_iter, cutoff=cutoff, scoring=scoring, n_jobs=n_jobs)
    missing_indexes = clf.get_indexes_of_missing_models()
    assert len(missing_indexes) == 1
    assert missing_indexes == set([2])


def test_prediction_with_missing_indexes():
    clf = ms.MultilabelSVM()
    parameter_grid = {"C": [1]}
    n_splits = 2
    max_iter = 1
    cutoff = 1
    scoring = 'f1'
    n_jobs = 1

    clf.fit(X=X, y=Y_missing, parameter_grid=parameter_grid, n_splits=n_splits,
            max_iter=max_iter, cutoff=cutoff, scoring=scoring, n_jobs=n_jobs)
    missing_indexes = clf.get_indexes_of_missing_models()
    prediction_dict = clf.predict(X)
    Y_pred = prediction_dict[ms.MultilabelSVM.PREDICTION_KEY]

    assert Y_pred.shape == Y_missing.shape
    assert len(missing_indexes) == 1
    assert missing_indexes == set([2])
    assert(len(Y_pred[:, 2]) == 4)
    assert_array_equal(Y_pred[:, 2], np.zeros(4))


def test_prediction_on_unfitted_model():
    clf = ms.MultilabelSVM()
    with pytest.raises(RuntimeError):
        clf.predict(X)


def test_getting_missing_indexes_on_unfitted_model():
    clf = ms.MultilabelSVM()
    with pytest.raises(RuntimeError):
        clf.get_indexes_of_missing_models()

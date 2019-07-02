import pytest
import numpy as np
from collections import namedtuple

from takepod.models.eurovoc_models import multilabel_svm as ms

import warnings
from sklearn.exceptions import FitFailedWarning, ConvergenceWarning
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)
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
    cut_off = 1
    scoring = 'f1'
    n_jobs = 1

    clf.fit(X=X, y=Y, parameter_grid=parameter_grid, n_splits=n_splits,
            max_iter=max_iter, cut_off=cut_off, scoring=scoring, n_jobs=n_jobs)
    Y_pred = clf.predict(X)
    assert Y_pred.shape == Y.shape


def test_missing_indexes():
    clf = ms.MultilabelSVM()
    parameter_grid = {"C": [1]}
    n_splits = 2
    max_iter = 1
    cut_off = 1
    scoring = 'f1'
    n_jobs = 1

    clf.fit(X=X, y=Y_missing, parameter_grid=parameter_grid, n_splits=n_splits,
            max_iter=max_iter, cut_off=cut_off, scoring=scoring, n_jobs=n_jobs)
    missing_indexes = clf.get_indexes_of_missing_models()
    assert missing_indexes == [2]


def test_prediction_with_missing_indexes():
    clf = ms.MultilabelSVM()
    parameter_grid = {"C": [1]}
    n_splits = 2
    max_iter = 1
    cut_off = 1
    scoring = 'f1'
    n_jobs = 1

    clf.fit(X=X, y=Y_missing, parameter_grid=parameter_grid, n_splits=n_splits,
            max_iter=max_iter, cut_off=cut_off, scoring=scoring, n_jobs=n_jobs)
    missing_indexes = clf.get_indexes_of_missing_models()
    Y_pred = clf.predict(X)

    assert Y_pred.shape == Y_missing.shape
    assert missing_indexes == [2]
    assert np.count_nonzero(Y_pred[:, 2]) == 0


def test_prediction_on_unfitted_model():
    clf = ms.MultilabelSVM()
    with pytest.raises(RuntimeError):
        clf.predict(X)


def test_getting_missing_indexes_on_unfitted_model():
    clf = ms.MultilabelSVM()
    with pytest.raises(RuntimeError):
        clf.get_indexes_of_missing_models()

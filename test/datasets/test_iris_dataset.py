import pytest


def test_iris_dataset():
    sklearn_datasets = pytest.importorskip("sklearn.datasets")
    from podium.datasets.iris_dataset import IrisDataset

    iris_ds = IrisDataset()

    x, y = sklearn_datasets.load_iris(True)

    assert len(iris_ds) == len(x)
    for i in range(0, len(x), 30):
        sepal_len, sepal_width, petal_len, petal_width = x[i]
        species = y[i]

        ex = iris_ds[i]

        assert ex.sepal_length[0] == sepal_len
        assert ex.sepal_width[0] == sepal_width
        assert ex.petal_length[0] == petal_len
        assert ex.petal_width[0] == petal_width
        assert ex.species[0] == species

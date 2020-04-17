from sklearn.datasets import load_iris

from podium.datasets import Dataset
from podium.storage import ExampleFactory, Field


class IrisDataset(Dataset):
    """This is the classic Iris dataset. This is perhaps the best known database to be
    found in the pattern recognition literature.

    The fields of this dataset are:
        sepal_length - float
        sepal_width - float
        petal_length - float
        petal_width - float
        species - int, specifying iris species
    """

    def __init__(self):
        """Loads the Iris dataset.
        """
        x, y = load_iris(True)

        fields = IrisDataset._get_default_fields()
        example_factory = ExampleFactory(fields)

        data = ((*x_, y_) for x_, y_ in zip(x, y))

        examples = list(map(example_factory.from_list, data))
        super().__init__(examples, fields)

    @staticmethod
    def _get_default_fields():
        def identity(x):
            return x

        sepal_len_field = Field("sepal_length", tokenize=False,
                                custom_numericalize=identity,
                                store_as_raw=True)
        sepal_width_field = Field("sepal_width", tokenize=False,
                                  custom_numericalize=identity,
                                  store_as_raw=True)
        petal_len_field = Field("petal_length", tokenize=False,
                                custom_numericalize=identity,
                                store_as_raw=True)
        petal_width_field = Field("petal_width", tokenize=False,
                                  custom_numericalize=identity,
                                  store_as_raw=True)

        species_field = Field("species", tokenize=False, store_as_raw=True,
                              custom_numericalize=identity, is_target=True)

        return sepal_len_field, sepal_width_field, \
            petal_len_field, petal_width_field, \
            species_field

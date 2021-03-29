from sklearn.datasets import load_iris

from podium.datasets import Dataset
from podium.datasets.example_factory import ExampleFactory
from podium.field import Field


class Iris(Dataset):
    """
    This is the classic Iris dataset. This is perhaps the best known database to
    be found in the pattern recognition literature.

    The fields of this dataset are:
        sepal_length - float
        sepal_width - float
        petal_length - float
        petal_width - float
        species - int, specifying iris species
    """

    def __init__(self):
        """
        Loads the Iris dataset.
        """
        x, y = load_iris(return_X_y=True)

        fields = Iris.get_default_fields()
        example_factory = ExampleFactory(fields)

        data = ((*x_, y_) for x_, y_ in zip(x, y))

        examples = [example_factory.from_list(raw_example) for raw_example in data]
        super().__init__(examples, fields)

    @staticmethod
    def get_default_fields():
        def identity(x):
            return x

        sepal_len_field = Field(
            "sepal_length", tokenizer=None, numericalizer=identity, keep_raw=True
        )
        sepal_width_field = Field(
            "sepal_width", tokenizer=None, numericalizer=identity, keep_raw=True
        )
        petal_len_field = Field(
            "petal_length", tokenizer=None, numericalizer=identity, keep_raw=True
        )
        petal_width_field = Field(
            "petal_width", tokenizer=None, numericalizer=identity, keep_raw=True
        )

        species_field = Field(
            "species",
            tokenizer=None,
            keep_raw=True,
            numericalizer=identity,
            is_target=True,
        )

        return (
            sepal_len_field,
            sepal_width_field,
            petal_len_field,
            petal_width_field,
            species_field,
        )

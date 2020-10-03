"""Module contains the CoNLL-U dataset."""
import logging

from podium.datasets import Dataset
from podium.storage import ExampleFactory, Field, Vocab


_LOGGER = logging.getLogger(__name__)


class CoNLLUDataset(Dataset):
    """A CoNLL-U dataset class. This class uses all default CoNLL-U fields."""

    def __init__(self, file_path, fields=None):
        """Dataset constructor.

        Parameters
        ----------
        file_path : str
            Path to the file containing the dataset.

        fields : dict(str, Field)
            Dictionary that maps the CoNLL-U field name to the field.
            If passed None the default set of fields will be used.
        """

        fields = fields or CoNLLUDataset.get_default_fields()
        examples = CoNLLUDataset._create_examples(file_path, fields)
        super().__init__(examples, fields)

    @staticmethod
    def _create_examples(file_path, fields):
        """Loads the dataset and extracts Examples.

        Parameters
        ----------
        file_path : str
            Path to the file wich contains the dataset.

        fields : dict(str, Field)
            Dictionary that maps the CoNLL-U field name to the field.

        Returns
        -------
        list(Example)
            A list of examples from the CoNLL-U dataset.

        Raises
        ------
        ImportError
            If the conllu library is not installed.

        ValueError
            If there is an error during parsing the file.
        """

        try:
            import conllu
        except ImportError as e:
            error_msg = (
                "Problem occurred while trying to import conllu. "
                "If the library is not installed visit "
                "https://pypi.org/project/conllu/ for more details."
            )
            _LOGGER.error(error_msg)
            raise e

        # we define a nested function that will catch parse exceptions,
        # but we don't let them point directly to the library code
        # so we raise `ValueError` and specify the original exception as a cause
        def safe_conllu_parse(in_file):
            try:
                yield from conllu.parse_incr(in_file)
            except Exception as e:
                error_msg = "Error occured during parsing the file"
                _LOGGER.error(error_msg)
                raise ValueError(error_msg) from e

        example_factory = ExampleFactory(fields)

        examples = []
        with open(file_path, encoding="utf-8") as in_file:

            for tokenlist in safe_conllu_parse(in_file):
                for token in tokenlist:
                    token = {
                        field_name: tuple(field_value.items())
                        if isinstance(field_value, dict)
                        else field_value
                        for field_name, field_value in token.items()
                    }

                    examples.append(example_factory.from_dict(token))

        return examples

    @staticmethod
    def get_default_fields():
        """Method returns a dict of default CoNLL-U fields.
        fields : id, form, lemma, upos, xpos, feats, head, deprel, deps, misc

        Returns
        -------
        fields : dict(str, Field)
            Dict containing all default CoNLL-U fields.
        """

        # numericalization of id is not allowed because
        # numericalization of integer ranges is undefined
        id = Field(name="id", tokenize=False, store_as_raw=True, is_numericalizable=False)

        form = Field(
            name="form", vocab=Vocab(specials=()), tokenize=False, store_as_raw=True
        )

        lemma = Field(
            name="lemma", vocab=Vocab(specials=()), tokenize=False, store_as_raw=True
        )

        upos = Field(
            name="upos",
            vocab=Vocab(specials=()),
            tokenize=False,
            store_as_raw=True,
            allow_missing_data=True,
        )

        xpos = Field(
            name="xpos",
            vocab=Vocab(specials=()),
            tokenize=False,
            store_as_raw=True,
            allow_missing_data=True,
        )

        feats = Field(
            name="feats",
            tokenize=False,
            store_as_tokenized=True,
            is_numericalizable=False,
            allow_missing_data=True,
        )

        head = Field(
            name="head",
            tokenize=False,
            store_as_raw=True,
            custom_numericalize=int,
            allow_missing_data=True,
        )

        deprel = Field(
            name="deprel", tokenize=False, store_as_raw=True, allow_missing_data=True
        )

        deps = Field(
            name="deps",
            tokenize=False,
            store_as_tokenized=True,
            is_numericalizable=False,
            allow_missing_data=True,
        )

        misc = Field(
            name="misc",
            tokenize=False,
            store_as_tokenized=True,
            is_numericalizable=False,
            allow_missing_data=True,
        )

        return {
            "id": id,
            "form": form,
            "lemma": lemma,
            "upos": upos,
            "xpos": xpos,
            "feats": feats,
            "head": head,
            "deprel": deprel,
            "deps": deps,
            "misc": misc,
        }

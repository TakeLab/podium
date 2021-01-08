"""
Module contains the CoNLL-U dataset.
"""
import collections

from podium.datasets import Dataset
from podium.datasets.example_factory import ExampleFactory
from podium.field import Field
from podium.vocab import Vocab


class CoNLLUDataset(Dataset):
    """
    A CoNLL-U dataset class.

    This class uses all default CoNLL-U fields.
    """

    def __init__(self, file_path, fields=None):
        """
        Dataset constructor.

        Parameters
        ----------
        file_path : str
            Path to the file containing the dataset.
        fields : Dict[str, Field]
            Dictionary that maps the CoNLL-U field name to the field.
            If passed None the default set of fields will be used.
        """

        fields = fields or CoNLLUDataset.get_default_fields()
        examples = CoNLLUDataset._create_examples(file_path, fields)
        super().__init__(examples, fields)

    @staticmethod
    def _create_examples(file_path, fields):
        """
        Loads the dataset and extracts Examples.

        Parameters
        ----------
        file_path : str
            Path to the file wich contains the dataset.
        fields : Dict[str, Field]
            Dictionary that maps the CoNLL-U field name to the field.

        Returns
        -------
        List[Example]
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
        except ImportError:
            print(
                "Problem occurred while trying to import conllu. "
                "If the library is not installed visit "
                "https://pypi.org/project/conllu/ for more details."
            )
            raise

        # we define a nested function that will catch parse exceptions,
        # but we don't let them point directly to the library code
        # so we raise `ValueError` and specify the original exception as a cause
        def safe_conllu_parse(in_file):
            try:
                yield from conllu.parse_incr(in_file)
            except Exception as e:
                raise ValueError("Error occured during parsing the file") from e

        field_names = list(fields)
        diff = set(field_names) - set(conllu.parser.DEFAULT_FIELDS)
        assert not diff, (
            "Only default CoNLL-U fields are supported; "
            f"found unsupported fields: {diff}"
        )

        example_factory = ExampleFactory(fields)

        examples = []
        with open(file_path, encoding="utf-8") as in_file:
            for tokenlist in safe_conllu_parse(in_file):
                example_dict = collections.defaultdict(lambda: [])
                for token in tokenlist:
                    for field_name in field_names:
                        example_dict[field_name].append(token[field_name])

                example = example_factory.from_dict(example_dict)
                examples.append(example)

        return examples

    @staticmethod
    def get_default_fields():
        """
        Method returns a dict of default CoNLL-U fields.

        Returns
        -------
        fields : Dict[str, Field]
            Dict containing all default CoNLL-U fields.
        """

        id = Field(name="id", tokenizer=None, numericalizer=None)

        form = Field(name="form", tokenizer=None, numericalizer=Vocab(specials=()))

        lemma = Field(name="lemma", tokenizer=None, numericalizer=Vocab(specials=()))

        upos = Field(
            name="upos",
            tokenizer=None,
            numericalizer=Vocab(specials=()),
        )

        xpos = Field(
            name="xpos",
            tokenizer=None,
            numericalizer=Vocab(specials=()),
        )

        feats = Field(name="feats", tokenizer=None, numericalizer=None)

        head = Field(
            name="head",
            tokenizer=None,
            numericalizer=int,
        )

        deprel = Field(name="deprel", tokenizer=None)

        deps = Field(name="deps", tokenizer=None, numericalizer=None)

        misc = Field(name="misc", tokenizer=None, numericalizer=None)

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

"""Module contains the CoNLL-U dataset."""
import csv
import functools

from podium.datasets import Dataset
from podium.storage import Field, LabelField, ExampleFactory, Vocab


class CoNLLUDataset(Dataset):
    """CoNLL-U dataset.

    Attributes
    ----------
    FIELD_NAMES : list(str)
        tuple of all default CoNLL-U field names
    """

    FIELDS_NAMES = ('id', 'form', 'lemma', 'upos', 'xpos',
                    'feats', 'head', 'deprel', 'deps', 'misc')

    def __init__(self, file_path, fields=None):
        """Dataset constructor.

        Parameters
        ----------
        file_path : str
            path to the file containing the dataset.

        fields : dict(str, Field)
            dictionary that maps the CoNLL-U field name to the field.
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
            dictionary that maps the CoNLL-U field name to the field.

        Returns
        -------
        list(Example)
            A list of examples from the CoNLL-U dataset.
        """

        example_factory = ExampleFactory(fields)

        examples = []
        with open(file_path, encoding='utf-8') as f:
            reader = csv.reader((line for line in f
                                if not (line.startswith('#') or line.isspace())),
                                delimiter='\t')

            for row in reader:
                # columns with '_' are marked as empty
                # and replaced with `None`
                # so the fields can process them as missing data
                row = {field_name: col if col != '_' else None
                       for field_name, col in zip(CoNLLUDataset.FIELDS_NAMES, row)}
                examples.append(example_factory.from_dict(row))

        return examples

    @staticmethod
    def get_default_fields():
        """Method returns a dict of default CoNLL-U fields.
        fields : id, form, lemma, upos, xpos, feats, head, deprel, deps, misc

        Returns
        -------
        fields : dict(str, Field)
            dict containing all default CoNLL-U fields.
        """

        def feats_tokenizer(string):
            return [feature.split('=') for feature in string.split('|')]

        def deps_tokenizer(string):
            return [dep.split(':') for dep in string.split('|')]

        misc_tokenizer = functools.partial(str.split, sep='|')

        # numericalization of id is not allowed because
        # numericalization of integer ranges is undefined
        id = Field(name='id',
                   tokenize=False,
                   store_as_raw=True,
                   is_numericalizable=False)

        form = Field(name='form',
                     vocab=Vocab(specials=()),
                     tokenize=False,
                     store_as_raw=True,
                     allow_missing_data=True)

        lemma = Field(name='lemma',
                      vocab=Vocab(specials=()),
                      tokenize=False,
                      store_as_raw=True,
                      allow_missing_data=True)

        upos = LabelField(name='upos',
                          allow_missing_data=True)

        xpos = LabelField(name='xpos',
                          allow_missing_data=True)

        feats = Field(name='feats',
                      tokenizer=feats_tokenizer,
                      is_numericalizable=False,
                      allow_missing_data=True)

        head = LabelField(name='head',
                          custom_numericalize=int,
                          allow_missing_data=True)

        deprel = LabelField(name='deprel',
                            allow_missing_data=True)

        deps = Field(name='deps',
                     tokenizer=deps_tokenizer,
                     is_numericalizable=False,
                     allow_missing_data=True)

        # misc can be formatted as a list that is split on '|'
        misc = Field(name='misc',
                     tokenizer=misc_tokenizer,
                     is_numericalizable=False,
                     allow_missing_data=True)

        return {
            'id': id,
            'form': form,
            'lemma': lemma,
            'upos': upos,
            'xpos': xpos,
            'feats': feats,
            'head': head,
            'deprel': deprel,
            'deps': deps,
            'misc': misc
        }

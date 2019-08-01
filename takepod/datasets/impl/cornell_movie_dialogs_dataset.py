"""Module contains Cornell Movie Dialogs datasets."""
import logging

from takepod.datasets.dataset import Dataset
from takepod.storage.example_factory import ExampleFactory
from takepod.storage.vocab import Vocab
from takepod.storage.field import Field, unpack_fields
from takepod.dataload.cornel_movie_dialogs import CornellMovieDialogsNamedTuple

_LOGGER = logging.getLogger(__name__)


class CornellMovieDialogsConversationalDataset(Dataset):
    """Cornell Movie Dialogs Conversational dataset which contains sentences and replies
    from movies."""
    def __init__(self, data, fields=None):
        """Dataset constructor.

        Parameters
        ----------
        data : CornellMovieDialogsNamedTuple
            cornell movie dialogs data
        fields : dict(str : Field)
            dictionary that maps field name to the field

        Raises
        ------
        ValueError
            if given data is None
        """
        if data is None:
            error_msg = "Specified data is None, dataset expects "\
                        "CornellMovieDialogsNamedTuple instance."
            _LOGGER.error(error_msg)
            raise ValueError(error_msg)
        if not fields:
            fields = CornellMovieDialogsConversationalDataset.get_default_fields()
        examples = CornellMovieDialogsConversationalDataset._create_examples(
            data=data, fields=fields
        )
        unpacked_fields = unpack_fields(fields=fields)
        super(CornellMovieDialogsConversationalDataset, self).__init__(
            **{"examples": examples, "fields": unpacked_fields})

    @staticmethod
    def _create_examples(data: CornellMovieDialogsNamedTuple, fields):
        """Method creates examples for Cornell Movie Dialogs dataset.

        Examples are created from the lines and conversations in data.

        Parameters
        ----------
        data : CornellMovieDialogsNamedTuple
            cornell movie dialogs data
        fields : dict(str : Field)
            dictionary mapping field names to fields

        Returns
        -------
        list(Example)
            list of created examples
        """
        example_factory = ExampleFactory(fields)
        examples = []
        lines = data.lines
        lines_dict = dict(zip(lines["lineID"], lines["text"]))
        conversations_lines = data.conversations["utteranceIDs"]
        for lines in conversations_lines:
            # we skip monologues
            if len(lines) < 2:
                continue
            for i in range(len(lines) - 1):
                statement = lines_dict.get(lines[i])
                reply = lines_dict.get(lines[i + 1])
                # some lines in the dataset are empty
                if not statement or not reply:
                    continue
                examples.append(example_factory.from_dict(
                    {"statement": statement, "reply": reply}))
        return examples

    @staticmethod
    def get_default_fields():
        """Method returns default Cornell Movie Dialogs fields: sentence and reply.
        Fields share same vocabulary.

        Returns
        -------
        fields : dict(str, Field)
            Dictionary mapping field name to field.
        """
        vocabulary = Vocab()
        statement = Field(name="statement", vocab=vocabulary, tokenizer="split",
                          language="en", tokenize=True, store_as_raw=False,
                          is_target=False)
        reply = Field(name="reply", vocab=vocabulary, tokenizer="split",
                      language="en", tokenize=True, store_as_raw=False, is_target=True)
        fields = {"statement": statement, "reply": reply}
        return fields
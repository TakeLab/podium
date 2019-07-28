from takepod.storage import dataset, ExampleFactory, Vocab, Field
from takepod.dataload.cornel_movie_dialogs import CornellMovieDialogsNamedTuple


class CornellMovieDialogsConversationalDataset(dataset.Dataset):
    def __init__(self, data, fields=None):
        if not fields:
            fields = CornellMovieDialogsConversationalDataset.get_default_fields()
        examples = CornellMovieDialogsConversationalDataset._create_examples(
            data=data, fields=fields
        )
        unpacked_fields = dataset.unpack_fields(fields=fields)
        super(CornellMovieDialogsConversationalDataset, self).__init__(
            **{"examples": examples, "fields": unpacked_fields})

    @staticmethod
    def _create_examples(data: CornellMovieDialogsNamedTuple, fields):
        example_factory = ExampleFactory(fields)
        examples = []
        lines = data.lines
        lines_dict = dict(zip(lines["lineID"], lines["text"]))
        conversations_lines = data.conversations["utteranceIDs"]
        for lines in conversations_lines:
            if len(lines) < 2:
                continue
            for i in range(len(lines)-1):
                statement = lines_dict[lines[i]]
                reply = lines_dict[lines[i+1]]
                if not statement or not reply:
                    continue
                examples.append(example_factory.from_dict(
                    {"statement": statement, "reply": reply}))
        return examples

    @staticmethod
    def get_default_fields():
        vocabulary = Vocab()
        statement = Field(name="statement", vocab=vocabulary, tokenizer="split",
                          language="en", tokenize=True, store_as_raw=False,
                          is_target=False)
        reply = Field(name="reply", vocab=vocabulary, tokenizer="split",
                      language="en", tokenize=True, store_as_raw=False, is_target=True)
        fields = {"statement": statement, "reply": reply}
        return fields

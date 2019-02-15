"""Module contains generic sequence labelling dataset."""
from takepod.storage import dataset, example


class SequenceLabellingDataset(dataset.Dataset):
    """
    Sequence labelling dataset. Examples in this dataset contain paired
    lists of words and tags. A single example represents a single sentence.

    Example for NER dataset:
    ['TakeLab', 'is', 'awesome', '.'] is paired with
    ['B-Organization', 'O', 'O', 'O']
    """

    def __init__(self, tokenized_documents, fields):
        """
        Dataset constructor.

        Parameters
        ----------
        tokenized_documents : list of lists of tuples
            List of tokenized documents. Each document is represented
            as a list of tuples (token, label). The sentences in document are
            delimited by tuple (None, None)

        fields : list[Field]
            List of fields
        """
        examples = []
        columns = []

        for document in tokenized_documents:
            for line in document:
                if is_delimiter_line(line):
                    if columns:
                        examples.append(example.Example.fromlist(columns, fields))
                    columns = []
                else:
                    for i, column in enumerate(line):
                        if len(columns) < i + 1:
                            columns.append([])
                        columns[i].append(column)

        super(SequenceLabellingDataset, self).__init__(examples, fields)


def is_delimiter_line(line):
    """
    Checks if the line is delimiter line. Delimiter line is a tuple with
    all elements set to None.

    Parameters
    ----------
    line : tuple
        tuple representing line elements.

    Returns
    -------
        True if the line is delimiter line.
    """
    return not any(line)

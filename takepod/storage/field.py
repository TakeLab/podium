import numpy as np


class Field(object):
    """
    Holds the preprocessing and numericalization logic for a single
    field of a dataset.
    """

    def __init__(self,
                 name,
                 tokenizer='split',
                 language='en',
                 vocab=None,
                 sequential=True,
                 store_raw=True,
                 eager=True,
                 custom_numericalize=float,
                 ):
        """Create a Field from arguments.

        Parameters
        ----------
        tokenizer : str | callable
            The tokenizer that is to be used when preprocessing raw data
            (only if sequential is True). The user can provide his own
            tokenizer as a callable object or specify one of the premade
            tokenizers by a string. The available premade tokenizers are:
                - 'split' - default str.split()
                - 'spacy' - the spacy tokenizer, using the 'en' language
                model by default (unless the user provides a different
                'language' parameter)
        language : str
            The language argument for the tokenizer (if necessary, e. g. for
            spacy).
            Default is 'en'.
        vocab : Vocab
            A vocab that this field will update after preprocessing and
            use to numericalize data.
            If None, the field won't use a vocab, in which case a custom
            numericalization function has to be given.
            Default is None.
        sequential : bool
            Whether the data should be tokenized when being preprocessed.
        store_raw : bool
            Whether to store untokenized data after tokenization.
        eager : bool
            Whether to build the vocabulary online, each time the field
            preprocesses raw data.
        custom_numericalize : callable
            The numericalization function that will be called if the field
            doesn't use a vocabulary.

        Raises
        ------
        ValueError
            If the given tokenizer is not a callable or a string, or is a
            string that doesn't correspond to any of the premade tokenizers.
        """

        self.name = name
        self.language = language

        if not store_raw and not sequential:
            raise ValueError(
                "Either 'store_raw' or 'sequential' must be true.")

        self.store_raw = store_raw
        self.sequential = sequential

        self.eager = eager
        self.vocab = vocab

        self.tokenizer = get_tokenizer(tokenizer, language)
        self.custom_numericalize = custom_numericalize

    @property
    def use_vocab(self):
        """
        A flag that tells whether the field uses a vocab or not.

        Returns
        -------
        bool
            Whether the field uses a vocab or not.
        """

        return self.vocab is not None

    def preprocess(self, raw):
        """Preprocesses raw data, tokenizing it if the field is sequential,
        updating the vocab if the field is eager and preserving the raw data
        if field's 'store_raw' is true.

        Parameters
        ----------
        raw : str
            The raw data that needs to be preprocessed.

        Returns
        -------
        (str, str)
            A tuple of strings (raw, tokenized). If the field's 'store_raw'
            attribute is False, then 'raw' will be None (we don't preserve
            the raw data). If field's 'sequential' attribute is False then
            'tokenized' will be None.
            The attributes 'sequential' and 'store_raw' will never both be
            False, so the function will never return (None, None).
        """

        tokenized = self.tokenizer(raw) if self.sequential else None
        raw = raw if self.store_raw else None

        if self.eager:
            self.update_vocab(raw, tokenized)

        return raw, tokenized

    def update_vocab(self, raw, tokenized):
        """Updates the vocab with a data point in its raw and tokenized form.
        If the field is sequential, the vocab is updated with the tokenized
        form (and 'raw' can be None), otherwise the raw form is used to
        update (and 'tokenized' can be None).

        Parameters
        ----------
        raw : str
            The raw form of the data point that the vocab is to be updated
            with. If the field is sequential, this parameter is ignored and
            can be None.
        tokenized : str
            The tokenized form of the data point that the vocab is to be
            updated with. If the field is NOT sequential, this parameter is
            ignored and can be None.
        """

        if not self.use_vocab:
            return

        data = tokenized if self.sequential else [raw]
        self.vocab += data

    def finalize(self):
        """Signals that this field's vocab can be built.
        """

        if self.use_vocab:
            self.vocab.finalize()

    def numericalize(self, data):
        """Numericalize the already preprocessed data point based either on
        the vocab that was previously built, or on a custom numericalization
        function, if the field doesn't use a vocab.

        Parameters
        ----------
        data : (str, str)
            Tuple of (raw, tokenized) of preprocessed input data. If the field
            is sequential, 'raw' is ignored and can be None. Otherwise,
            'sequential' is ignored and can be None.
        """

        raw, tokenized = data

        # raw data is just a string, so we need to wrap it into an iterable
        data = tokenized if self.sequential else [raw]

        if self.use_vocab:
            return self.vocab.numericalize(data)
        else:
            # custom numericalization for non-vocab data
            # (such as floating point data Fields)
            return np.array([self.custom_numericalize(tok) for tok in data])

    def pad_to_length(self, row, length, custom_pad_symbol=None,
                      pad_left=False, truncate_left=False):
        """Either pads the given row with pad symbols, or truncates the row
        to be of given length. The vocab provides the pad symbol for all
        fields that have vocabs, otherwise the pad symbol has to be given as
        a parameter.

        Parameters
        ----------
        row : np.ndarray
            The row of numericalized data that is to be padded / truncated.
        length : int
            The desired length of the row.
        custom_pad_symbol : int
            The pad symbol that is to be used if the field doesn't have a
            vocab. If the field has a vocab, this parameter is ignored and can
            be None.

        Raises
        ------
        ValueError
            If the field doesn't use a vocab and no custom pad symbol was
            given.
        """

        if len(row) > length:
            # truncating

            if truncate_left:
                row = row[len(row) - length:]
            else:
                row = row[:length]

        elif len(row) < length:
            # padding

            if self.use_vocab:
                pad_symbol = self.vocab.pad_symbol
            else:
                pad_symbol = custom_pad_symbol

            if pad_symbol is None:
                raise ValueError('Must provide a custom pad symbol if the '
                                 'field has no vocab.')

            diff = length - len(row)

            if pad_left:
                row = np.append([pad_symbol] * diff, row)
            else:
                row = np.append(row, [pad_symbol] * diff)

        return row


def get_tokenizer(tokenizer, language='en'):
    """
    Returns a tokenizer according to the parameters given.

    Parameters
    ----------
    tokenizer : str | callable
        If a callable object is given, it will just be returned. Otherwise, a
        string can be given to create one of the premade tokenizers.
        The available premade tokenizers are:
            - 'split' - default str.split()
            - 'spacy' - the spacy tokenizer, using the 'en' language
            model by default (unless the user provides a different
            'language' parameter)

    language : str
        The language argument for the tokenizer (if necessary, e. g. for
        spacy). Default is 'en'.

    Returns
    -------
        The created (or given) tokenizer.

    Raises
    ------
    ValueError
        If the given tokenizer is not a callable or a string, or is a string
        that doesn't correspond to any of the premade tokenizers.
    """
    # Add every new tokenizer to this "factory" method
    if callable(tokenizer):
        # if arg is already a function, just return it
        return tokenizer

    elif tokenizer == 'spacy':
        try:
            import spacy
            spacy_tokenizer = spacy.load(language)

            # closures instead of lambdas because they are serializable
            def spacy_tokenize(string):
                # need to wrap in a function to access .text
                return [token.text for token in
                        spacy_tokenizer.tokenizer(string)]

            return spacy_tokenize
        except OSError:
            print(f'Please install SpaCy and the SpaCy {language} tokenizer. '
                  f'See the docs at https://spacy.io for more information.')
            raise

    elif tokenizer == "split":
        return str.split

    else:
        raise ValueError(f"Wrong value given for the tokenizer: {tokenizer}")

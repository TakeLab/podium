"""Module contains dataset's field definition and methods for construction."""
from collections import deque

import numpy as np

from takepod.preproc.tokenizers import get_tokenizer


class Field(object):
    """Holds the preprocessing and numericalization logic for a single
    field of a dataset.
    """

    def __init__(self,
                 name,
                 tokenizer='split',
                 language='en',
                 vocab=None,
                 tokenize=True,
                 store_as_raw=True,
                 store_as_tokenized=False,
                 eager=True,
                 custom_numericalize=float,
                 is_target=False,
                 fixed_length=None
                 ):
        """Create a Field from arguments.

        Parameters
        ----------
        name : str
            Field name, used for referencing data in the dataset.
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
        tokenize : bool
            Whether the data should be tokenized when being preprocessed.
            If True, store_as_tokenized must be False.
        store_as_raw : bool
            Whether to store untokenized preprocessed data.
            If True, ''
        store_as_tokenized : bool
            Whether to store the data as tokenized.
            Data will be stored as-is and no preprocessing will be done.
            If True, store_raw and tokenize must be False.
        eager : bool
            Whether to build the vocabulary online, each time the field
            preprocesses raw data.
        custom_numericalize : callable
            The numericalization function that will be called if the field
            doesn't use a vocabulary.
        is_target : bool
            Whether this field is a target variable. Affects iteration over
            batches. Default: False.
        fixed_length : int, optional
            To which length should the field be fixed. If it is not None every
            example in the field will be truncated or padded to given length.
            Default: None.

        Raises
        ------
        ValueError
            If the given tokenizer is not a callable or a string, or is a
            string that doesn't correspond to any of the premade tokenizers.
        """

        self.name = name
        self.language = language
        self.sequential = tokenize or store_as_tokenized

        if store_as_tokenized and tokenize:
            raise ValueError(
                "store_as_tokenized' and 'tokenize' both set to True."
                " They can either both be False, or only one set to True"
            )

        if not store_as_raw and not self.sequential:
            raise ValueError(
                "Either 'store_as_raw', 'tokenize_raw'"
                " or 'store_as_tokenized' must be True.")

        if store_as_raw and store_as_tokenized:
            raise ValueError(
                "'store_as_raw' and 'store_as_tokenized' both set to True."
                " They can either both be False, or only one set to True"
            )

        self.store_as_raw = store_as_raw
        self.tokenize = tokenize
        self.store_as_tokenized = store_as_tokenized

        self.eager = eager
        self.vocab = vocab

        self.tokenizer = get_tokenizer(tokenizer, language)
        self.custom_numericalize = custom_numericalize

        self.is_target = is_target
        self.fixed_length = fixed_length

        self.pretokenize_hooks = deque()
        self.posttokenize_hooks = deque()

    @property
    def use_vocab(self):
        """A flag that tells whether the field uses a vocab or not.

        Returns
        -------
        bool
            Whether the field uses a vocab or not.
        """

        return self.vocab is not None

    def add_pretokenize_hook(self, hook):
        """Add a pre-tokenization hook to the Field.
        If multiple hooks are added to the field, the order of their execution
        will be the same as the order in which they were added to the field,
        each subsequent hook taking the output of the previous hook as its
        input.
        If the same function is added to the Field as a hook multiple times,
        it will be executed that many times.
        The output of the final pre-tokenization hook is the raw data that the
        tokenizer will get as its input.

        Pretokenize hooks have the following signature:
            func pre_tok_hook(raw_data):
                raw_data_out = do_stuff(raw_data)
                return raw_data_out

        This can be used to eliminate encoding errors in data, replace numbers
        and names, etc.

        Parameters
        ----------
        hook : callable
            The pre-tokenization hook that we want to add to the field.
        """

        self.pretokenize_hooks.append(hook)

    def add_posttokenize_hook(self, hook):
        """Add a post-tokenization hook to the Field.
        If multiple hooks are added to the field, the order of their execution
        will be the same as the order in which they were added to the field,
        each subsequent hook taking the output of the previous hook as its
        input.
        If the same function is added to the Field as a hook multiple times,
        it will be executed that many times.
        Post-tokenization hooks are called only if the Field is sequential
        (in non-sequential fields there is no tokenization and only
        pre-tokenization hooks are called).
        The output of the final post-tokenization hook are the raw and
        tokenized data that the preprocess function will use to produce its
        result.

        Posttokenize hooks have the following outline:
            func post_tok_hook(raw_data, tokenized_data):
                raw_out, tokenized_out = do_stuff(raw_data, tokenized_data)
                return raw_out, tokenized_out

        where 'tokenized_data' is and 'tokenized_out' should be an iterable.

        Parameters
        ----------
        hook : callable
            The post-tokenization hook that we want to add to the field.
        """

        self.posttokenize_hooks.append(hook)

    def remove_pretokenize_hooks(self):
        """Remove all the pre-tokenization hooks that were added to the Field.
        """
        self.pretokenize_hooks.clear()

    def remove_posttokenize_hooks(self):
        """Remove all the post-tokenization hooks that were added to the Field.
        """
        self.posttokenize_hooks.clear()

    def _run_pretokenization_hooks(self, data):
        for hook in self.pretokenize_hooks:
            data = hook(data)

        return data

    def _run_posttokenization_hooks(self, data, tokens):
        for hook in self.posttokenize_hooks:
            data, tokens = hook(data, tokens)

        return data, list(tokens)

    def preprocess(self, data):
        """Preprocesses raw data, tokenizing it if the field is sequential,
        updating the vocab if the field is eager and preserving the raw data
        if field's 'store_raw' is true.

        Parameters
        ----------
        data : str or iterable(hashable)
            The raw data that needs to be preprocessed.
            String if store_as_raw and/or tokenize_raw attributes are True.
            iterable(hashable) if store_as_tokenized attribute is True.

        Returns
        -------
        (str, Iterable(hashable))
            A tuple of (raw, tokenized). If the field's 'store_as_raw'
            attribute is False, then 'raw' will be None (we don't preserve
            the raw data). If field's 'tokenize' and 'store_as_tokenized' attributes
            are False then 'tokenized' will be None.
            The attributes 'store_as_raw', 'store_as_tokenized' and 'tokenize' will never
            all be False, so the function will never return (None, None).
        """

        tokens = None

        if self.store_as_tokenized:
            # Store data as tokens
            _, tokens = self._run_posttokenization_hooks(None, data)

        else:
            # Preprocess the raw input

            data = self._run_pretokenization_hooks(data)

            if self.tokenize:
                # Tokenize the preprocessed raw data

                tokens = self.tokenizer(data)

                data, tokens = self._run_posttokenization_hooks(data, tokens)

        if self.eager and self.use_vocab:
            self.update_vocab(data, tokens)

        raw = data if self.store_as_raw else None

        return raw, tokens

    def update_vocab(self, raw, tokenized):
        """Updates the vocab with a data point in its raw and tokenized form.
        If the field is sequential, the vocab is updated with the tokenized
        form (and 'raw' can be None), otherwise the raw form is used to
        update (and 'tokenized' can be None).

        Parameters
        ----------
        raw : hashable
            The raw form of the data point that the vocab is to be updated
            with. If the field is sequential, this parameter is ignored and
            can be None.
        tokenized : iterable(hashable)
            The tokenized form of the data point that the vocab is to be
            updated with. If the field is NOT sequential
            ('store_as_tokenized' and 'tokenize' attributes are False),
            this parameter is ignored and can be None.
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

    def _numericalize_tokens(self, tokens):
        """Numericalizes an iterable of tokens.
        If use_vocab is True, numericalization of the vocab is used. Else
        the custom_numericalize hook is used.

        Parameters
        ----------
        tokens : iterable(hashable)
            Iterable of hashable objects to be numericalized.

        Returns
        -------
        numpy array
            Array of numericalized representations of the tokens.

        """
        if self.use_vocab:
            return self.vocab.numericalize(tokens)

        else:
            # custom numericalization for non-vocab data
            # (such as floating point data Fields)
            return np.array([self.custom_numericalize(tok) for tok in tokens])

    def numericalize(self, data):
        """Numericalize the already preprocessed data point based either on
        the vocab that was previously built, or on a custom numericalization
        function, if the field doesn't use a vocab.

        Parameters
        ----------
        data : (hashable, iterable(hashable))
            Tuple of (raw, tokenized) of preprocessed input data. If the field
            is sequential, 'raw' is ignored and can be None. Otherwise,
            'sequential' is ignored and can be None.

        Returns
        -------
        numpy array
            Array of stoi indexes of the tokens.

        """
        raw, tokenized = data

        # raw data is just a string, so we need to wrap it into an iterable
        tokens = tokenized if self.sequential else [raw]

        return self._numericalize_tokens(tokens)

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
        pad_left : bool
            If True padding will be done on the left side, otherwise on the
            right side. Default: False.
        truncate_left : bool
            If True field will be trucated on the left side, otherwise on the
            right side. Default: False.

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
                pad_symbol = self.vocab.pad_symbol()
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


class TokenizedField(Field):
    """
    Tokenized version of the Field. Holds the preprocessing and
    numericalization logic for the pre-tokenized dataset fields.
    """

    def __init__(self,
                 name,
                 vocab=None,
                 eager=True,
                 custom_numericalize=float,
                 is_target=False,
                 fixed_length=None):

        super().__init__(
            name=name,
            vocab=vocab,
            store_as_raw=False,
            tokenize=False,
            store_as_tokenized=True,
            eager=eager,
            custom_numericalize=custom_numericalize,
            is_target=is_target,
            fixed_length=fixed_length
        )


class MultilabelField(TokenizedField):

    def __init__(self,
                 name,
                 vocab=None,
                 eager=True,
                 custom_numericalize=float,
                 is_target=False,
                 fixed_length=None):

        if vocab is not None and vocab.has_specials:
            raise ValueError("Vocab contains special symbols."
                             " Vocabs with special symbols cannot be used"
                             " with multilabel fields.")

        super().__init__(name,
                         vocab=vocab,
                         eager=eager,
                         custom_numericalize=custom_numericalize,
                         is_target=is_target,
                         fixed_length=fixed_length)

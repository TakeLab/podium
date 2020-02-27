"""Module contains dataset's field definition and methods for construction."""
import logging
import itertools
from collections import deque

import numpy as np

from takepod.preproc.tokenizers import get_tokenizer

_LOGGER = logging.getLogger(__name__)


class PretokenizationPipeline:

    def __init__(self, hooks=()):
        self.hooks = deque(hooks)

    def add_hook(self, hook):
        self.hooks.append(hook)

    def process(self, raw):
        processed = raw

        for hook in self.hooks:
            processed = hook(processed)

        return processed

    def __call__(self, raw):
        return self.process(raw)

    def clear(self):
        self.hooks.clear()


class PosttokenizationPipeline:
    def __init__(self, hooks=()):
        self.hooks = deque(hooks)

    def add_hook(self, hook):
        self.hooks.append(hook)

    def process(self, raw, tokenized):
        processed_raw, processed_tokenized = raw, tokenized

        for hook in self.hooks:
            processed_raw, processed_tokenized = hook(processed_raw, processed_tokenized)

        if processed_tokenized is not None \
                and not isinstance(processed_tokenized, (list, tuple)):
            processed_tokenized = list(processed_tokenized)

        return processed_raw, processed_tokenized

    def __call__(self, raw, tokenized):
        return self.process(raw, tokenized)

    def clear(self):
        self.hooks.clear()


class MultioutputField:
    """Field that does pretokenization and tokenization once and passes it to its
    output fields. Output fields are any type of field. The output fields are used only
    for posttokenization processing (posttokenization hooks and vocab updating)."""

    def __init__(self,
                 output_fields,
                 tokenizer='split',
                 language='en'):
        """Field that does pretokenization and tokenization once and passes it to its
        output fields. Output fields are any type of field. The output fields are used
        only for posttokenization processing (posttokenization hooks and vocab updating).

        Parameters
        ----------
         output_fields : iterable
            iterable containig the output fields. The pretokenization hooks and tokenizer
            in these fields are ignored and only posttokenization hooks are used.
         tokenizer : str | callable
            The tokenizer that is to be used when preprocessing raw data
            (only if 'tokenize' is True). The user can provide his own
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
        """

        self.language = language
        self._tokenizer_arg = tokenizer
        self.pretokenization_pipeline = PretokenizationPipeline()
        self.tokenizer = get_tokenizer(tokenizer, language)
        self.output_fields = deque(output_fields)

    def add_pretokenize_hook(self, hook):
        """Add a pre-tokenization hook to the MultioutputField.
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
        self.pretokenization_pipeline.add_hook(hook)

    def _run_pretokenization_hooks(self, data):
        """Runs pretokenization hooks on the raw data and returns the result.

        Parameters
        ----------
        data : hashable
            data to be processed

        Returns
        -------
        hashable
            processed data

        """

        return self.pretokenization_pipeline(data)

    def add_output_field(self, field):
        """
        Adds the passed field to this field's output fields.

        Parameters
        ----------
        field : Field
            Field to add to output fields.
        """
        self.output_fields.append(field)

    def preprocess(self, data):
        data = self._run_pretokenization_hooks(data)
        tokens = self.tokenizer(data)
        return tuple(field._process_tokens(data, tokens) for field in self.output_fields)

    def get_output_fields(self):
        """
        Returns an Iterable of the contained output fields.

        Returns
        -------
        Iterable :
            an Iterable of the contained output fields.
        """
        return self.output_fields

    def remove_pretokenize_hooks(self):
        """Remove all the pre-tokenization hooks that were added to the MultioutputField.
        """
        self.pretokenization_pipeline.clear()


class Field:
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
                 is_numericalizable=True,
                 custom_numericalize=None,
                 is_target=False,
                 fixed_length=None,
                 allow_missing_data=False,
                 missing_data_token=-1
                 ):
        """Create a Field from arguments.

        Parameters
        ----------
        name : str
            Field name, used for referencing data in the dataset.
        tokenizer : str | callable
            The tokenizer that is to be used when preprocessing raw data
            (only if 'tokenize' is True). The user can provide his own
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
            If True, the raw data will be run through the pretokenize hooks,
            tokenized using the tokenizer, run through the posttokenize hooks
            and then stored in the 'tokenized' part of the example tuple.
            If True, 'store_as_tokenized' must be False.
        store_as_raw : bool
            Whether to store untokenized preprocessed data.
            If True, the raw data will be run trough the provided pretokenize
            hooks and stored in the 'raw' part of the example tuple.
            If True, 'store_as_tokenized' must be False.
        store_as_tokenized : bool
            Whether to store the data as tokenized.
            If True, the raw data will be run through the provided posttokenize
            hooks and stored in the 'tokenized' part of the example tuple.
            If True, store_raw and tokenize must be False.
        eager : bool
            Whether to build the vocabulary online, each time the field
            preprocesses raw data.
        is_numericalizable : bool
            Whether the output of tokenizer can be numericalized.

            If true, the output of the tokenizer is presumed to be a list of tokens and
            will be numericalized using the provided Vocab or custom_numericalize.
            For numericalizable fields, Iterator will generate batch fields containing
             numpy matrices.

             If false, the out of the tokenizer is presumed to be a custom datatype.
             Posttokenization hooks aren't allowed to be added as they can't be called
             on custom datatypes. For non-numericalizable fields, Iterator will generate
             batch fields containing lists of these custom data type instances returned
             by the tokenizer.

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
        allow_missing_data : bool
            Whether the field allows missing data. In the case 'allow_missing_data'
            is false and None is sent to be preprocessed, an ValueError will be raised.
            If 'allow_missing_data' is True, if a None is sent to be preprocessed, it will
            be stored and later numericalized properly.
            Default: False
        missing_data_token : number
            Token to use to mark batch rows as missing. If data for a field is missing,
            its matrix row will be filled with this value. For non numericalizable fields,
            this parameter is ignored and the value will be None.
            Default: -1

        Raises
        ------
        ValueError
            If the given tokenizer is not a callable or a string, or is a
            string that doesn't correspond to any of the premade tokenizers.
        """

        self.name = name
        self.language = language
        self._tokenizer_arg = tokenizer
        self.is_numericalizable = is_numericalizable

        if store_as_tokenized and tokenize:
            error_msg = "Store_as_tokenized' and 'tokenize' both set to True." \
                        " You can either store the data as tokenized, " \
                        "tokenize it or do neither, but you can't do both."
            _LOGGER.error(error_msg)
            raise ValueError(error_msg)

        if not store_as_raw and not tokenize and not store_as_tokenized:
            error_msg = "At least one of 'store_as_raw', 'tokenize'" \
                        " or 'store_as_tokenized' must be True."
            _LOGGER.error(error_msg)
            raise ValueError(error_msg)

        if store_as_raw and store_as_tokenized:
            error_msg = "'store_as_raw' and 'store_as_tokenized' both set to" \
                        " True. You can't store the same value as raw and as " \
                        "tokenized. Maybe you wanted to tokenize the raw " \
                        "data? (the 'tokenize' parameter)"
            _LOGGER.error(error_msg)
            raise ValueError(error_msg)

        if not is_numericalizable \
                and (custom_numericalize is not None or vocab is not None):
            error_msg = "Field that is not numericalizable can't have " \
                        "custom_numericalize or vocab."

            _LOGGER.error(error_msg)
            raise ValueError(error_msg)

        self.is_sequential = (store_as_tokenized or tokenize) and is_numericalizable
        self.store_as_raw = store_as_raw
        self.tokenize = tokenize
        self.store_as_tokenized = store_as_tokenized

        self.eager = eager
        self.vocab = vocab

        self.tokenizer = get_tokenizer(tokenizer, language)
        self.custom_numericalize = custom_numericalize

        self.is_target = is_target
        self.fixed_length = fixed_length

        self.pretokenize_pipeline = PretokenizationPipeline()
        self.posttokenize_pipeline = PosttokenizationPipeline()
        self.allow_missing_data = allow_missing_data
        self.missing_data_token = missing_data_token

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
        self.pretokenize_pipeline.add_hook(hook)

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
        if not self.is_numericalizable:
            error_msg = "Field is declared as non numericalizable. Posttokenization " \
                        "hooks aren't used in such fields."
            _LOGGER.error(error_msg)
            raise ValueError(error_msg)

        self.posttokenize_pipeline.add_hook(hook)

    def remove_pretokenize_hooks(self):
        """Remove all the pre-tokenization hooks that were added to the Field.
        """
        self.pretokenize_pipeline.clear()

    def remove_posttokenize_hooks(self):
        """Remove all the post-tokenization hooks that were added to the Field.
        """
        self.posttokenize_pipeline.clear()

    def _run_pretokenization_hooks(self, data):
        """Runs pretokenization hooks on the raw data and returns the result.

        Parameters
        ----------
        data : hashable
            data to be processed

        Returns
        -------
        hashable
            processed data

        """
        return self.pretokenize_pipeline(data)

    def _run_posttokenization_hooks(self, data, tokens):
        """Runs posttokenization hooks on tokenized data.

        Parameters
        ----------
        data : hashable
            raw data that was processed with '_run_pretokenization_hooks'.

        tokens : iterable(hashable)
            iterable of tokens resulting from the tokenization of the processed raw data.

        Returns
        -------
        (data, list(tokens))
            Returns a tuple containing the data and list of tokens processed by
            posttokenization hooks.

        """
        return self.posttokenize_pipeline(data, tokens)

    def preprocess(self, data):
        """Preprocesses raw data, tokenizing it if the field is sequential,
        updating the vocab if the field is eager and preserving the raw data
        if field's 'store_raw' is true.

        Parameters
        ----------
        data : str or iterable(hashable)
            The raw data that needs to be preprocessed.
            String if 'store_as_raw' and/or 'tokenize' attributes are True.
            iterable(hashable) if store_as_tokenized attribute is True.

        Returns
        -------
        (str, Iterable(hashable))
            A tuple of (raw, tokenized). If the field's 'store_as_raw'
            attribute is False, then 'raw' will be None (we don't preserve
            the raw data). If field's 'tokenize' and 'store_as_tokenized'
            attributes are False then 'tokenized' will be None.
            The attributes 'store_as_raw', 'store_as_tokenized' and 'tokenize'
            will never all be False, so the function will never return (None, None).
        """

        if data is None:
            if not self.allow_missing_data:
                error_msg = "Missing data not allowed in field {}".format(self.name)
                _LOGGER.error(error_msg)
                raise ValueError(error_msg)

            else:
                return (self.name, (None, None)),

        if self.store_as_tokenized:
            # Store data as tokens
            data, tokens = None, data

        else:
            # Preprocess the raw input
            data = self._run_pretokenization_hooks(data)
            tokens = self.tokenizer(data) if self.tokenize else None

        return self._process_tokens(data, tokens),

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

        data = tokenized if self.tokenize or self.store_as_tokenized else [raw]
        self.vocab += data

    def finalize(self):
        """Signals that this field's vocab can be built.
        """

        if self.use_vocab:
            self.vocab.finalize()

    def _process_tokens(self, data, tokens):
        """
        Runs posttokenization processing on the provided data and tokens and updates
        the vocab if needed. Used by Multioutput field.

        Parameters
        ----------
        data
            data processed by Pretokenization hooks

        tokens : list
            tokenized data

        Returns
        -------
        name , (data, tokens)
            Returns and tuple containing this both field's name and a tuple containing
            the data and tokens processed by posttokenization hooks.
        """

        if self.is_numericalizable:
            data, tokens = self._run_posttokenization_hooks(data, tokens)

        if self.eager and self.use_vocab and not self.vocab.finalized:
            self.update_vocab(data, tokens)

        data = data if self.store_as_raw else None

        return self.name, (data, tokens)

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
        if self.custom_numericalize is None and self.use_vocab:
            return self.vocab.numericalize(tokens)

        # custom numericalization for non-vocab data
        # (such as floating point data Fields)
        return np.array([self.custom_numericalize(tok) for tok in tokens])

    def get_default_value(self):
        """Method obtains default field value for missing data.

        Returns
        -------
            missing_symbol index or None
                The index of the missing data token, if this field is numericalizable.
                None value otherwise.

        Raises
        ------
        ValueError
            If missing data is not allowed in this field.
        """
        if not self.allow_missing_data:
            error_msg = "Missing data not allowed in field {}".format(self.name)
            _LOGGER.error(error_msg)
            raise ValueError(error_msg)

        if self.is_numericalizable:
            return self.missing_data_token

        else:
            return None

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
            Array of stoi indexes of the tokens, if data exists.
            None, if data is missing and missing data is allowed.

        Raises
        ------
        ValueError
            If data is None and missing data is not allowed in this field.
        """
        raw, tokenized = data

        if raw is None and tokenized is None:
            if not self.allow_missing_data:
                error_msg = "Missing value found in field {}.".format(self.name)
                _LOGGER.error(error_msg)
                raise ValueError(error_msg)

            else:
                return None

        # raw data is just a string, so we need to wrap it into an iterable
        tokens = tokenized if self.tokenize or self.store_as_tokenized else [raw]

        if self.is_numericalizable:
            return self._numericalize_tokens(tokens)

        else:
            return tokens

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
                pad_symbol = self.vocab.pad_symbol_index()
            else:
                pad_symbol = custom_pad_symbol

            if pad_symbol is None:
                error_msg = 'Must provide a custom pad symbol if the ' \
                            'field has no vocab.'
                _LOGGER.error(error_msg)
                raise ValueError(error_msg)

            diff = length - len(row)

            if pad_left:
                row = np.pad(row, (diff, 0), 'constant',
                             constant_values=pad_symbol)
            else:
                row = np.pad(row, (0, diff), 'constant',
                             constant_values=pad_symbol)

        return row

    def get_numericalization_for_example(self, example, cache=True):
        """Returns the numericalized data of this field for the provided example.
        The numericalized data is generated and cached in the example if 'cache' is true
        and the cached data is not already present. If already cached, the cached data is
        returned.

        Parameters
        ----------
        example : Example
            example to get numericalized data for.

        cache : bool
            whether to store the cache the calculated numericalization if not already
            cached

        Returns
        -------
        numericalized data : numpy array
            The numericalized data.
        """
        cache_field_name = "{}_".format(self.name)
        numericalization = getattr(example, cache_field_name)

        if numericalization is None:
            example_data = getattr(example, self.name)
            numericalization = self.numericalize(example_data)
            if cache:
                setattr(example, cache_field_name, numericalization)

        return numericalization

    def __getstate__(self):
        """Method obtains field state. It is used for pickling dataset data
        to file.

        Returns
        -------
        state : dict
            dataset state dictionary
        """
        state = self.__dict__.copy()
        del state['tokenizer']
        return state

    def __setstate__(self, state):
        """Method sets field state. It is used for unpickling dataset data
        from file.

        Parameters
        ----------
        state : dict
            dataset state dictionary
        """
        self.__dict__.update(state)
        self.tokenizer = get_tokenizer(self._tokenizer_arg, self.language)

    def __str__(self):
        return "{}[name: {}, is_sequential: {}, is_target: {}]".format(
            self.__class__.__name__, self.name, self.is_sequential, self.is_target)

    def get_output_fields(self):
        """Returns an Iterable of the contained output fields.

        Returns
        -------
        Iterable :
            an Iterable of the contained output fields.
        """
        return self,


class TokenizedField(Field):
    """
    Tokenized version of the Field. Holds the preprocessing and
    numericalization logic for the pre-tokenized dataset fields.
    """

    def __init__(self,
                 name,
                 vocab=None,
                 eager=True,
                 custom_numericalize=None,
                 is_target=False,
                 fixed_length=None,
                 allow_missing_data=False):
        super().__init__(
            name=name,
            vocab=vocab,
            store_as_raw=False,
            tokenize=False,
            store_as_tokenized=True,
            eager=eager,
            custom_numericalize=custom_numericalize,
            is_target=is_target,
            fixed_length=fixed_length,
            allow_missing_data=allow_missing_data
        )


class MultilabelField(TokenizedField):
    """Class used for storing pre-tokenized labels.
    Used for multilabeled datasets.
    """

    def __init__(self,
                 name,
                 num_of_classes=None,
                 vocab=None,
                 eager=True,
                 allow_missing_data=False,
                 custom_numericalize=None):
        """Create a MultilabelField from arguments.

                Parameters
                ----------
                name : str
                    Field name, used for referencing data in the dataset.

                num_of_classes : int, optional
                    Number of valid classes.
                    Also defines size of the numericalized vector.
                    If none, size of the vocabulary is used.

                vocab : Vocab
                    A vocab that this field will update after preprocessing and
                    use to numericalize data.
                    If None, the field won't use a vocab, in which case a custom
                    numericalization function has to be given.
                    Default is None.

                eager : bool
                    Whether to build the vocabulary online, each time the field
                    preprocesses raw data.

                allow_missing_data : bool
                    Whether the field allows missing data.
                    In the case 'allow_missing_data'
                    is false and None is sent to be preprocessed, an ValueError
                    will be raised. If 'allow_missing_data' is True, if a None is sent to
                    be preprocessed, it will be stored and later numericalized properly.
                    If the field is sequential the numericalization
                    of a missing data field will be an empty numpy Array,
                    else the numericalization will be a numpy Array
                    containing a single np.Nan ([np.Nan])
                    Default: False

                custom_numericalize : callable(str) -> int
                    Callable that takes a string and returns an int.
                    Used to index classes.

                Raises
                ------
                ValueError
                    If the provided Vocab contains special symbols.
                """

        if vocab is not None and vocab.has_specials:
            error_msg = "Vocab contains special symbols." \
                        " Vocabs with special symbols cannot be used" \
                        " with multilabel fields."
            _LOGGER.error(error_msg)
            raise ValueError(error_msg)

        self.num_of_classes = num_of_classes
        super().__init__(name,
                         vocab=vocab,
                         eager=eager,
                         custom_numericalize=custom_numericalize,
                         is_target=True,
                         fixed_length=num_of_classes,
                         allow_missing_data=allow_missing_data)

    def finalize(self):
        super().finalize()
        if self.num_of_classes is None:
            self.fixed_length = self.num_of_classes = len(self.vocab)

        if self.use_vocab and len(self.vocab) > self.num_of_classes:
            error_msg = "Number of classes in data is greater than the declared number " \
                        "of classes. Declared: {}, Actual: {}" \
                .format(self.num_of_classes, len(self.vocab))
            _LOGGER.error(error_msg)
            raise ValueError(error_msg)

    def _numericalize_tokens(self, tokens):
        if self.use_vocab:
            token_numericalize = self.vocab.stoi.get

        else:
            token_numericalize = self.custom_numericalize

        return numericalize_multihot(tokens, token_numericalize, self.num_of_classes)


def numericalize_multihot(tokens, token_indexer, num_of_classes):
    active_classes = list(map(token_indexer, tokens))
    multihot_encoding = np.zeros(num_of_classes, dtype=np.bool)
    multihot_encoding[active_classes] = 1
    return multihot_encoding


def unpack_fields(fields):
    """Flattens the given fields object into a flat list of fields.

    Parameters
    ----------
    fields : (list | dict)
        List or dict that can contain nested tuples and None as values and
        column names as keys (dict).

    Returns
    -------
    list[Field]
        A flat list of Fields found in the given 'fields' object.
    """

    unpacked_fields = list()

    fields = fields.values() if isinstance(fields, dict) else fields

    # None values represent columns that should be ignored
    for field in filter(lambda f: f is not None, fields):
        if isinstance(field, tuple):
            # Map fields to their output field lists
            output_fields = map(lambda f: f.get_output_fields(), field)

            # Flatten output fields to a flat list
            output_fields = itertools.chain.from_iterable(output_fields)

        else:
            output_fields = field.get_output_fields()

        unpacked_fields.extend(output_fields)

    return unpacked_fields

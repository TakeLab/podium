"""Module contains dataset's field definition and methods for construction."""
import logging
import itertools
from collections import deque

import numpy as np

from podium.storage.vocab import Vocab
from podium.preproc.tokenizers import get_tokenizer
from podium.util import log_and_raise_error

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
                 tokenizer='split'):
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
        # TODO rework MultioutputField
        self._tokenizer_arg = tokenizer
        self.pretokenization_pipeline = PretokenizationPipeline()
        self.tokenizer = get_tokenizer(tokenizer)
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
                 keep_raw=False,
                 numericalizer=None,  # TODO Better arg name?
                 is_target=False,
                 fixed_length=None,
                 allow_missing_data=False,
                 disable_batch_matrix=False,
                 padding_token=-999,
                 missing_data_token=-1,
                 pretokenize_hooks=(),
                 posttokenize_hooks=()
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
        keep_raw : bool
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

            If false, the output of the tokenizer is presumed to be a custom datatype.
            Posttokenization hooks aren't allowed to be added as they can't be called
            on custom datatypes. For non-numericalizable fields, Iterator will generate
            batch fields containing lists of these custom data type instances returned
            by the tokenizer.
        custom_numericalize : callable
            The numericalization function that will be called if the field
            doesn't use a vocabulary. If using custom_numericalize and padding is
            required, please ensure that the `missing_data_token` is of the same type
            as the value returned by custom_numericalize.
        batch_as_matrix: bool
            Whether the batch created for this field will be compressed into a matrix.
            This parameter is ignored if is_numericalizable is set to False.
            If True, the batch returned by an Iterator or Dataset.batch() will contain
            a matrix of numericalizations for all examples.
            If False, a list of unpadded vectors will be returned instead. For missing
            data, the value in the list will be None.
        padding_token : int
            If custom_numericalize is provided and padding the batch matrix is needed,
            this token is used to pad the end of the matrix row.
            If custom_numericalize is None, this is ignored.
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
            its matrix row will be filled with this value. For non-numericalizable fields,
            this parameter is ignored and the value will be None.
            If using custom_numericalize and padding is required, please ensure that
            the `missing_data_token` is of the same type as the value returned by
            custom_numericalize.
            Default: -1

        Raises
        ------
        ValueError
            If the given tokenizer is not a callable or a string, or is a
            string that doesn't correspond to any of the premade tokenizers.
        """

        self.name = name
        self.disable_batch_matrix = disable_batch_matrix
        self._tokenizer_arg_string = tokenizer if isinstance(tokenizer, str) else None

        if tokenizer is None:
            self.tokenizer = None
        else:
            self.tokenizer = get_tokenizer(tokenizer)

        if isinstance(numericalizer, Vocab):
            self.vocab = numericalizer
            self.numericalizer = self.vocab.__getitem__
        else:
            self.vocab = None
            self.numericalizer = numericalizer

        self.keep_raw = keep_raw
        self.padding_token = padding_token
        self.is_target = is_target
        self.fixed_length = fixed_length
        self.allow_missing_data = allow_missing_data
        self.missing_data_token = missing_data_token
        self.pretokenize_pipeline = PretokenizationPipeline()
        self.posttokenize_pipeline = PosttokenizationPipeline()

        if pretokenize_hooks is not None:
            if not isinstance(pretokenize_hooks, (list, tuple)):
                pretokenize_hooks = (pretokenize_hooks,)
            for hook in pretokenize_hooks:
                self.add_pretokenize_hook(hook)

        if posttokenize_hooks is not None:
            if not isinstance(posttokenize_hooks, (list, tuple)):
                posttokenize_hooks = (posttokenize_hooks,)
            for hook in posttokenize_hooks:
                self.add_posttokenize_hook(hook)

    @property
    def eager(self):
        return self.vocab is not None and self.vocab.eager

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

        Raises
        ------
            If field is declared as non numericalizable.
        """
        if not self.is_numericalizable:
            error_msg = "Field is declared as non numericalizable. Posttokenization " \
                        "hooks aren't used in such fields."
            log_and_raise_error(ValueError, _LOGGER, error_msg)

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
        updating the vocab if the vocab is eager and preserving the raw data
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

        Raises
        ------
            If data is None and missing data is not allowed.
        """

        if data is None:
            if not self.allow_missing_data:
                error_msg = f"Missing data not allowed in field {self.name}"
                log_and_raise_error(ValueError, _LOGGER, error_msg)

            else:
                return (self.name, (None, None)),

        # Preprocess the raw input
        # TODO keep unprocessed or processed raw?
        processed_raw = self._run_pretokenization_hooks(data)
        tokenized = self.tokenizer(processed_raw) if self.tokenizer is not None \
            else processed_raw

        return self._process_tokens(processed_raw, tokenized),

    def _process_tokens(self, raw, tokens):
        """Runs posttokenization processing on the provided data and tokens and updates
        the vocab if needed. Used by Multioutput field.

        Parameters
        ----------
        data
            data processed by Pretokenization hooks

        tokens : list
            tokenized data

        Returns
        -------
        name, (data, tokens)
            Returns and tuple containing this both field's name and a tuple containing
            the data and tokens processed by posttokenization hooks.
        """

        raw, tokenized = self._run_posttokenization_hooks(raw, tokens)
        raw = raw if self.keep_raw else None

        if self.eager and not self.vocab.finalized:
            self.update_vocab(tokenized)
        return self.name, (raw, tokenized)

    def update_vocab(self, tokenized):
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
            return  # TODO throw Error?

        data = tokenized if isinstance(tokenized, (list, tuple)) else (tokenized,)
        self.vocab += data

    @property
    def finalized(self) -> bool:
        """Returns whether the field's Vocab vas finalized. If the field has no
        vocab, returns True.

        Returns
        -------
        bool
            Whether the field's Vocab vas finalized. If the field has no
            vocab, returns True.
        """
        return True if self.vocab is None else self.vocab.finalized

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
        _, tokenized = data

        if tokenized is None:
            if not self.allow_missing_data:
                error_msg = f"Missing value found in field {self.name}."
                log_and_raise_error(ValueError, _LOGGER, error_msg)

            else:
                return None

        if self.numericalizer is None:
            # data can not be numericalized, return tokenized as-is
            return tokenized

        tokens = tokenized if isinstance(tokenized, (list, tuple)) else [tokenized]

        if self.use_vocab:
            return self.vocab.numericalize(tokens)
        else:
            return np.array([self.numericalizer(t) for t in tokens])

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
            error_msg = f"Missing data not allowed in field {self.name}"
            log_and_raise_error(ValueError, _LOGGER, error_msg)

        return self.missing_data_token

    def get_numericalization_for_example(self, example, cache=False):
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
        numericalization = getattr(example, cache_field_name, None)

        if numericalization is None:
            example_data = getattr(example, self.name)
            numericalization = self.numericalize(example_data)
            if cache:
                setattr(example, cache_field_name, numericalization)

        return numericalization

    def to_batch(self, examples):
        numericalizations = []

        for example in examples:
            numericalization = self.get_numericalization_for_example(example)
            numericalizations.append(numericalization)

        # casting to matrix can only be attempted if all values are either
        # None or np.ndarray
        possible_cast_to_matrix = not any(
            x is not None and not isinstance(x, (np.ndarray, int, float))
            for x in numericalizations
        )
        if len(numericalizations) > 0 \
                and not self.disable_batch_matrix \
                and possible_cast_to_matrix:
            return self._arrays_to_matrix(numericalizations)

        else:
            return numericalizations

    def _arrays_to_matrix(self, arrays):
        pad_length = self._get_pad_length(arrays)
        padded_arrays = [self._pad_to_length(a, pad_length) for a in arrays]
        return np.array(padded_arrays)

    def _get_pad_length(self, numericalizations):
        # the fixed_length attribute of Field has priority over the max length
        # of all the examples in the batch
        if self.fixed_length is not None:
            return self.fixed_length

        # if fixed_length is None, then return the maximum length of all the
        # examples in the batch
        def num_length(n): return 1 if n is None else len(n)

        return max(map(num_length, numericalizations))

    def _pad_to_length(self, array, length, custom_pad_symbol=None,
                       pad_left=False, truncate_left=False):
        """Either pads the given row with pad symbols, or truncates the row
        to be of given length. The vocab provides the pad symbol for all
        fields that have vocabs, otherwise the pad symbol has to be given as
        a parameter.

        Parameters
        ----------
        array : np.ndarray
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
        if array is None:
            return np.full(shape=length, fill_value=self.missing_data_token)

        if isinstance(array, (int, float)):
            array = np.array([array])

        if len(array) > length:
            # truncating

            if truncate_left:
                array = array[len(array) - length:]
            else:
                array = array[:length]

        elif len(array) < length:
            # padding

            if custom_pad_symbol is not None:
                pad_symbol = custom_pad_symbol

            elif self.use_vocab:
                pad_symbol = self.vocab.padding_index()

            else:
                pad_symbol = self.padding_token

            if pad_symbol is None:
                error_msg = 'Must provide a custom pad symbol if the ' \
                            'field has no vocab.'
                log_and_raise_error(ValueError, _LOGGER, error_msg)

            diff = length - len(array)

            if pad_left:
                array = np.pad(array, (diff, 0), 'constant',
                               constant_values=pad_symbol)
            else:
                array = np.pad(array, (0, diff), 'constant',
                               constant_values=pad_symbol)

        return array

    def __getstate__(self):
        """Method obtains field state. It is used for pickling dataset data
        to file.

        Returns
        -------
        state : dict
            dataset state dictionary
        """
        state = self.__dict__.copy()
        if self._tokenizer_arg_string is not None:
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
        if self._tokenizer_arg_string is not None:
            self.tokenizer = get_tokenizer(self._tokenizer_arg_string)

    def __repr__(self):
        if self.use_vocab:
            return "{}[name: {}, is_target: {}, vocab: {}]".format(
                self.__class__.__name__, self.name, self.is_target,
                self.vocab)
        else:
            return "{}[name: {}, is_target: {}]".format(
                self.__class__.__name__, self.name, self.is_target)

    def get_output_fields(self):
        """Returns an Iterable of the contained output fields.

        Returns
        -------
        Iterable :
            an Iterable of the contained output fields.
        """
        return self,


class LabelField(Field):
    def __init__(self,
                 name,
                 numericalizer=None,
                 allow_missing_data=False,
                 is_target=True,
                 missing_data_token=-1,
                 label_processing_hooks=()
                 ):
        if numericalizer is None:
            # Default to a vocabulary if custom numericalize is not set
            numericalizer = Vocab(specials=())

        if isinstance(numericalizer, Vocab) and numericalizer.has_specials:
            error_msg = "Vocab contains special symbols." \
                        " Vocabs with special symbols cannot be used" \
                        " with LabelFields."
            log_and_raise_error(ValueError, _LOGGER, error_msg)

        super().__init__(name,
                         tokenizer=None,
                         keep_raw=False,
                         numericalizer=numericalizer,
                         is_target=is_target,
                         fixed_length=1,
                         allow_missing_data=True,
                         missing_data_token=missing_data_token,
                         pretokenize_hooks=label_processing_hooks
                         )


# class TokenizedField(Field):
#     """Tokenized version of the Field. Holds the preprocessing and
#     numericalization logic for the pre-tokenized dataset fields.
#     """
#
#     def __init__(self,
#                  name,
#                  vocab=None,
#                  eager=True,
#                  custom_numericalize=None,
#                  batch_as_matrix=True,
#                  padding_token=-999,
#                  is_target=False,
#                  fixed_length=None,
#                  allow_missing_data=False,
#                  missing_data_token=-1):
#         super().__init__(
#             name=name,
#             vocab=vocab,
#             keep_raw=False,
#             tokenize=False,
#             store_as_tokenized=True,
#             eager=eager,
#             custom_numericalize=custom_numericalize,
#             batch_as_matrix=batch_as_matrix,
#             padding_token=padding_token,
#             is_target=is_target,
#             fixed_length=fixed_length,
#             allow_missing_data=allow_missing_data,
#             missing_data_token=missing_data_token
#         )


class MultilabelField(Field):
    """Class used for storing pre-tokenized labels.
    Used for multilabeled datasets.
    """

    def __init__(self,
                 name,
                 tokenizer=None,
                 numericalizer=None,
                 num_of_classes=None,
                 is_target=True,
                 allow_missing_data=False,
                 missing_data_token=-1,
                 pretokenize_hooks=(),
                 posttokenize_hooks=()):
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

        custom_numericalize : callable(str) -> int
            Callable that takes a string and returns an int.
            Used to index classes.

        batch_as_matrix: bool
            Whether the batch created for this field will be compressed into a
            matrix. This parameter is ignored if is_numericalizable is set to
            False.
            If True, the batch returned by an Iterator or Dataset.batch() will
            contain a matrix of numericalizations for all examples.
            If False, a list of unpadded vectors will be returned instead.
            For missing data, the value in the list will be None.

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

        missing_data_token : number
            Token to use to mark batch rows as missing. If data for a field is
            missing, its matrix row will be filled with this value.
            For non-numericalizable fields, this parameter is ignored and the
            value will be None. If using custom_numericalize and padding is
            required, please ensure that the `missing_data_token` is of the same
            type as the value returned by custom_numericalize.
            Default: -1

        Raises
        ------
        ValueError
            If the provided Vocab contains special symbols.
        """

        if numericalizer is None:
            numericalizer = Vocab(specials=())

        if isinstance(numericalizer, Vocab) and numericalizer.has_specials:
            error_msg = "Vocab contains special symbols." \
                        " Vocabs with special symbols cannot be used" \
                        " with MultilabelFields."
            log_and_raise_error(ValueError, _LOGGER, error_msg)

        self.num_of_classes = num_of_classes
        super().__init__(name,
                         tokenizer=tokenizer,
                         keep_raw=False,
                         numericalizer=numericalizer,
                         is_target=is_target,
                         fixed_length=num_of_classes,
                         allow_missing_data=allow_missing_data,
                         missing_data_token=missing_data_token,
                         pretokenize_hooks=pretokenize_hooks,
                         posttokenize_hooks=posttokenize_hooks
                         )

    def finalize(self):
        super().finalize()
        if self.num_of_classes is None:
            self.fixed_length = self.num_of_classes = len(self.vocab)

        if self.use_vocab and len(self.vocab) > self.num_of_classes:
            error_msg = "Number of classes in data is greater than the declared number " \
                        f"of classes. Declared: {self.num_of_classes}, " \
                        f"Actual: {len(self.vocab)}"
            log_and_raise_error(ValueError, _LOGGER, error_msg)

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
        _, tokenized = data

        if tokenized is None:
            if not self.allow_missing_data:
                error_msg = f"Missing value found in field {self.name}."
                log_and_raise_error(ValueError, _LOGGER, error_msg)

            else:
                return None

        if self.numericalizer is None:
            # data can not be numericalized, return tokenized as-is
            return tokenized

        tokens = tokenized if isinstance(tokenized, (list, tuple)) else [tokenized]

        if self.use_vocab:
            active_classes = self.vocab.numericalize(tokens)
        else:
            active_classes = np.array([self.numericalizer(t) for t in tokens])

        multihot = np.full(shape=self.num_of_asses, fill_value=0, dtype=np.int)
        multihot[active_classes] = 1
        return multihot


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

    unpacked_fields = []

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

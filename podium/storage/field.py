"""Module contains dataset's field definition and methods for construction."""
import logging
import itertools
from collections import deque
from collections.abc import Iterator
from typing import Optional, List, Tuple, Any, Union, Callable, Iterable

import numpy as np

from podium.preproc.tokenizers import get_tokenizer
from podium.storage.vocab import Vocab
from podium.util import log_and_raise_error

_LOGGER = logging.getLogger(__name__)

# Type definitions to make type hints more readable
pretok_hook_type = Callable[[Any], Any]
posttok_hook_type = Callable[[Any, List[str]], Tuple[Any, List[str]]]
tokenizer_type = Callable[[Any], List[str]]
tokenizer_arg_type = Optional[Union[str, tokenizer_type]]
numericalizer_type = Callable[[str], Union[int, float]]


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

        if isinstance(processed_tokenized, Iterator):
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
                 output_fields: List["Field"],
                 tokenizer: tokenizer_arg_type = 'split'):
        """Field that does pretokenization and tokenization once and passes it to its
        output fields. Output fields are any type of field. The output fields are used
        only for posttokenization processing (posttokenization hooks and vocab updating).

        Parameters
        ----------
         output_fields : List[Field],
            List containig the output fields. The pretokenization hooks and tokenizer
            in these fields are ignored and only posttokenization hooks are used.
         tokenizer : Optional[Union[str, Callable]]
            The tokenizer that is to be used when preprocessing raw data
            (only if 'tokenize' is True). The user can provide his own
            tokenizer as a callable object or specify one of the premade
            tokenizers by a string. The available premade tokenizers are:
                - 'split' - default str.split()
                - 'spacy-lang' - the spacy tokenizer. The language model can be defined
                by replacing `lang` with the language model name. For example `spacy-en`
        """

        self._tokenizer_arg = tokenizer
        self.pretokenization_pipeline = PretokenizationPipeline()
        self.tokenizer = get_tokenizer(tokenizer)
        self.output_fields = deque(output_fields)

    def add_pretokenize_hook(self, hook: pretok_hook_type):
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
        hook : Callable[[Any], Any]
            The pre-tokenization hook that we want to add to the field.
        """
        self.pretokenization_pipeline.add_hook(hook)

    def _run_pretokenization_hooks(self, data: Any) -> Any:
        """Runs pretokenization hooks on the raw data and returns the result.

        Parameters
        ----------
        data : Any
            data to be processed

        Returns
        -------
        Any
            processed data

        """

        return self.pretokenization_pipeline(data)

    def add_output_field(self, field: "Field"):
        """
        Adds the passed field to this field's output fields.

        Parameters
        ----------
        field : Field
            Field to add to output fields.
        """
        self.output_fields.append(field)

    def preprocess(self, data: Any) -> Iterable[Tuple[str, Tuple[Optional[Any], Any]]]:
        """Preprocesses raw data, tokenizing it if required. The outputfields update their
         vocabs if required and preserve the raw data if the output field's
         'keep_raw' is true.

        Parameters
        ----------
        data : Any
            The raw data that needs to be preprocessed.

        Returns
        -------
        Iterable[Tuple[str, Tuple[Optional[Any], Any]]]
            An Iterable containing the raw and tokenized data of all the output fields.
            The structure of the returned tuples is (name, (raw, tokenized)), where 'name'
            is the name of the output field and raw and tokenized are processed data.

        Raises
        ------
            If data is None and missing data is not allowed.
        """
        data = self._run_pretokenization_hooks(data)
        tokens = self.tokenizer(data) if self.tokenizer is not None \
            else data
        return tuple(field._process_tokens(data, tokens) for field in self.output_fields)

    def get_output_fields(self) -> Iterable["Field"]:
        """
        Returns an Iterable of the contained output fields.

        Returns
        -------
        Iterable[Field] :
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
                 name: str,
                 tokenizer: tokenizer_arg_type = 'split',
                 keep_raw: bool = False,
                 # TODO Better arg name?
                 numericalizer: Optional[Union[Vocab, numericalizer_type]] = None,
                 is_target: bool = False,
                 fixed_length: Optional[int] = None,
                 allow_missing_data: bool = False,
                 disable_batch_matrix: bool = False,
                 padding_token: Union[int, float] = -999,
                 missing_data_token: Union[int, float] = -1,
                 pretokenize_hooks: Iterable[pretok_hook_type] = (),
                 posttokenize_hooks: Iterable[posttok_hook_type] = ()
                 ):
        """Create a Field from arguments.

        Parameters
        ----------
        name : str
            Field name, used for referencing data in the dataset.

        tokenizer : str | callable
            The tokenizer that is to be used when preprocessing raw data.
            The user can provide his own tokenizer as a callable object or specify one of
            the registered tokenizers by a string. The available pre-registered tokenizers
             are:
                - 'split' - default str.split(). Custom separator can be provided as
                `split-sep` where `sep` is the separator string.
                - 'spacy-lang' - the spacy tokenizer. The language model can be defined
                by replacing `lang` with the language model name. For example `spacy-en`.
            If None, the data will not be tokenized and post-tokenization hooks wont be
            called. The provided data will be stored in the `tokenized` data field as-is.

        keep_raw : bool
            Whether to store untokenized preprocessed data.
            If True, the raw data will be run trough the provided pretokenize
            hooks and stored in the 'raw' part of the example tuple.

        numericalizer : callable
            Object used to numericalize tokens.
            Can either be a Vocab, a custom numericalization callable or None.
            If it's a Vocab, this field will update it after preprocessing (or during
            finalization if eager is False) and use it to numericalize data. Also, the
            Vocab's padding token will be used instead of the Field's.
            If it's a Callable, It will be used to numericalize data token by token.
            If None, numericalization won't be attempted and batches will be created as
            lists instead of numpy matrices.

         is_target : bool
            Whether this field is a target variable. Affects iteration over
            batches.

        fixed_length : int, optional
            To which length should the field be fixed. If it is not None every
            example in the field will be truncated or padded to given length
            during batching. If the batched data is not a vector, this parameter is
            ignored.

        allow_missing_data : bool
            Whether the field allows missing data. In the case 'allow_missing_data'
            is False and None is sent to be preprocessed, an ValueError will be raised.
            If 'allow_missing_data' is True, if a None is sent to be preprocessed, it will
            be stored and later numericalized properly.

        disable_batch_matrix: bool
            Whether the batch created for this field will be compressed into a matrix.
            If False, the batch returned by an Iterator or Dataset.batch() will contain
            a matrix of numericalizations for all examples (if possible).
            If True, a list of unpadded vectors(or other data type) will be returned
            instead. For missing data, the value in the list will be None.

        padding_token : int
            Padding token used when numericalizer is a callable. If the numericalizer is
            None or a Vocab, this value is ignored.

        missing_data_token : Union[int, float]
            Token to use to mark batch rows as missing. If data for a field is missing,
            its matrix row will be filled with this value. For non-numericalizable fields,
            this parameter is ignored and the value will be None.

        pretokenize_hooks: Iterable[Callable[[Any], Any]]
            Iterable containing pretokenization hooks. Providing hooks in this way is
            identical to calling `add_pretokenize_hook`.

        posttokenize_hooks: Iterable[Callable[[Any, List[str]], Tuple[Any, List[str]]]]
            Iterable containing posttokenization hooks. Providing hooks in this way is
            identical to calling `add_posttokenize_hook`.

        Raises
        ------
        ValueError
            If the given tokenizer is not a callable or a string, or is a
            string that doesn't correspond to any of the registered tokenizers.
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
        self.pretokenize_pipeline = PretokenizationPipeline()
        self.posttokenize_pipeline = PosttokenizationPipeline()
        self.allow_missing_data = allow_missing_data
        self.missing_data_token = missing_data_token

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
        """A flag that tells whether this field has a Vocab and whether that Vocab is
        marked as eager.

        Returns
        -------
        bool
            whether this field has a Vocab and whether that Vocab is
            marked as eager
        """
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

    def add_pretokenize_hook(self, hook: pretok_hook_type):
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

    def add_posttokenize_hook(self, hook: posttok_hook_type):
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
        self.posttokenize_pipeline.add_hook(hook)

    def remove_pretokenize_hooks(self):
        """Remove all the pre-tokenization hooks that were added to the Field.
        """
        self.pretokenize_pipeline.clear()

    def remove_posttokenize_hooks(self):
        """Remove all the post-tokenization hooks that were added to the Field.
        """
        self.posttokenize_pipeline.clear()

    def _run_pretokenization_hooks(self, data: Any) -> Any:
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

    def _run_posttokenization_hooks(self, data: Any, tokens: List[str]) \
            -> Tuple[Any, List[str]]:
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

    def preprocess(self, data: Any) \
            -> Iterable[Tuple[str, Tuple[Any, Optional[List[str]]]]]:
        """Preprocesses raw data, tokenizing it if required,
        updating the vocab if the vocab is eager and preserving the raw data
        if field's 'store_raw' is true.

        Parameters
        ----------
        data : str or iterable(hashable)
            The raw data that needs to be preprocessed.

        Returns
        -------
        ((str, Iterable(hashable)), )
            A tuple containing one tuple of the format (field_name, (raw, tokenized)).
            Raw is set to None if `keep_raw` is disabled.
            Both raw and tokenized will be set to none if None is passed as `data` and
            `allow_missing_data` is enabled.

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

    def update_vocab(self, tokenized: List[str]):
        """Updates the vocab with a data point in its tokenized form.
        If the field does not do tokenization,

        Parameters
        ----------
        tokenized : Union[Any, List(str)]
            The tokenized form of the data point that the vocab is to be
            updated with.
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

    def _process_tokens(self, raw: Any, tokens: Union[Any, List[str]]) \
            -> Tuple[str, Tuple[Any, Optional[Any, List[str]]]]:
        """Runs posttokenization processing on the provided data and tokens and updates
        the vocab if needed. Used by Multioutput field.

        Parameters
        ----------
        raw: Any
            data processed by Pretokenization hooks

        tokens : List[str]
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

    def get_default_value(self) -> Union[int, float]:
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

    def numericalize(self, data: Tuple[Optional[Any], Optional[Union[Any, List[str]]]]) \
            -> Optional[Any, np.ndarray]:
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

    def pad_to_length(self, array: np.array, length: int,
                      custom_pad_symbol: Optional[Union[int, float]] = None,
                      pad_left: bool = False, truncate_left: bool = False):
        """Either pads the given row with pad symbols, or truncates the row
        to be of given length. The vocab provides the pad symbol for all
        fields that have vocabs, otherwise the pad symbol has to be given as
        a parameter.

        Parameters
        ----------
        array : numpy.ndarray
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

        Returns
        -------
        numpy.ndarray
            Numpy array padded or truncated to `length`.

        Raises
        ------
        ValueError
            If the field doesn't use a vocab and no custom pad symbol was
            given.
        """
        if array is None:
            return np.full(shape=length, fill_value=self.get_default_value())

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

    def get_numericalization_for_example(self, example: 'Example', cache: bool = True) \
            -> Optional[Union[Any, np.ndarray]]:
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
        Union[numpy.ndarray, Any]
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

    def get_output_fields(self) -> Iterable["Field"]:
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
                         allow_missing_data=allow_missing_data,
                         missing_data_token=missing_data_token,
                         pretokenize_hooks=label_processing_hooks
                         )


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
        active_classes = super(MultilabelField, self).numericalize(data)

        multihot = np.zeros(shape=self.num_of_classes, dtype=np.int)
        if len(active_classes) > 0:
            multihot[active_classes] = 1
        return multihot


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

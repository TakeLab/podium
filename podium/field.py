"""
Module contains dataset's field definition and methods for construction.
"""
import itertools
import textwrap
from collections import deque
from collections.abc import Iterator
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

import numpy as np

from podium.preproc.hooks import HookType
from podium.preproc.tokenizers import get_tokenizer
from podium.utils.general_utils import repr_type_and_attrs
from podium.vocab import Vocab


PretokenizationHookType = Callable[[Any], Any]
PosttokenizationHookType = Callable[[Any, List[str]], Tuple[Any, List[str]]]
TokenizerType = Optional[Union[str, Callable[[Any], List[str]]]]
NumericalizerType = Callable[[str], Union[int, float]]


class _PretokenizationPipeline:
    def __init__(self, hooks=()):
        self.hooks = deque(hooks)

    def add_hook(self, hook: PretokenizationHookType):
        hook_type = getattr(hook, "__hook_type__", HookType.PRETOKENIZE)
        if hook_type != HookType.PRETOKENIZE:
            raise ValueError("Hook is not a pretokenization hook")
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


class _PosttokenizationPipeline:
    def __init__(self, hooks=()):
        self.hooks = deque(hooks)

    def add_hook(self, hook: PosttokenizationHookType):
        hook_type = getattr(hook, "__hook_type__", HookType.POSTTOKENIZE)
        if hook_type != HookType.POSTTOKENIZE:
            raise ValueError(
                "Hook is not a posttokenization hook. Please wrap your hook with the `podium.preproc.as_posttokenize_hook` function to turn it into a posttokenization hook."
            )
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


class Field:
    """
    Holds the preprocessing and numericalization logic for a single field of a
    dataset.
    """

    def __init__(
        self,
        name: str,
        tokenizer: TokenizerType = "split",
        keep_raw: bool = False,
        numericalizer: Optional[Union[Vocab, NumericalizerType]] = None,
        is_target: bool = False,
        include_lengths: bool = False,
        fixed_length: Optional[int] = None,
        allow_missing_data: bool = False,
        disable_batch_matrix: bool = False,
        disable_numericalize_caching: bool = False,
        padding_token: Union[int, float] = -999,
        missing_data_token: Union[int, float] = -1,
        pretokenize_hooks: Optional[Iterable[PretokenizationHookType]] = None,
        posttokenize_hooks: Optional[Iterable[PosttokenizationHookType]] = None,
    ):
        """
        Create a Field from arguments.

        Parameters
        ----------
        name : str
            Field name, used for referencing data in the dataset.

        tokenizer : str | callable | optional
            The tokenizer used when preprocessing raw data.

            Users can provide their own tokenizers as a ``callable`` or specify one of
            the registered tokenizers by passing a string keyword.
            Available pre-registered tokenizers are:

            - 'split' - default, ``str.split()``. A custom separator can be provided as
              ``split-sep`` where ``sep`` is the separator string.
            - 'spacy-lang' - the spacy tokenizer. The language model can be defined
              by replacing ``lang`` with the language model name.
              Example ``spacy-en_core_web_sm``

            If ``None``, the data will not be tokenized. Raw input data will be
            stored as ``tokenized``.

        keep_raw : bool
            Flag determining whether to store raw data.

            If ``True``, raw data will be stored in the 'raw' part of the Example tuple.

        numericalizer : callable
            Object used to numericalize tokens.

            Can be a ``Vocab``, a custom numericalizer callable or ``None``.
            If the numericalizer is a ``Vocab`` instance, Vocab's padding token will
            be used instead of the Field's.
            If a ``Callable`` is passed, it will be used to numericalize data token by token.
            If ``None``, numericalization won't be attempted for this Fieldand batches will
            be created as lists instead of numpy matrices.

        is_target : bool
            Flag indicating whether this field contains a target (response) variable.

            Affects iteration over batches by separating target and non-target Fields.

        include_lengths : bool
            Flag indicating whether the batch representation of this field should
            include the length of every instance in the batch.

            If ``True``, the batch element under the name
            of this Field will be a tuple of ``(numericalized values, lengths)``.

        fixed_length : int, optional
            Number indicating to which length should the field length be fixed.
            If set, every example in the field will be truncated or padded to the given length
            during batching. If the batched data is not a sequence, this parameter is
            ignored.
            If ``None``, the batch data will be padded to the length of the longest instance
            in each minibatch.

        allow_missing_data : bool
            A flag determining if the Field allows missing data. If
            ``allow_missing_data=False`` and a ``None`` value is present in the raw data,
            a ``ValueError`` will be raised.
            If ``allow_missing_data=True``, and a ``None`` value is present in the raw data,
            it will be stored and numericalized properly.
            Defaults to ``False``.

        disable_batch_matrix: bool
            Flag indicating whether the data contained in this Field should be packed into
            a matrix.
            If ``False``, the batch returned by an ``Iterator`` or ``Dataset.batch()`` will
            contain a padded matrix of numericalizations for all examples.
            If ``True``, a list of unpadded numericalizations will be returned
            instead. For missing data, the value in the list will be None.
            Defaults to ``False``.

        disable_numericalize_caching : bool
            Flag which determines whether the numericalization of this field should be
            cached.

            Should be set to ``True`` if the numericalization can differ
            between ``numericalize`` function calls for the same instance.
            When set to ``False``, the numericalization values will be cached upon
            first computation and reused in each subsqeuent time.

            The flag is passed to the numericalizer to indicate
            use of its nondeterministic setting. This flag is mainly intended be used in the
            case of masked language modelling, where we wish the inputs to be masked
            (nondeterministic), and the outputs (labels) to not be masked while using the
            same Vocab.
            Defaults to ``False``.

        padding_token : int
            Padding token used when numericalizer is a callable. If the numericalizer is
            ``None`` or a ``Vocab`` instance, this value is ignored.
            Defaults to ``-999``.

        missing_data_token : Union[int, float]
            Token used to mark instance data as missing. For non-numericalizable fields,
            this parameter is ignored and their value will be ``None``.
            Defaults to ``-1``.

        pretokenize_hooks: Iterable[Callable[[Any], Any]]
            Iterable containing pretokenization hooks. Providing hooks in this way is
            identical to calling ``add_pretokenize_hook``.

        posttokenize_hooks: Iterable[Callable[[Any, List[str]], Tuple[Any, List[str]]]]
            Iterable containing posttokenization hooks. Providing hooks in this way is
            identical to calling ``add_posttokenize_hook``.

        Raises
        ------
        ValueError
            If the given tokenizer is not a callable or a string, or is a
            string that doesn't correspond to any of the registered tokenizers.
        """

        if not isinstance(name, str):
            raise ValueError(
                f"Name must be a string," f" got type '{type(name).__name__}' instead."
            )
        self._name = name
        self._disable_batch_matrix = disable_batch_matrix
        self._disable_numericalize_caching = disable_numericalize_caching
        self._tokenizer_arg_string = tokenizer if isinstance(tokenizer, str) else None

        if tokenizer is None:
            self._tokenizer = None
        else:
            self._tokenizer = get_tokenizer(tokenizer)

        if isinstance(numericalizer, Vocab):
            self._vocab = numericalizer
            self._numericalizer = self.vocab.__getitem__
        else:
            self._vocab = None
            self._numericalizer = numericalizer

        self._keep_raw = keep_raw

        if not isinstance(padding_token, (int, float)):
            raise ValueError(
                f"Padding token of Field '{name}' is of type"
                f" '{type(padding_token).__name__}'. Must be int or float"
            )
        self._padding_token = padding_token

        self._is_target = is_target

        # TODO: @mttk perform a sanity check here (if fixed length is set etc etc)
        self._include_lengths = include_lengths

        if fixed_length is not None and not isinstance(fixed_length, int):
            raise ValueError(
                f"`fixed_length` of Field `{name}` is of type"
                f" {type(fixed_length).__name__}. Must be None or int."
            )
        self._fixed_length = fixed_length

        self._pretokenize_pipeline = _PretokenizationPipeline()
        self._posttokenize_pipeline = _PosttokenizationPipeline()
        self._allow_missing_data = allow_missing_data

        if not isinstance(missing_data_token, (int, float)):
            raise ValueError(
                f"Missing data token of Field '{name}' is of type"
                f" '{type(missing_data_token).__name__}'. Must be int or float."
            )
        self._missing_data_token = missing_data_token

        if pretokenize_hooks is not None:
            if not isinstance(pretokenize_hooks, (list, tuple)):
                pretokenize_hooks = [pretokenize_hooks]
            for hook in pretokenize_hooks:
                self.add_pretokenize_hook(hook)

        if posttokenize_hooks is not None:
            if not isinstance(posttokenize_hooks, (list, tuple)):
                posttokenize_hooks = [posttokenize_hooks]
            for hook in posttokenize_hooks:
                self.add_posttokenize_hook(hook)

    @property
    def name(self):
        """
        The name of this field.
        """
        return self._name

    @property
    def eager(self):
        """
        A flag that tells whether this field has a Vocab and whether that Vocab
        is marked as eager.

        Returns
        -------
        bool
            Whether this field has a Vocab and whether that Vocab is
            marked as eager
        """
        return self.vocab is not None and self.vocab.eager

    @property
    def vocab(self):
        """
        The field's Vocab or None.

        Returns
        -------
        Vocab, optional
            Returns the field's Vocab if defined or None.
        """
        return self._vocab

    @property
    def disable_numericalize_caching(self):
        return self._disable_numericalize_caching

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

    @property
    def is_target(self):
        return self._is_target

    @property
    def include_lengths(self):
        return self._include_lengths

    def add_pretokenize_hook(self, hook: PretokenizationHookType):
        """
        Add a pre-tokenization hook to the Field. If multiple hooks are added to
        the field, the order of their execution will be the same as the order in
        which they were added to the field, each subsequent hook taking the
        output of the previous hook as its input. If the same function is added
        to the Field as a hook multiple times, it will be executed that many
        times. The output of the final pre-tokenization hook is the raw data
        that the tokenizer will get as its input.

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
        self._pretokenize_pipeline.add_hook(hook)

    def add_posttokenize_hook(self, hook: PosttokenizationHookType):
        """
        Add a post-tokenization hook to the Field. If multiple hooks are added
        to the field, the order of their execution will be the same as the order
        in which they were added to the field, each subsequent hook taking the
        output of the previous hook as its input. If the same function is added
        to the Field as a hook multiple times, it will be executed that many
        times. Post-tokenization hooks are called only if the Field is
        sequential (in non-sequential fields there is no tokenization and only
        pre-tokenization hooks are called). The output of the final post-
        tokenization hook are the raw and tokenized data that the preprocess
        function will use to produce its result.

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
        self._posttokenize_pipeline.add_hook(hook)

    def remove_pretokenize_hooks(self):
        """
        Remove all the pre-tokenization hooks that were added to the Field.
        """
        self._pretokenize_pipeline.clear()

    def remove_posttokenize_hooks(self):
        """
        Remove all the post-tokenization hooks that were added to the Field.
        """
        self._posttokenize_pipeline.clear()

    def _run_pretokenization_hooks(self, data: Any) -> Any:
        """
        Runs pretokenization hooks on the raw data and returns the result.

        Parameters
        ----------
        data : hashable
            data to be processed

        Returns
        -------
        hashable
            processed data
        """
        return self._pretokenize_pipeline(data)

    def _run_posttokenization_hooks(
        self, data: Any, tokens: List[str]
    ) -> Tuple[Any, List[str]]:
        """
        Runs posttokenization hooks on tokenized data.

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
        return self._posttokenize_pipeline(data, tokens)

    def preprocess(
        self, data: Any
    ) -> Iterable[Tuple[str, Tuple[Any, Optional[List[str]]]]]:
        """
        Preprocesses raw data, tokenizing it if required, updating the vocab if
        the vocab is eager and preserving the raw data if field's 'store_raw' is
        true.

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
            if not self._allow_missing_data:
                raise ValueError(f"Missing data not allowed in field {self.name}")

            else:
                return ((self.name, (None, None)),)

        # Preprocess the raw input
        # TODO keep unprocessed or processed raw?
        processed_raw = self._run_pretokenization_hooks(data)
        tokenized = (
            self._tokenizer(processed_raw)
            if self._tokenizer is not None
            else processed_raw
        )

        return (self._process_tokens(processed_raw, tokenized),)

    def update_vocab(self, tokenized: List[str]):
        """
        Updates the vocab with a data point in its tokenized form. If the field
        does not do tokenization,

        Parameters
        ----------
        tokenized : Union[Any, List(str)]
            The tokenized form of the data point that the vocab is to be
            updated with.
        """

        if not self.use_vocab:
            return  # TODO throw Error?

        data = tokenized if isinstance(tokenized, (list, tuple)) else (tokenized,)
        self._vocab += data

    @property
    def is_finalized(self) -> bool:
        """
        Returns whether the field's Vocab vas finalized. If the field has no
        vocab, returns True.

        Returns
        -------
        bool
            Whether the field's Vocab vas finalized. If the field has no
            vocab, returns True.
        """
        return True if self.vocab is None else self.vocab.is_finalized

    def finalize(self):
        """
        Signals that this field's vocab can be built.
        """

        if self.use_vocab:
            self.vocab.finalize()

    def _process_tokens(
        self, raw: Any, tokens: Union[Any, List[str]]
    ) -> Tuple[str, Tuple[Any, Optional[Union[Any, List[str]]]]]:
        """
        Runs posttokenization processing on the provided data and tokens and
        updates the vocab if needed. Used by Multioutput field.

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

        # Apply the special tokens. These act as a post-tokenization
        # hook, but are applied separately as we want to encapsulate
        # that logic in their class to minimize code changes.
        if self.use_vocab:
            for special_token in self.vocab.specials:
                tokenized = special_token.apply(tokenized)

        raw = raw if self._keep_raw else None

        # Self.eager checks if a vocab is used so this won't error
        if self.eager and not self.vocab.is_finalized:
            self.update_vocab(tokenized)
        return self.name, (raw, tokenized)

    def get_default_value(self) -> Union[int, float]:
        """
        Method obtains default field value for missing data.

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
        if not self._allow_missing_data:
            raise ValueError(f"Missing data not allowed in field {self.name}")

        return self._missing_data_token

    def numericalize(
        self, data: Tuple[Optional[Any], Optional[Union[Any, List[str]]]]
    ) -> Optional[Union[Any, np.ndarray]]:
        """
        Numericalize the already preprocessed data point based either on the
        vocab that was previously built, or on a custom numericalization
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
            if not self._allow_missing_data:
                raise ValueError(f"Missing value found in field {self.name}.")

            else:
                return None

        if self._numericalizer is None:
            # data can not be numericalized, return tokenized as-is
            return tokenized

        tokens = tokenized if isinstance(tokenized, (list, tuple)) else [tokenized]

        if self.use_vocab:
            return self.vocab.numericalize(tokens)
        else:
            return np.array([self._numericalizer(t) for t in tokens])

    def _pad_to_length(
        self,
        array: np.ndarray,
        length: int,
        custom_pad_symbol: Optional[Union[int, float]] = None,
        pad_left: bool = False,
        truncate_left: bool = False,
    ):
        """
        Either pads the given row with pad symbols, or truncates the row to be
        of given length. The vocab provides the pad symbol for all fields that
        have vocabs, otherwise the pad symbol has to be given as a parameter.

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
                array = array[len(array) - length :]
            else:
                array = array[:length]

        elif len(array) < length:
            # padding

            if custom_pad_symbol is not None:
                pad_symbol = custom_pad_symbol

            elif self.use_vocab:
                pad_symbol = self.vocab.get_padding_index()

            else:
                pad_symbol = self._padding_token

            if pad_symbol is None:
                raise ValueError(
                    "Must provide a custom pad symbol if the field has no vocab."
                )

            diff = length - len(array)

            if pad_left:
                array = np.pad(array, (diff, 0), "constant", constant_values=pad_symbol)
            else:
                array = np.pad(array, (0, diff), "constant", constant_values=pad_symbol)

        return array

    def get_numericalization_for_example(
        self, example, cache: bool = True
    ) -> Optional[Union[Any, np.ndarray]]:
        """
        Returns the numericalized data of this field for the provided example.
        The numericalized data is generated and cached in the example if 'cache'
        is true and the cached data is not already present. If already cached,
        the cached data is returned.

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
        cache_field_name = f"{self.name}_"
        numericalization = example.get(cache_field_name)

        # Check if this concrete field can be cached.

        cache = cache and not self.disable_numericalize_caching

        if numericalization is None:
            example_data = example[self.name]
            numericalization = self.numericalize(example_data)
            if cache:
                example[cache_field_name] = numericalization

        return numericalization

    def __getstate__(self):
        """
        Method obtains field state. It is used for pickling dataset data to
        file.

        Returns
        -------
        state : dict
            dataset state dictionary
        """
        state = self.__dict__.copy()
        if self._tokenizer_arg_string is not None:
            del state["_tokenizer"]
        return state

    def __setstate__(self, state):
        """
        Method sets field state. It is used for unpickling dataset data from
        file.

        Parameters
        ----------
        state : dict
            dataset state dictionary
        """
        self.__dict__.update(state)
        if self._tokenizer_arg_string is not None:
            self._tokenizer = get_tokenizer(self._tokenizer_arg_string)

    def __repr__(self):
        attrs = {
            "name": self.name,
            "keep_raw": self._keep_raw,
            "is_target": self.is_target,
        }
        if self.use_vocab:
            attrs["vocab"] = self.vocab
        return repr_type_and_attrs(self, attrs, with_newlines=True)

    def get_output_fields(self) -> Iterable["Field"]:
        """
        Returns an Iterable of the contained output fields.

        Returns
        -------
        Iterable :
            an Iterable of the contained output fields.
        """
        return (self,)


class MultioutputField:
    """
    Field that does pretokenization and tokenization once and passes it to its
    output fields.

    Output fields are any type of field. The output fields are used only for
    posttokenization processing (posttokenization hooks and vocab updating).
    """

    def __init__(
        self,
        output_fields: List["Field"],
        tokenizer: TokenizerType = "split",
        pretokenize_hooks: Optional[Iterable[PretokenizationHookType]] = None,
    ):
        """
        Field that does pretokenization and tokenization once and passes it to
        its output fields. Output fields are any type of field. The output
        fields are used only for posttokenization processing (posttokenization
        hooks and vocab updating).

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
              by replacing `lang` with the language model name. For example `spacy-en_core_web_sm`

        pretokenize_hooks: Iterable[Callable[[Any], Any]]
            Iterable containing pretokenization hooks. Providing hooks in this way is
            identical to calling `add_pretokenize_hook`.
        """

        self._tokenizer_arg = tokenizer
        self._pretokenization_pipeline = _PretokenizationPipeline()

        if pretokenize_hooks is not None:
            if not isinstance(pretokenize_hooks, (list, tuple)):
                pretokenize_hooks = [pretokenize_hooks]
            for hook in pretokenize_hooks:
                self.add_pretokenize_hook(hook)

        self._tokenizer = get_tokenizer(tokenizer)
        self._output_fields = deque(output_fields)

    def add_pretokenize_hook(self, hook: PretokenizationHookType):
        """
        Add a pre-tokenization hook to the MultioutputField. If multiple hooks
        are added to the field, the order of their execution will be the same as
        the order in which they were added to the field, each subsequent hook
        taking the output of the previous hook as its input. If the same
        function is added to the Field as a hook multiple times, it will be
        executed that many times. The output of the final pre-tokenization hook
        is the raw data that the tokenizer will get as its input.

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
        self._pretokenization_pipeline.add_hook(hook)

    def _run_pretokenization_hooks(self, data: Any) -> Any:
        """
        Runs pretokenization hooks on the raw data and returns the result.

        Parameters
        ----------
        data : Any
            data to be processed

        Returns
        -------
        Any
            processed data
        """

        return self._pretokenization_pipeline(data)

    def add_output_field(self, field: "Field"):
        """
        Adds the passed field to this field's output fields.

        Parameters
        ----------
        field : Field
            Field to add to output fields.
        """
        self._output_fields.append(field)

    def preprocess(self, data: Any) -> Iterable[Tuple[str, Tuple[Optional[Any], Any]]]:
        """
        Preprocesses raw data, tokenizing it if required. The outputfields
        update their vocabs if required and preserve the raw data if the output
        field's 'keep_raw' is true.

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
        tokens = self._tokenizer(data) if self._tokenizer is not None else data
        return tuple(field._process_tokens(data, tokens) for field in self._output_fields)

    def get_output_fields(self) -> Iterable["Field"]:
        """
        Returns an Iterable of the contained output fields.

        Returns
        -------
        Iterable[Field] :
            an Iterable of the contained output fields.
        """
        return self._output_fields

    def remove_pretokenize_hooks(self):
        """
        Remove all the pre-tokenization hooks that were added to the
        MultioutputField.
        """
        self._pretokenization_pipeline.clear()

    def __repr__(self):
        fields_str = ",\n".join(
            textwrap.indent(repr(f), " " * 4) for f in self._output_fields
        )
        fields_str = f"[\n{fields_str}\n    \n]"
        attrs = {"fields": fields_str}
        return repr_type_and_attrs(self, attrs, with_newlines=True, repr_values=False)


class LabelField(Field):
    """
    Field subclass used when no tokenization is required.

    For example, with a field that has a single value denoting a label.
    """

    def __init__(
        self,
        name: str,
        numericalizer: Optional[Union[Vocab, NumericalizerType]] = None,
        allow_missing_data: bool = False,
        disable_batch_matrix: bool = False,
        disable_numericalize_caching: bool = False,
        is_target: bool = True,
        missing_data_token: Union[int, float] = -1,
        pretokenize_hooks: Optional[Iterable[PretokenizationHookType]] = None,
    ):
        """
        Field subclass used when no tokenization is required. For example, with
        a field that has a single value denoting a label.

        Parameters
        ----------
        name : str
            Field name, used for referencing data in the dataset.

        numericalizer : callable
            Object used to numericalize tokens.
            Can either be a Vocab, a custom numericalization callable or None.
            If it's a Vocab, this field will update it after preprocessing (or during
            finalization if eager is False) and use it to numericalize data. Also, the
            Vocab's padding token will be used instead of the Field's.
            If it's a Callable, It will be used to numericalize data token by token.
            If None, numericalization won't be attempted and batches will be created as
            lists instead of numpy matrices.

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

        disable_numericalize_caching : bool
            The flag which determines whether the numericalization of this field should be
            cached. This flag should be set to True if the numericalization can differ
            between `numericalize` function calls for the same instance. When set to False,
            the numericalization values will be cached and reused each time the instance
            is used as part of a batch. The flag is passed to the numericalizer to indicate
            use of its nondeterministic setting. This flag is mainly intended be used in the
            case of masked language modelling, where we wish the inputs to be masked
            (nondeterministic), and the outputs (labels) to not be masked while using the
            same vocabulary.

        is_target : bool
            Whether this field is a target variable. Affects iteration over
            batches.

        missing_data_token : Union[int, float]
            Token to use to mark batch rows as missing. If data for a field is missing,
            its matrix row will be filled with this value. For non-numericalizable fields,
            this parameter is ignored and the value will be None.

        pretokenize_hooks: Iterable[Callable[[Any], Any]]
            Iterable containing pretokenization hooks. Providing hooks in this way is
            identical to calling `add_pretokenize_hook`.
        """
        if numericalizer is None:
            # Default to a vocabulary if custom numericalize is not set
            numericalizer = Vocab(specials=())

        if isinstance(numericalizer, Vocab) and numericalizer.specials:
            raise ValueError(
                "Vocab contains special symbols."
                " Vocabs with special symbols cannot be used"
                " with LabelFields."
            )

        super().__init__(
            name,
            tokenizer=None,
            keep_raw=False,
            numericalizer=numericalizer,
            is_target=is_target,
            include_lengths=False,
            fixed_length=1,
            allow_missing_data=allow_missing_data,
            disable_batch_matrix=disable_batch_matrix,
            disable_numericalize_caching=disable_numericalize_caching,
            missing_data_token=missing_data_token,
            pretokenize_hooks=pretokenize_hooks,
        )


class MultilabelField(Field):
    """
    Field subclass used to get multihot encoded vectors in batches.

    Used in cases when a field can have multiple classes active at a time.
    """

    def __init__(
        self,
        name: str,
        tokenizer: TokenizerType = None,
        numericalizer: Optional[Union[Vocab, NumericalizerType]] = None,
        num_of_classes: Optional[int] = None,
        is_target: bool = True,
        include_lengths: bool = False,
        allow_missing_data: bool = False,
        disable_batch_matrix: bool = False,
        disable_numericalize_caching: bool = False,
        missing_data_token: Union[int, float] = -1,
        pretokenize_hooks: Optional[Iterable[PretokenizationHookType]] = None,
        posttokenize_hooks: Optional[Iterable[PosttokenizationHookType]] = None,
    ):
        """
        Create a MultilabelField from arguments.

        Parameters
        ----------
        name : str
            Field name, used for referencing data in the dataset.

         tokenizer : str | callable | optional
            The tokenizer that is to be used when preprocessing raw data.
            The user can provide his own tokenizer as a callable object or specify one of
            the registered tokenizers by a string. The available pre-registered tokenizers
            are:

            - 'split' - default str.split(). Custom separator can be provided as
              `split-sep` where `sep` is the separator string.
            - 'spacy-lang' - the spacy tokenizer. The language model can be defined
              by replacing `lang` with the language model name. For example `spacy-en_core_web_sm`.

            If None, the data will not be tokenized and post-tokenization hooks wont be
            called. The provided data will be stored in the `tokenized` data field as-is.

        numericalizer : callable
            Object used to numericalize tokens.
            Can either be a Vocab, a custom numericalization callable or None.
            If it's a Vocab, this field will update it after preprocessing (or during
            finalization if eager is False) and use it to numericalize data. The Vocab
            must not contain any special symbols (like PAD or UNK).
            If it's a Callable, It will be used to numericalize data token by token.
            If None, numericalization won't be attempted and batches will be created as
            lists instead of numpy matrices.

        num_of_classes : int, optional
            Number of valid classes.
            Also defines size of the numericalized vector.
            If none, size of the vocabulary is used.

         is_target : bool
            Whether this field is a target variable. Affects iteration over
            batches.

         include_lengths : bool
            Whether the batch representation of this field should include the length
            of every instance in the batch. If true, the batch element under the name
            of this Field will be a tuple of (numericalized values, lengths).

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

        disable_numericalize_caching : bool
            The flag which determines whether the numericalization of this field should be
            cached. This flag should be set to True if the numericalization can differ
            between `numericalize` function calls for the same instance. When set to False,
            the numericalization values will be cached and reused each time the instance
            is used as part of a batch. The flag is passed to the numericalizer to indicate
            use of its nondeterministic setting. This flag is mainly intended be used in the
            case of masked language modelling, where we wish the inputs to be masked
            (nondeterministic), and the outputs (labels) to not be masked while using the
            same vocabulary.

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
            If the provided Vocab contains special symbols.
        """

        if numericalizer is None:
            numericalizer = Vocab(specials=())

        if isinstance(numericalizer, Vocab) and numericalizer.specials:
            raise ValueError(
                "Vocab contains special symbols."
                " Vocabs with special symbols cannot be used"
                " with MultilabelFields."
            )

        self._num_of_classes = num_of_classes
        super().__init__(
            name,
            tokenizer=tokenizer,
            keep_raw=False,
            numericalizer=numericalizer,
            is_target=is_target,
            include_lengths=include_lengths,
            fixed_length=num_of_classes,
            allow_missing_data=allow_missing_data,
            disable_batch_matrix=disable_batch_matrix,
            disable_numericalize_caching=disable_numericalize_caching,
            missing_data_token=missing_data_token,
            pretokenize_hooks=pretokenize_hooks,
            posttokenize_hooks=posttokenize_hooks,
        )

    def finalize(self):
        """
        Signals that this field's vocab can be built.
        """
        super().finalize()
        if self._num_of_classes is None:
            self.fixed_length = self._num_of_classes = len(self.vocab)

        if self.use_vocab and len(self.vocab) > self._num_of_classes:
            raise ValueError(
                "Number of classes in data is greater than the declared number "
                f"of classes. Declared: {self._num_of_classes}, "
                f"Actual: {len(self.vocab)}"
            )

    def numericalize(
        self, data: Tuple[Optional[Any], Optional[Union[Any, List[str]]]]
    ) -> np.ndarray:
        """
        Numericalize the already preprocessed data point based either on the
        vocab that was previously built, or on a custom numericalization
        function, if the field doesn't use a vocab. Returns a numpy array
        containing a multihot encoded vector of num_of_classes length.

        Parameters
        ----------
        data : (hashable, iterable(hashable))
            Tuple of (raw, tokenized) of preprocessed input data. If the field
            is sequential, 'raw' is ignored and can be None. Otherwise,
            'sequential' is ignored and can be None.

        Returns
        -------
        numpy array
            One-hot encoded vector of `num_of_classes` length.

        Raises
        ------
        ValueError
            If data is None and missing data is not allowed in this field.
        """
        active_classes = super(MultilabelField, self).numericalize(data)

        multihot = np.zeros(shape=self._num_of_classes, dtype=np.int)
        if len(active_classes) > 0:
            multihot[active_classes] = 1
        return multihot


def unpack_fields(fields):
    """
    Flattens the given fields object into a flat list of fields.

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

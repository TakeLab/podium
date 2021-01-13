from bisect import bisect_right
from itertools import chain, islice
from math import ceil
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from podium.datasets.dataset import DatasetBase, Example
from podium.field import Field


def concat(
    datasets: List[DatasetBase],
    field_overrides: Optional[Union[Dict[str, Field], List[Field]]] = None,
) -> DatasetBase:
    """
    Concatenate datasets in the passed order.

    Two or more datasets can be concatenated. New fields can be provided as
    'field_overrides' that will be updated with all examples.

    Parameters
    ----------
        datasets: List[DatasetBase]
            A list datasets to be concatenated.
        field_overrides: Union[Dict[str, Field], List[Field]]
            A dict or list containing fields that will be used to override
            existing fields. Can be either a dict mapping old field names to new
            ones, or a list, in which case the field with the same name will be
            overridden. The overridden field will not be present in the
            concatenated view. The override field (if eager) will be updated
            with all examples from the concatenation.

    Returns
    -------
    DatasetBase
        A view of the concatenated Datasets.
    """
    return DatasetConcatView(datasets, field_overrides)


class DatasetConcatView(DatasetBase):
    """
    View used for dataset concatenation.

    Two or more datasets can be concatenated. New fields can be provided as
    'field_overrides' that will be updated with all examples.
    """

    def __init__(
        self,
        datasets: List[DatasetBase],
        field_overrides: Optional[Union[Dict[str, Field], List[Field]]] = None,
    ):
        """
        View used for dataset concatenation.

        Two or more datasets can be
        concatenated. New fields can be provided as 'field_overrides' that will
        be updated with all examples.

        Parameters
        ----------
        datasets: List[DatasetBase]
            A list datasets to be concatenated.
        field_overrides: Union[Dict[str, Field], List[Field]]
            A dict or list containing fields that will be used to override
            existing fields. Can be either a dict mapping old field names to new
            ones, or a list, in which case the field with the same name will be
            overridden. The overridden field will not be present in the
            concatenated view. The override field (if eager) will be updated
            with all examples from the concatenation.
        """
        if isinstance(datasets, DatasetBase):
            # Wrap single dataset in a list
            self._datasets = [datasets]
        elif isinstance(datasets, (list, tuple)):
            self._datasets = list(datasets)
        else:
            err_msg = (
                f"Invalid 'dataset' argument to {type(self).__name__}. "
                f"Supported values are lists or tuples of DatasetBase instances, "
                f"or a single DatasetBase instance. "
                f"Passed type: {type(datasets).__name__}"
            )
            raise TypeError(err_msg)

        if isinstance(field_overrides, list):
            field_overrides = {f.name: f for f in field_overrides}

        self._len = sum([len(ds) for ds in datasets])

        self._cumulative_lengths = [len(self._datasets[0])]
        for dataset in islice(datasets, 1):
            cumulative_len = self._cumulative_lengths[-1] + len(dataset)
            self._cumulative_lengths.append(cumulative_len)

        self._field_overrides = field_overrides or {}

        self._field_mapping = DatasetConcatView._create_field_mapping(
            self._datasets, self._field_overrides
        )
        self._reverse_field_name_mapping_dict = None

        fields = list(self._field_mapping.values())

        super().__init__(fields)
        self._update_override_fields()

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[Example]:
        original_examples = chain(*self._datasets)
        mapped_examples = map(self._map_example, original_examples)
        yield from mapped_examples

    def __getattr__(self, field: Union[str, Field]) -> Iterator[Tuple[Any, Any]]:
        view_field_name = field if isinstance(field, str) else field.name
        if view_field_name not in self._reverse_field_name_mapping:
            # TODO better error message?
            err_msg = (
                f'Field "{view_field_name}" not present in this '
                f"{type(self).__name__}. "
                f"Fields: {list(f.name for f in self.fields)}"
            )
            raise ValueError(err_msg)

        original_field_name = self._reverse_field_name_mapping[view_field_name]
        for ds in self._datasets:
            yield from getattr(ds, original_field_name)

    def __getitem__(self, item):

        if isinstance(item, int):
            dataset, index = self._translate_index(item)
            return self._map_example(dataset[index])

        else:
            return create_view(self, item)

    @staticmethod
    def _create_field_mapping(
        datasets: List[DatasetBase], field_overrides: Dict[str, Field]
    ) -> Dict[str, Field]:
        """
        Creates a dict mapping field names in the original datasets to fields
        they were overridden with: {"field_name" -> Field('override_field')}

        Parameters
        ----------
        datasets: List[DatasetBase]
            List of original datasets.

        field_overrides: Dict[str, Field]
            Dict mapping field names to override fields.

        Returns
        -------
        Dict[str, Field]
            A dict mapping all field names from the original datasets to Field
            instances. Respects field overrides.
        """
        intersection_field_names = DatasetConcatView._get_intersection_field_names(
            datasets
        )

        # Check for empty intersection
        if len(intersection_field_names) == 0:
            err_msg = (
                "Empty field name intersection. "
                "No field name is contained in all passed Datasets."
            )
            raise ValueError(err_msg)

        # Check for invalid overrides
        intersection_field_names_set = set(intersection_field_names)
        for fname in field_overrides:
            if fname not in intersection_field_names_set:
                err_msg = (
                    f'Override field name "{fname}" not contained in the '
                    f"intersection of passed datasets' fields: "
                    f"{intersection_field_names}"
                )
                raise ValueError(err_msg)

        field_mapping = {}
        default_field_dict = datasets[0].field_dict
        for f_name in intersection_field_names:
            if f_name in field_overrides:
                # ignore field and take the override
                override_field = field_overrides[f_name]
                field_mapping[f_name] = override_field
            else:
                # take the field from the first dataset
                original_field = default_field_dict[f_name]
                field_mapping[f_name] = original_field

        return field_mapping

    def _get_examples(self) -> List[Example]:
        return list(self)

    @property
    def _reverse_field_name_mapping(self):
        if self._reverse_field_name_mapping_dict is None:
            self._reverse_field_name_mapping_dict = {
                mapped_field.name: orig_fname
                for orig_fname, mapped_field in self._field_mapping.items()
            }
        return self._reverse_field_name_mapping_dict

    def _update_override_fields(self) -> None:
        """
        Updates and finalizes all eager override fields.
        """
        eager_fields = {
            n: f for n, f in self._field_overrides.items() if not f.finalized and f.eager
        }

        if eager_fields:
            original_examples = chain(*self._datasets)
            for ex in original_examples:
                for original_field_name, override_field in eager_fields.items():
                    _, tokenized = ex[original_field_name]
                    override_field.update_vocab(tokenized)
            for eager_field in eager_fields.values():
                eager_field.finalize()

    def _map_example(self, example: Example) -> Example:
        """
        Transforms an example from a backing dataset into the format of the
        view, respecting field overrides.

        Parameters
        ----------
        example: Example
            Original Example to be mapped.
        Returns
        -------
        Example
            An example mapped to the format of this view.
        """
        new_example = Example()
        for original_field_name, mapped_field in self._field_mapping.items():
            new_example[mapped_field.name] = example[original_field_name]
        return new_example

    def _translate_index(self, index: int) -> Tuple[DatasetBase, int]:
        """
        For an index in the view, returns the backing Dataset it belongs to and
        the index of the example in that Dataset.

        Parameters
        ----------
        index: int
            The index to be translated.

        Returns
        -------
        (DatasetBase, int)
            The dataset that contains the indexed example and its index in that
            dataset.
        """
        if index >= len(self):
            raise IndexError(f"Index {index} out of range. Length: {len(self)}")

        if index < 0:
            # correct for negative indexing
            index %= len(self)

        # Use binary search to determine the index of the containing dataset
        dataset_index = bisect_right(self._cumulative_lengths, index)
        offset = self._cumulative_lengths[dataset_index - 1] if dataset_index > 0 else 0
        translated_index = index - offset
        dataset = self._datasets[dataset_index]
        return dataset, translated_index

    @staticmethod
    def _get_intersection_field_names(datasets: List[DatasetBase]) -> List[str]:
        field_dict = datasets[0].field_dict
        intersection_field_names = set(field_dict.keys())
        for ds in datasets[1:]:
            # Calculate the intersection of all field names
            intersection_field_names.intersection_update(ds.field_dict.keys())
        return list(intersection_field_names)


def create_view(dataset: DatasetBase, i: Union[Sequence[int], slice]) -> DatasetBase:
    """
    Creates a view that is appropriate for the passed indexing method.

    Parameters
    ----------
    dataset: DatasetBase
        The dataset the view will be created on.
    i: Union[Sequence[int], slice]
        The indices contained in the view.

    Returns
    -------
        A view on the passed dataset.
    """
    if isinstance(i, slice):
        return DatasetSlicedView(dataset, i)
    else:
        return DatasetIndexedView(dataset, i)


class DatasetIndexedView(DatasetBase):
    """
    View over a DatasetBase class.
    """

    def __init__(self, dataset: DatasetBase, indices: Sequence[int]):
        """
        Creates a view over the passed dataset.

        Parameters
        ----------
        dataset: DatasetBase
            The dataset the view will be created over.
        indices: Sequence[int]
            A sequence of ints that represent the indices of the examples in the
            dataset that will be contained in the view. Ordering and duplication
            will be respected.
        """
        if not isinstance(dataset, DatasetBase):
            err_msg = (
                f"'dataset' parameter must be of type DatasetBase. "
                f"Passed type: {type(dataset).__name__}"
            )
            raise TypeError(err_msg)

        self._dataset = dataset
        self._indices = indices
        super().__init__(dataset.fields)

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, item):
        if isinstance(item, int):
            index = self._indices[item]
            return self._dataset[index]

        if isinstance(item, slice):
            new_indices = self._indices[item]
            return DatasetIndexedView(self._dataset, new_indices)

        else:
            new_indices = [self._indices[i] for i in item]
            return DatasetIndexedView(self._dataset, new_indices)

    def __iter__(self):
        yield from (self._dataset[i] for i in self._indices)

    def _get_examples(self) -> List[Example]:
        return list(self)


class DatasetSlicedView(DatasetBase):
    """
    View over a DatasetBase class.
    """

    def __init__(self, dataset: DatasetBase, s: slice):
        """
        Creates a view over the passed dataset.

        Parameters
        ----------
        dataset: DatasetBase
            The dataset the view will be created over.
        s: slice
            A slice indexing the wanted examples.
        """
        if not isinstance(dataset, DatasetBase):
            err_msg = (
                f"'dataset' parameter must be of type DatasetBase. "
                f"Passed type: {type(dataset).__name__}"
            )
            raise TypeError(err_msg)

        if not isinstance(s, slice):
            err_msg = (
                f"'s' parameter must be of type slice. "
                f"Passed type: {type(s).__name__}"
            )
            raise TypeError(err_msg)

        self._dataset = dataset
        start, stop, step = s.indices(len(dataset))
        self._slice = slice(start, stop, step)
        self._len = self._calculate_length()
        super().__init__(dataset.fields)

    def _calculate_length(self) -> int:
        """
        Calculates the number of examples in this view.

        Returns
        -------
        int:
            The number of examples in this view.
        """
        start, stop, step = self._slice.start, self._slice.stop, self._slice.step
        if step < 0:
            start, stop = stop, start
            step *= -1

        return ceil(max(stop - start, 0) / step)

    def __len__(self):
        return self._len

    def __iter__(self):
        start, stop, step = self._slice.start, self._slice.stop, self._slice.step
        for i in range(start, stop, step):
            yield self._dataset[i]

    def __getitem__(self, item):
        if isinstance(item, int):
            if item >= len(self):
                err_msg = f"Index {item} out of bounds. Length: {len(self)}."
                raise IndexError(err_msg)

            if item < 0:
                item %= len(self)

            index = self._slice.start + item * self._slice.step
            return self._dataset[index]

        else:
            return create_view(self, item)

    def _get_examples(self) -> List[Example]:
        return list(self)

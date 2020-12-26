from bisect import bisect_right
from itertools import chain, islice
from math import ceil
from typing import Any, Dict, Iterator, List, Sequence, Tuple, Union

from podium.datasets.dataset_abc import DatasetABC
from podium.storage import Example, Field


class DatasetConcatView(DatasetABC):
    def __init__(
        self, datasets: List[DatasetABC], field_overrides: Dict[str, Field] = None
    ):
        if isinstance(datasets, DatasetABC):
            # Wrap single dataset in a list
            self._datasets = [datasets]
        elif isinstance(datasets, (list, tuple)):
            self._datasets = list(datasets)
        else:
            err_msg = (
                f"Invalid 'dataset' argument to {type(self).__name__}. "
                f"Supported values are lists or tuples of DatasetABC instances, "
                f"or a single DatasetABC instance. "
                f"Passed type: {type(datasets).__name__}"
            )
            raise TypeError(err_msg)

        self._len = sum(map(len, datasets))

        self._cumulative_lengths = [len(self._datasets[0])]
        for dataset in islice(datasets, 1):
            cumulative_len = self._cumulative_lengths[-1] + len(dataset)
            self._cumulative_lengths.append(cumulative_len)

        self._field_overrides = field_overrides or {}
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
        for fname in self._field_overrides:
            if fname not in intersection_field_names_set:
                err_msg = (
                    f'Override field name "{fname}" not contained in the '
                    f"intersection of passed datasets' fields: "
                    f"{intersection_field_names}"
                )
                raise ValueError(err_msg)

        field_mapping = {}
        default_field_dict = self._datasets[0].field_dict
        for f_name in intersection_field_names:
            if f_name in self._field_overrides:
                # ignore field and take the override
                override_field = self._field_overrides[f_name]
                field_mapping[f_name] = override_field
            else:
                # take the field from the first dataset
                original_field = default_field_dict[f_name]
                field_mapping[f_name] = original_field

        self._field_mapping = field_mapping
        self._reverse_field_name_mapping = {
            mapped_field.name: orig_fname
            for orig_fname, mapped_field in self._field_mapping.items()
        }
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
            err_msg = f'Field "{view_field_name}" not present in this ' \
                      f'{type(self).__name__}. ' \
                      f'Fields: {list(f.name for f in self.fields)}'
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

    def _get_examples(self) -> List[Example]:
        sublists = map(lambda ds: ds.examples, self._datasets)
        return [ex for sublist in sublists for ex in sublist]

    def _update_override_fields(self):
        eager_fields = {
            n: f for n, f in self._field_overrides.items() if not f.finalized and f.eager
        }

        if eager_fields:
            original_examples = chain(*self._datasets)
            for ex in original_examples:
                for original_field_name, override_field in eager_fields:
                    _, tokenized = ex[original_field_name]
                    override_field.update(tokenized)
            for eager_field in eager_fields.values():
                eager_field.finalize()

    def _map_example(self, example: Example) -> Example:
        new_example = Example()
        for original_field_name, mapped_field in self._field_mapping.items():
            new_example[mapped_field.name] = example[original_field_name]
        return new_example

    def _translate_index(self, index: int) -> Tuple[DatasetABC, int]:
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
    def _get_intersection_field_names(datasets: List[DatasetABC]) -> List[str]:
        field_dict = datasets[0].field_dict
        intersection_field_names = set(field_dict.keys())
        for ds in datasets[1:]:
            # Calculate the intersection of all field names
            intersection_field_names.intersection_update(ds.field_dict.keys())
        return list(intersection_field_names)


def create_view(dataset: DatasetABC, i: Union[Sequence[int], slice]) -> DatasetABC:
    if isinstance(i, slice):
        return DatasetSlicedView(dataset, i)
    else:
        return DatasetIndexedView(dataset, i)


class DatasetIndexedView(DatasetABC):
    def __init__(self, dataset: DatasetABC, indices: Sequence[int]):
        self._dataset = dataset
        self._indices = indices
        super().__init__(dataset.field_dict)

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


class DatasetSlicedView(DatasetABC):
    def __init__(self, dataset: DatasetABC, s: slice):
        self._dataset = dataset
        start, stop, step = s.indices(len(dataset))
        self._slice = slice(start, stop, step)
        super().__init__(dataset.field_dict)

    def __len__(self):
        start, stop, step = self._slice.start, self._slice.stop, self._slice.step
        if step < 0:
            start, stop = stop, start
            step *= -1

        return ceil(max(stop - start, 0) / step)

    def __iter__(self):
        start, stop, step = self._slice.start, self._slice.stop, self._slice.step
        for i in range(start, stop, step):
            yield self._dataset[i]

    def __getitem__(self, item):
        if isinstance(item, int):
            # TODO check bounds
            index = self._slice.start + item * self._slice.step
            return self._dataset[index]

        else:
            return create_view(self, item)

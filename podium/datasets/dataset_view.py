from itertools import chain
from typing import Any, Dict, Iterator, List, Tuple, Union, Sequence

from podium.datasets.dataset_abc import DatasetABC
from podium.storage import Example, Field


class DatasetConcatView(DatasetABC):

    def __init__(self, datasets: List[DatasetABC], field_overrides: Dict[str, Field]):
        self._datasets = list(datasets)
        # TODO add warning for no datasets ?
        self._len = sum(map(len, datasets))

        self._cumulative_lengths = [len(self._datasets[0])]
        for dataset in datasets[1:]:
            cumulative_len = self._cumulative_lengths[-1] + len(dataset)
            self._cumulative_lengths.append(cumulative_len)

        self._field_overrides = dict(field_overrides)
        # TODO add warning for non-existing override mapping
        # TODO add warning for empty field intersection

        intersection_field_names = DatasetConcatView._get_intersection_field_names(
            datasets
        )
        field_mapping = {}
        for f_name in intersection_field_names:
            if f_name in field_overrides:
                # ignore field and take the override
                override_field = field_overrides[f_name]
                field_mapping[f_name] = override_field
            else:
                # take the field from the first dataset
                original_field = self._datasets[0].field_dict[f_name]
                field_mapping[f_name] = original_field

        self._field_mapping = field_mapping
        field_dict = {f.name: f for f in field_mapping.values()}

        self._update_override_fields()
        super().__init__(field_dict)

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[Example]:
        original_examples = chain(*self._datasets)
        mapped_examples = map(self._map_example, original_examples)
        yield from mapped_examples

    def __getattr__(self, field: Union[str, Field]) -> Iterator[Tuple[Any, Any]]:
        field_name = field if isinstance(field, str) else field.name
        attr_getters = [getattr(ds, field_name) for ds in self._datasets]
        yield from chain(*attr_getters)

    def __getitem__(self, item):

        if isinstance(item, int):
            dataset, index = self._translate_index(item)
            return dataset[index]

        # TODO imlpement after DatasetIndexedView
        pass

    def _get_examples(self) -> List[Example]:
        sublists = map(lambda ds: ds.examples, self._datasets)
        return [ex for sublist in sublists for ex in sublist]

    def _update_override_fields(self):
        eager_fields = {
            n: f for n, f in self._field_overrides.items() if not f.finalized and f.eager
        }
        non_eager_fields = {
            n: f
            for n, f in self._field_overrides.items()
            if not f.finalized and not f.eager
        }

        if eager_fields:
            original_examples = chain(*self._datasets)
            for ex in original_examples:
                for original_field_name, override_field in eager_fields:
                    _, tokenized = ex[original_field_name]
                    override_field.update(tokenized)
            for eager_field in eager_fields.values():
                eager_field.finalize()

        if non_eager_fields:
            original_examples = chain(*self._datasets)
            for ex in original_examples:
                for original_field_name, override_field in non_eager_fields:
                    _, tokenized = ex[original_field_name]
                    override_field.update(tokenized)
            for non_eager_field in non_eager_fields.values():
                non_eager_field.finalize()

    def _map_example(self, example: Example) -> Example:
        new_example = Example()
        for original_field_name, mapped_field in self._field_mapping:
            new_example[mapped_field.name] = example[original_field_name]
        return new_example

    def _translate_index(self, index: int) -> Tuple[DatasetABC, int]:
        assert index < len(self), f"Index {index} out of range. Length: {len(self)}"

        if index < 0:
            # correct for negative indexing
            index %= len(self)

        # TODO implement better search alg? Binary search?
        dataset_index = 0
        for cumulative_len in self._cumulative_lengths:
            if index < cumulative_len:
                break
            dataset_index += 1

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

    # TODO Implement __repr__


# TODO create_view function

class DatasetIndexedView(DatasetABC):

    def __init__(self,
                 dataset: DatasetABC,
                 indices: Sequence[int]):
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

    # TODO implement __repr__
from itertools import chain
from typing import Dict, List, Iterator

from podium.datasets.dataset_abc import DatasetABC
from podium.storage import Example, Field


class DatasetConcatView(DatasetABC):

    def __init__(self, datasets: List[DatasetABC], field_overrides: Dict[str, Field]):
        self._datasets = list(datasets)
        self._len = sum(map(len, datasets))

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

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[Example]:
        original_examples = chain(*self._datasets)
        mapped_examples = map(self._map_example, original_examples)
        yield from mapped_examples

    def _map_example(self, example: Example) -> Example:
        new_example = Example()
        for original_field_name, mapped_field in self._field_mapping:
            new_example[mapped_field.name] = example[original_field_name]
        return new_example

    @staticmethod
    def _get_intersection_field_names(datasets: List[DatasetABC]) -> List[str]:
        field_dict = datasets[0].field_dict
        intersection_field_names = set(field_dict.keys())
        for ds in datasets[1:]:
            # Calculate the intersection of all field names
            intersection_field_names.intersection_update(ds.field_dict.keys())
        return list(intersection_field_names)

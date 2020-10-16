import os
import dill
import pytest
import tempfile
import numpy as np

from collections import Counter
from json import JSONDecodeError
from podium.datasets.dataset import Dataset
from podium.datasets.hierarhical_dataset import HierarchicalDataset
from podium.datasets.tabular_dataset import TabularDataset
from podium.storage.field import Field, MultioutputField, unpack_fields
from podium.storage.example_factory import ExampleFactory
from podium.storage.vocab import Vocab
from podium.datasets.iterator import Iterator

FORMAT_USE_DICT_COMBINATIONS = (
    ("csv", True),
    ("csv", False),
    ("tsv", True),
    ("tsv", False),
    ("json", True)
)

TEXT = (
    "ovaj text", "ovaj isto", "bla bla",
    "komentar", "---", "pls",
    "help", "im", "trapped",
    "in", "takelab", "...."
)

FIELD_DATA = (
    ("text", True),
    ("label", False)
)

TABULAR_TEXT = (
    "odlicni cevapi",
    "ma ful, \"odlicni\" cevapi ..",
    "mozd malo prepikantni cevapi, al inace ok",
    "nema veganskih cevapa..u kojem stoljecu zivimo?"
)

TABULAR_RATINGS = (1, 0, 1, 0)

TABULAR_SOURCES = (
    "www.volimljuto.hr",
    "www.mrzimljuto.hr",
    "www.nekadminepaseljuto.hr",
    "www.neamideje.hr"
)


class MockField:
    def __init__(self, name, eager, keep_raw=True, tokenize=True, is_target=False):
        self.name = name
        self.eager = eager
        self.keep_raw = keep_raw
        self.tokenize = True

        self.finalized = False
        self.updated_count = 0

        self.use_vocab = True

        self.is_target = is_target

    def preprocess(self, data):
        raw = data if self.keep_raw else None
        tokenized = [data] if self.tokenize else data

        return (self.name, (raw, tokenized)),

    def update_vocab(self, tokenized):
        assert not self.eager
        self.updated_count += 1

    def finalize(self):
        self.finalized = True

    def get_output_fields(self):
        return self,

    def __repr__(self):
        return self.name


class MockExample:
    def __init__(self, fields, data):
        self.accessed = False

        for f, d in zip(fields, data):
            for name, data in f.preprocess(d):
                self.__setattr__(name, data)

    def __getattribute__(self, item):
        if item not in {"accessed", "__setattr__"}:
            self.accessed = True

        return super().__getattribute__(item)


def test_finalize_fields(data, field_list):
    dataset = create_dataset(data, field_list)

    for f in field_list:
        # before finalization, no field's dict was updated
        assert f.updated_count == 0
        assert not f.finalized

    dataset.finalize_fields()

    for f in field_list:
        # during finalization, only non-eager field's dict should be updated
        assert f.updated_count == (len(data) if (not f.eager) else 0)

        # all fields should be finalized
        assert f.finalized


def test_finalize_fields_pickle(data, field_list, tmpdir):
    dataset = create_dataset(data, field_list)
    dataset.finalize_fields()
    dataset_file = os.path.join(tmpdir, "dataset.pkl")

    with open(dataset_file, "wb") as fdata:
        dill.dump(dataset, fdata)

    with open(dataset_file, "rb") as fdata:
        loaded_dataset = dill.load(fdata)
        for f in loaded_dataset.field_dict.values():
            assert f.updated_count == (len(data) if (not f.eager) else 0)
            assert f.finalized


def test_not_finalize_fields_pickle(data, field_list, tmpdir):
    dataset = create_dataset(data, field_list)
    dataset_file = os.path.join(tmpdir, "dataset.pkl")

    with open(dataset_file, "wb") as fdata:
        dill.dump(dataset, fdata)

    with open(dataset_file, "rb") as fdata:
        loaded_dataset = dill.load(fdata)
        for f in loaded_dataset.field_dict.values():
            assert f.updated_count == 0
            assert not f.finalized


def test_finalize_fields_after_split(data, field_list):
    dataset = create_dataset(data, field_list)

    d_train, d_val, d_test = dataset.split([0.3334, 0.333, 0.333])

    # using only the train set to build the vocabs for non-eager fields
    dataset.finalize_fields(d_train)

    # only the train set examples should have been accessed
    # during finalization
    for example in d_train.examples:
        assert example.accessed

    for example in d_val.examples:
        assert not example.accessed

    for example in d_test.examples:
        assert not example.accessed


def test_filter_dataset_inplace(data, field_list):
    dataset = create_dataset(data, field_list)
    assert len(dataset) == 12
    dataset.filter(lambda ex: ex.label[0] > 7, inplace=True)
    assert len(dataset) == 5
    for ex in dataset:
        assert ex.label[0] > 7


def test_filter_dataset_copy(data, field_list):
    dataset = create_dataset(data, field_list)
    assert len(dataset) == 12
    filtered_dataset = dataset.filter(
        lambda ex: ex.label[0] > 7, inplace=False)
    assert len(dataset) == 12
    assert len(filtered_dataset) == 5
    for ex in filtered_dataset:
        assert ex.label[0] > 7


def test_filtered_inplace_dataset_pickling(data, field_list, tmpdir):
    dataset = create_dataset(data, field_list)
    assert len(dataset) == 12
    dataset.filter(lambda ex: ex.label[0] > 7, inplace=True)
    assert len(dataset) == 5

    dataset_file = os.path.join(tmpdir, "dataset.pkl")

    with open(dataset_file, "wb") as fdata:
        dill.dump(dataset, fdata)

    with open(dataset_file, "rb") as fdata:
        loaded_dataset = dill.load(fdata)
        assert len(loaded_dataset) == 5
        for ex in dataset:
            assert ex.label[0] > 7


@pytest.mark.parametrize(
    "float_ratio, expected_train_len, expected_test_len",
    [
        (0.4, 5, 7),
        (0.42, 5, 7),
        (0.35, 4, 8),
        (0.1, 1, 11),
        (0.9, 11, 1),
    ]
)
def test_split_float_ratio(float_ratio, expected_train_len, expected_test_len,
                           data, field_list):
    expected_total_len = expected_train_len + expected_test_len

    dataset = create_dataset(data, field_list)
    assert len(dataset.examples) == expected_total_len

    train_d, test_d = dataset.split(float_ratio)

    assert len(train_d) + len(test_d) == expected_total_len
    assert len(train_d.examples) == expected_train_len
    assert len(test_d.examples) == expected_test_len


@pytest.mark.parametrize(
    "train_test_ratio, expected_train_len, expected_test_len",
    [
        ((0.1, 0.9), 1, 11),
        ([10.0, 2.0], 10, 2),
        ([0.92, 0.08], 11, 1),
        ((0.5, 0.5), 6, 6),
        ([0.41, 0.59], 5, 7),
        ([0.49, 0.51], 6, 6),
    ]
)
def test_split_train_test_ratio(train_test_ratio, expected_train_len,
                                expected_test_len, data, field_list):
    expected_total_len = expected_train_len + expected_test_len

    dataset = create_dataset(data, field_list)
    assert len(dataset.examples) == expected_total_len

    train_d, test_d = dataset.split(train_test_ratio)

    assert len(train_d) + len(test_d) == expected_total_len
    assert len(train_d.examples) == expected_train_len
    assert len(test_d.examples) == expected_test_len


@pytest.mark.parametrize(
    "train_val_test_ratio, exp_train_len, exp_val_len, exp_test_len",
    [
        ([0.4, 0.2, 0.4], 5, 2, 5),
        ((6.0, 1.0, 5.0), 6, 1, 5),
        ((0.33, 0.33, 0.33), 4, 4, 4),
        ([0.84, 0.08, 0.08], 10, 1, 1),
        ([0.08, 0.84, 0.08], 1, 10, 1),
        ((0.3, 0.3, 0.3), 4, 4, 4),
    ]
)
def test_split_train_val_test_ratio(
        train_val_test_ratio, exp_train_len, exp_val_len, exp_test_len,
        data, field_list):
    exp_total_len = exp_train_len + exp_val_len + exp_test_len

    dataset = create_dataset(data, field_list)
    assert len(dataset.examples) == exp_total_len

    train_d, val_d, test_d = dataset.split(train_val_test_ratio)

    assert len(train_d) + len(val_d) + len(test_d) == exp_total_len
    assert len(train_d.examples) == exp_train_len
    assert len(val_d.examples) == exp_val_len
    assert len(test_d.examples) == exp_test_len


@pytest.mark.parametrize(
    "split_ratio",
    [
        (0.3, 0.3, 0.3),
        (0.2, 0.4, 0.4),
        (0.2, 0.3, 0.5)
    ]
)
def test_split_non_overlap(split_ratio, data, field_list):
    dataset = create_dataset(data, field_list)

    label_set = set(map(lambda ex: ex.label[0], dataset.examples))
    train_d, val_d, test_d = dataset.split(split_ratio)

    train_label_set = set(map(lambda ex: ex.label[0], train_d.examples))
    val_label_set = set(map(lambda ex: ex.label[0], val_d.examples))
    test_label_set = set(map(lambda ex: ex.label[0], test_d.examples))

    assert not train_label_set.intersection(val_label_set)
    assert not train_label_set.intersection(test_label_set)
    assert not val_label_set.intersection(test_label_set)
    assert train_label_set.union(val_label_set).union(
        test_label_set) == label_set


@pytest.mark.parametrize(
    "ratio",
    [
        [0.62, 0.2, 0.1, 0.7],
        None,
        1.5,
        -0.2,
        [0.3, 0.0, 0.7],
        [0.998, 0.001, 0.001],  # these are incorrect ratios because for the
        (0.998, 0.001, 0.001),  # given dataset they would result in some
        [0.999, 0.001],  # splits having 0 (the same ratios could be
        0.999,  # valid with larger datasets)
    ]
)
def test_split_wrong_ratio(data, field_list, ratio):
    dataset = create_dataset(data, field_list)

    # all the ratios provided are wrong in their own way
    with pytest.raises(ValueError):
        dataset.split(ratio)


def test_split_stratified_ok(data_for_stratified, field_list):
    dataset = create_dataset(data_for_stratified, field_list)

    # should split evenly
    split_ratio = [0.33, 0.33, 0.33]

    train_d, val_d, test_d = dataset.split(split_ratio, stratified=True,
                                           random_state=1)

    assert len(train_d) == len(test_d)
    assert len(train_d) == len(val_d)
    assert len(train_d) + len(val_d) + len(test_d) == len(dataset)

    train_label_counter = Counter(ex.label[0] for ex in train_d.examples)
    val_label_counter = Counter(ex.label[0] for ex in val_d.examples)
    test_label_counter = Counter(ex.label[0] for ex in test_d.examples)

    all_labels = set(ex.label[0] for ex in dataset.examples)

    # stratified split preserves the percentage of each class in every split
    # if the splits are the same (1/3), then the number of examples with
    # the same class will be the same in all three splits
    for label in all_labels:
        assert train_label_counter[label] == val_label_counter[label]
        assert train_label_counter[label] == test_label_counter[label]
        assert val_label_counter[label] == test_label_counter[label]


def test_split_stratified_custom_name(data_for_stratified, field_list):
    dataset = create_dataset(data_for_stratified, field_list)

    dataset.split(split_ratio=0.5, stratified=True, strata_field_name="label")


def test_split_stratified_exception_invalid_name(data_for_stratified,
                                                 field_list):
    dataset = create_dataset(data_for_stratified, field_list)

    # when field with the given name doesn't exist
    with pytest.raises(ValueError):
        dataset.split(split_ratio=0.5, stratified=True,
                      strata_field_name="NOT_label")


def test_split_stratified_exception_no_target(data_for_stratified,
                                              field_list):
    for field in field_list:
        field.is_target = False

    dataset = create_dataset(data_for_stratified, field_list)

    # when there is no target fields and the strata field name is not given
    with pytest.raises(ValueError):
        dataset.split(split_ratio=0.5, stratified=True)


@pytest.mark.parametrize(
    "file_format, use_dict",
    FORMAT_USE_DICT_COMBINATIONS
)
def test_tabular_dataset_should_ignore(file_format, use_dict,
                                       tabular_dataset_fields, tabular_data,
                                       file_path):
    tabular_data["should_ignore"] = ["a", "b", "c", "d"]

    SHOULD_IGNORE = None
    if not use_dict:
        tabular_dataset_fields.append(SHOULD_IGNORE)

    dataset = create_tabular_dataset(tabular_dataset_fields, file_format,
                                     file_path, use_dict)

    # SHOULD_IGNORE was ignored
    assert "should_ignore" not in set(f.name for f in dataset.fields)


@pytest.mark.parametrize(
    "file_format, use_dict",
    FORMAT_USE_DICT_COMBINATIONS
)
def test_tabular_dataset_preserve_sort_key(file_format, use_dict,
                                           tabular_dataset_fields, file_path):
    sort_key_str = "d_sort_key"

    dataset = create_tabular_dataset(tabular_dataset_fields, file_format,
                                     file_path, use_dict)
    dataset.sort_key = sort_key_str

    dataset.finalize_fields()
    d_train, d_test = dataset.split(split_ratio=0.5, shuffle=False)

    # the sort key should be passed from the original dataset
    assert d_train.sort_key == sort_key_str
    assert d_test.sort_key == sort_key_str


@pytest.mark.parametrize(
    "file_format, use_dict",
    FORMAT_USE_DICT_COMBINATIONS
)
def test_tabular_dataset_pickle_sort_key(
        file_format, use_dict, tabular_dataset_fields, file_path, tmpdir):
    sort_key_str = "d_sort_key"
    dataset = create_tabular_dataset(tabular_dataset_fields, file_format,
                                     file_path, use_dict,
                                     sort_key=sort_key_str)
    dataset.finalize_fields()

    dataset_file = os.path.join(tmpdir, "dataset.pkl")

    with open(dataset_file, "wb") as fdata:
        dill.dump(dataset, fdata)

    with open(dataset_file, "rb") as fdata:
        loaded_dataset = dill.load(fdata)
        assert loaded_dataset.sort_key == sort_key_str


@pytest.mark.parametrize(
    "file_format, use_dict",
    FORMAT_USE_DICT_COMBINATIONS
)
def test_tabular_dataset_iterate_over_dataset(file_format, use_dict,
                                              tabular_dataset_fields,
                                              tabular_data, file_path):
    dataset = create_tabular_dataset(tabular_dataset_fields, file_format,
                                     file_path, use_dict)

    field_name = "text"
    for example, val in zip(dataset, tabular_data[field_name]):
        expected_data = val, [val]

        assert getattr(example, field_name) == expected_data


@pytest.mark.parametrize(
    "file_format, use_dict",
    FORMAT_USE_DICT_COMBINATIONS
)
def test_tabular_dataset_iterate_over_examples(file_format, use_dict,
                                               tabular_dataset_fields,
                                               tabular_data, file_path):
    dataset = create_tabular_dataset(tabular_dataset_fields, file_format,
                                     file_path, use_dict)

    field_name = "text"
    for example, val in zip(dataset.examples, tabular_data[field_name]):
        expected_data = val, [val]

        assert getattr(example, field_name) == expected_data


@pytest.mark.parametrize(
    "file_format, use_dict",
    FORMAT_USE_DICT_COMBINATIONS
)
def test_tabular_dataset_iterate_over_single_field(file_format, use_dict,
                                                   tabular_dataset_fields,
                                                   tabular_data, file_path):
    dataset = create_tabular_dataset(tabular_dataset_fields, file_format,
                                     file_path, use_dict)

    field_name = "text"

    # iterating over a single field of a dataset
    field_val_expected_data_tuples = zip(getattr(dataset, field_name),
                                         tabular_data[field_name])

    for field_value, val in field_val_expected_data_tuples:
        expected_data = (val, [val])

        assert field_value == expected_data


@pytest.mark.parametrize(
    "file_format, use_dict",
    FORMAT_USE_DICT_COMBINATIONS
)
def test_tabular_dataset_get_example_by_index(file_format, use_dict,
                                              tabular_dataset_fields,
                                              tabular_data, file_path):
    dataset = create_tabular_dataset(tabular_dataset_fields, file_format,
                                     file_path, use_dict)

    field_name = "text"
    index_of_example = 3

    val = tabular_data[field_name][index_of_example]
    expected_data = val, [val]

    assert getattr(dataset[index_of_example], field_name) == expected_data


@pytest.mark.parametrize(
    "file_format, use_dict",
    FORMAT_USE_DICT_COMBINATIONS
)
def test_tabular_dataset_exception(file_format, use_dict,
                                   tabular_dataset_fields, file_path):
    # skip_header True when using a dict
    if use_dict:
        with pytest.raises(ValueError):
            TabularDataset(file_path, file_format, tabular_dataset_fields,
                           skip_header=True)

    # wrong file_format given
    with pytest.raises(ValueError):
        TabularDataset(file_path, "wrong_format", tabular_dataset_fields)

    td = TabularDataset(file_path, file_format, tabular_dataset_fields)

    # accessing a non-existing field
    with pytest.raises(AttributeError):
        next(getattr(td, "not_text"))


def test_dataset_slicing(data, field_list):
    dataset = create_dataset(data, field_list)

    def fst(x):
        return x[0]

    def get_raw(example):
        return example.text[0]

    dataset_0_4 = dataset[:4]
    assert list(map(get_raw, dataset_0_4)) == list(map(fst, data[0:4]))
    assert isinstance(dataset_0_4, Dataset)

    dataset_2_end = dataset[2:]
    assert list(map(get_raw, dataset_2_end)) == list(map(fst, data[2:None]))
    assert isinstance(dataset_2_end, Dataset)

    dataset_2_4 = dataset[2:4]
    assert list(map(get_raw, dataset_2_4)) == list(map(fst, data[2:4]))
    assert isinstance(dataset_2_4, Dataset)

    dataset_2_5_neg1 = dataset[2:5:-1]
    assert list(map(get_raw, dataset_2_5_neg1)) == list(map(fst, data[2:5:-1]))
    assert isinstance(dataset_2_5_neg1, Dataset)

    dataset_2_6_neg2 = dataset[2:6:-2]
    assert list(map(get_raw, dataset_2_6_neg2)) == list(map(fst, data[2:6:-2]))
    assert isinstance(dataset_2_6_neg2, Dataset)


def test_dataset_multiindexing(data, field_list):
    dataset = create_dataset(data, field_list)

    def get_raw(example):
        return example.text[0]

    def test_indexing(indexes):
        true_data = [data[i][0] for i in indexes]
        indexed_dataset = dataset[indexes]
        assert isinstance(indexed_dataset, Dataset)
        indexed_dataset_raw = map(get_raw, indexed_dataset)
        assert all(a == b for a, b in zip(indexed_dataset_raw, true_data))

    test_indexing(list(range(0, 10)))
    test_indexing(list(range(9, 0, -1)))
    test_indexing(list(range(8, 1, -2)))
    test_indexing([0, 1, 1, 1, 2, 3, 4, 5, 1, 10, 2])
    test_indexing(np.array([0, 2, 3, 5, 3]))
    test_indexing(list(range(1, 10, 3)))


def test_dataset_deep_copy(data, field_list):
    original_dataset = create_dataset(data, field_list)
    original_examples = original_dataset.examples

    dataset_no_deep_copy = original_dataset.get(slice(0, 5), deep_copy=False)
    for original, copy in zip(original_dataset.fields, dataset_no_deep_copy.fields):
        assert copy is original
    for original, copy in zip(original_examples, dataset_no_deep_copy.examples):
        assert copy is original

    dataset_deep_copy = original_dataset.get(slice(0, 5), deep_copy=True)

    assert original_dataset.fields is not dataset_deep_copy.fields
    for original, copy in zip(original_dataset.fields, dataset_deep_copy.fields):
        assert copy is not original

    for original, copy in zip(original_examples, dataset_deep_copy.examples):
        assert copy is not original
        assert copy.text == original.text
        assert copy.label == original.label

    original_example = original_examples[2]
    no_copy_example = original_dataset.get(2, deep_copy=False)
    indexed_example = original_dataset[2]
    deep_copied_example = original_dataset.get(2, deep_copy=True)

    assert no_copy_example is original_example
    assert indexed_example is original_example
    assert deep_copied_example is not original_example
    assert deep_copied_example.text == original_example.text
    assert deep_copied_example.label == original_example.label


def test_dataset_multiindexing_pickling(data, field_list):
    dataset = create_dataset(data, field_list)

    def example_equals(a, b):
        return a.text == b.text and a.label == b.label

    indexed_dataset = dataset[0, 2, 3]
    with tempfile.TemporaryFile() as file:
        dill.dump(indexed_dataset, file)
        file.seek(0)
        loaded_dataset = dill.load(file)

    assert isinstance(loaded_dataset, Dataset)
    assert len(indexed_dataset) == len(loaded_dataset)
    assert all(example_equals(a, b) for a, b in zip(indexed_dataset, loaded_dataset))


@pytest.fixture
def field_list():
    return [MockField(field_name, eager, is_target=(field_name == "label"))
            for field_name, eager in FIELD_DATA]


@pytest.fixture
def data():
    labels = range(1, len(TEXT) + 1)
    data = list(zip(TEXT, labels))

    return data


@pytest.fixture
def data_for_stratified():
    labels = [1] * 3 + [0] * 9

    data = list(zip(TEXT, labels))

    return data


@pytest.fixture
def tabular_dataset_fields(use_dict):
    TEXT = MockField('text', eager=True)
    CHARS = MockField('chars', eager=True)
    RATING = MockField('rating', tokenize=False, eager=True)
    SOURCE = MockField('source', tokenize=False, eager=False)

    if use_dict:
        fields = {"text": (TEXT, CHARS), "rating": RATING, "source": SOURCE}
    else:
        fields = [(TEXT, CHARS), RATING, SOURCE]

    return fields


@pytest.fixture
def tabular_data():
    return {"text": TABULAR_TEXT, "rating": TABULAR_RATINGS,
            "source": TABULAR_SOURCES}


def create_dataset(data, field_list):
    examples = [MockExample(field_list, d) for d in data]

    return Dataset(examples, field_list)


def create_tabular_dataset(fields, file_format, file_path, use_dict,
                           sort_key=None):
    skip_header = file_format in {"csv", "tsv"} and not use_dict

    return TabularDataset(file_path, file_format, fields,
                          skip_header=skip_header, sort_key=sort_key)


def test_attribute_error(data, field_list):
    dataset = create_dataset(data, field_list)
    with pytest.raises(AttributeError):
        dataset.non_existent_attribute


def test_attribute_iteration(data, field_list):
    dataset = create_dataset(data, field_list)

    i = 0
    for x, y in zip(dataset.text, TEXT):
        assert x[0] == y
        i += 1

    assert len(TEXT) == i


def test_unpack_fields():
    field1 = Field("field1")
    field2 = Field("field2")
    field3 = Field("field3")

    output_fields = field1, field2
    mo_field = MultioutputField(output_fields)

    assert unpack_fields([field1, field2]) == [field1, field2]
    assert unpack_fields([field1, (field2, field3)]) == [field1, field2, field3]
    assert unpack_fields([field3, mo_field]) == [field3, field1, field2]

    field_dict = {
        "1": field3,
        "2": mo_field
    }

    unpacked_fields = unpack_fields(field_dict)

    assert len(unpacked_fields) == 3
    assert all(f in unpacked_fields for f in (field1, field2, field3))


def test_eager_tokenization():

    def create_dataset():

        fields = (
            Field("text", numericalizer=Vocab()),
            Field("source", numericalizer=Vocab(), tokenizer=list)
        )
        example_factory = ExampleFactory(fields)

        examples = [example_factory.from_list(data)
                    for data
                    in zip(TABULAR_TEXT, TABULAR_SOURCES)]

        dataset = Dataset(examples, fields)
        return dataset

    dataset_lazy = create_dataset()
    dataset_eager = create_dataset()

    for example_eager in dataset_eager:
        assert example_eager.text_ is None
        assert example_eager.source_ is None

    dataset_eager.finalize_fields()
    # Numericalize eagerly
    dataset_eager.numericalize_examples()

    dataset_lazy.finalize_fields()
    # Numericalize Lazily
    for _ in Iterator(dataset_lazy, 100):
        pass

    for example_eager, example_lazy in zip(dataset_eager, dataset_lazy):
        assert example_eager.text_ is not None
        assert all(example_eager.text_ == example_lazy.text_)

        assert example_eager.source_ is not None
        assert all(example_eager.source_ == example_lazy.source_)


@pytest.fixture
def hierarchical_dataset_fields():
    name_field = Field("name", keep_raw=True, tokenizer=None)
    number_field = Field("number", keep_raw=True, tokenizer=None)

    fields = {
        "name": name_field,
        "number": number_field
    }
    return fields


@pytest.fixture
def hierarchical_dataset_parser():
    return HierarchicalDataset.get_default_dict_parser("children")


@pytest.fixture
def hierarchical_dataset(hierarchical_dataset_fields, hierarchical_dataset_parser):
    return HierarchicalDataset.from_json(dataset=HIERARCHIAL_DATASET_JSON_EXAMPLE,
                                         fields=hierarchical_dataset_fields,
                                         parser=hierarchical_dataset_parser)


def test_create_hierarchical_dataset_from_json(hierarchical_dataset):
    root_nodes = hierarchical_dataset._root_nodes

    assert root_nodes[0].example.name[0] == "parent1"
    assert root_nodes[0].example.number[0] == 1

    assert root_nodes[1].example.name[0] == "parent2"
    assert root_nodes[1].example.number[0] == 5

    assert root_nodes[0].children[0].example.name[0] == "c11"
    assert root_nodes[0].children[0].example.number[0] == 2

    assert root_nodes[0].children[0].children[0].example.name[0] == "c111"
    assert root_nodes[0].children[0].children[0].example.number[0] == 3

    assert root_nodes[0].children[1].example.name[0] == "c12"
    assert root_nodes[0].children[1].example.number[0] == 4

    assert root_nodes[1].children[0].example.name[0] == "c21"
    assert root_nodes[1].children[0].example.number[0] == 6

    assert len(hierarchical_dataset) == 10
    assert hierarchical_dataset.depth == 2


def test_flatten_hierarchical_dataset(hierarchical_dataset):
    count = 0
    for index, example in enumerate(hierarchical_dataset.flatten()):
        assert example.number[0] == index + 1
        count += 1

    assert count == 10


def test_hierarchical_dataset_example_indexing(hierarchical_dataset):
    assert hierarchical_dataset[0].name[0] == "parent1"
    assert hierarchical_dataset[1].name[0] == "c11"
    assert hierarchical_dataset[2].name[0] == "c111"
    assert hierarchical_dataset[3].name[0] == "c12"
    assert hierarchical_dataset[4].name[0] == "parent2"
    assert hierarchical_dataset[5].name[0] == "c21"
    assert hierarchical_dataset[6].name[0] == "c22"
    assert hierarchical_dataset[7].name[0] == "c23"
    assert hierarchical_dataset[8].name[0] == "c231"
    assert hierarchical_dataset[9].name[0] == "c24"


def test_hierarchical_dataset_finalize_fields(hierarchical_dataset_parser):
    name_vocab = Vocab()
    number_vocab = Vocab()
    name_field = Field("name", keep_raw=True, tokenizer=None, numericalizer=name_vocab)
    number_field = Field("number", keep_raw=True, tokenizer=None,
                         numericalizer=number_vocab)

    fields = {
        "name": name_field,
        "number": number_field
    }
    dataset = HierarchicalDataset.from_json(dataset=HIERARCHIAL_DATASET_JSON_EXAMPLE,
                                            fields=fields,
                                            parser=hierarchical_dataset_parser)
    dataset.finalize_fields()
    assert name_vocab.finalized
    assert number_vocab.finalized


def test_hierarchical_dataset_invalid_json_fail(hierarchical_dataset_fields):
    with pytest.raises(JSONDecodeError):
        HierarchicalDataset.from_json(INVALID_JSON, hierarchical_dataset_fields,
                                      HierarchicalDataset
                                      .get_default_dict_parser("children"))


def test_hierarchical_dataset_json_root_element_not_list_fail(
        hierarchical_dataset_fields):
    with pytest.raises(ValueError):
        HierarchicalDataset.from_json(JSON_ROOT_NOT_LIST,
                                      hierarchical_dataset_fields,
                                      HierarchicalDataset.get_default_dict_parser(
                                          "children")
                                      )


def test_hierarchical_dataset_context_iteration(hierarchical_dataset):
    c111_expected_context = ["parent1", "c11"]
    c111_context = list(map(lambda x: x.name[0], hierarchical_dataset.get_context(2)))
    assert c111_context == c111_expected_context

    c23_expected_context_0_lvl = ["parent2", "c21", "c22"]
    c23_context_0_lvl = list(
        map(
            lambda x: x.name[0], hierarchical_dataset.get_context(7, 0)
        )
    )

    assert c23_context_0_lvl == c23_expected_context_0_lvl


def test_hierarchical_dataset_pickle(tmpdir, hierarchical_dataset):
    dataset_file = os.path.join(tmpdir, "dataset.pkl")

    with open(dataset_file, "wb") as fdata:
        dill.dump(hierarchical_dataset, fdata)

    with open(dataset_file, "rb") as fdata:
        loaded_dataset = dill.load(fdata)
        test_create_hierarchical_dataset_from_json(loaded_dataset)


HIERARCHIAL_DATASET_JSON_EXAMPLE = """
[
{
    "name" : "parent1",
    "number" : 1,
    "children" : [
        {
            "name" : "c11",
            "number" : 2,
            "children" : [
                {
                    "name" : "c111",
                    "number" : 3
                }
            ]
        },
        {
            "name" : "c12",
            "number" : 4
        }
    ]
},
{
    "name" : "parent2",
    "number" : 5,
    "children" : [
        {
            "name" : "c21",
            "number" : 6
        },
        {
            "name" : "c22",
            "number" : 7,
            "children" : []
        },
        {
            "name" : "c23",
            "number" : 8,
            "children" : [
                {
                    "name" : "c231",
                    "number" : 9
                }
            ]
        },
        {
            "name" : "c24",
            "number" : 10
        }
    ]
}
]
"""

INVALID_JSON = """
[
{
    "name" : "parent1",
    "number" : 1,
    "children" : [
        {
            "name" : "c11",
            "number" : 2,
            "children" : [
                {
                    "name" : "c111",
                    "number" : 3,
                    "children" : []
                }
            ]
        },
"""

JSON_ROOT_NOT_LIST = """
{
    "name" : "parent1",
    "number" : 1,
    "children" : [
        {
            "name" : "c11",
            "number" : 2,
            "children" : [
                {
                    "name" : "c111",
                    "number" : 3,
                    "children" : []
                }
            ]
        }
        ]
}
"""

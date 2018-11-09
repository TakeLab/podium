import os
from collections import Counter

import pandas as pd

from takepod.storage.dataset import Dataset, TabularDataset
import pytest

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
    def __init__(self, name, eager, sequential=True):
        self.name = name
        self.eager = eager
        self.sequential = sequential

        self.finalized = False
        self.updated_count = 0

        self.use_vocab = True

    def preprocess(self, data):
        return (data, [data]) if self.sequential else (data, None)

    def update_vocab(self, raw, tokenized):
        assert not self.eager
        self.updated_count += 1

    def finalize(self):
        self.finalized = True

    def __repr__(self):
        return self.name


class MockExample:
    def __init__(self, fields, data):
        self.accessed = False

        for f, d in zip(fields, data):
            self.__setattr__(f.name, f.preprocess(d))

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
        [0.998, 0.001, 0.001],   # these are incorrect ratios because for the
        (0.998, 0.001, 0.001),   # given dataset they would result in some
        [0.999, 0.001],          # splits having 0 (the same ratios could be
        0.999,                   # valid with larger datasets)
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


def test_split_stratified_exception(data_for_stratified, field_list):
    dataset = create_dataset(data_for_stratified, field_list)

    # when field with the given name doesn't exist
    with pytest.raises(ValueError):
        dataset.split(split_ratio=0.5, stratified=True,
                      strata_field_name="NOT_label")


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


@pytest.fixture
def field_list():
    return [MockField(field_name, eager) for field_name, eager in FIELD_DATA]


@pytest.fixture()
def data():
    labels = range(1, len(TEXT) + 1)
    data = list(zip(TEXT, labels))

    return data


@pytest.fixture()
def data_for_stratified():
    labels = [1] * 3 + [0] * 9

    data = list(zip(TEXT, labels))

    return data


@pytest.fixture()
def tabular_dataset_fields(use_dict):
    TEXT = MockField('text', eager=True)
    CHARS = MockField('chars', eager=True)
    RATING = MockField('rating', sequential=False, eager=True)
    SOURCE = MockField('source', sequential=False, eager=False)

    if use_dict:
        fields = {"text": (TEXT, CHARS), "rating": RATING, "source": SOURCE}
    else:
        fields = [(TEXT, CHARS), RATING, SOURCE]

    return fields


@pytest.fixture()
def tabular_data():
    return {"text": TABULAR_TEXT, "rating": TABULAR_RATINGS,
            "source": TABULAR_SOURCES}


@pytest.fixture()
def file_path(tmpdir, file_format, tabular_data):
    # tmpdir is a default pytest fixture
    path = os.path.join(tmpdir, "sample." + file_format)

    if file_format == "csv":
        create_temp_csv(path, ",", tabular_data)
    elif file_format == "tsv":
        create_temp_csv(path, "\t", tabular_data)
    else:
        create_temp_json(path, tabular_data)

    yield path


def create_dataset(data, field_list):
    examples = [MockExample(field_list, d) for d in data]

    return Dataset(examples, field_list)


def create_tabular_dataset(fields, file_format, file_path, use_dict):
    skip_header = (file_format in {"csv", "tsv"}) and (not use_dict)

    return TabularDataset(file_path, file_format, fields,
                          skip_header=skip_header)


def create_temp_csv(path, delimiter, data):
    df = pd.DataFrame(data)
    df.to_csv(path, sep=delimiter, index=False)


def create_temp_json(path, data):
    df = pd.DataFrame(data)
    lines = (df.loc[i].to_json() + "\n" for i in df.index)

    with open(path, "w") as f:
        f.writelines(lines)
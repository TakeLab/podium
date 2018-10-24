import os

import pandas as pd

from takepod.storage.dataset import Dataset, TabularDataset
import pytest


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


class MockExample:
    def __init__(self, fields, data):
        self.accessed = False

        for f, d in zip(fields, data):
            self.__setattr__(f.name, f.preprocess(d))

    def __getattribute__(self, item):
        if item not in {"accessed", "__setattr__"}:
            self.accessed = True

        return super().__getattribute__(item)


@pytest.mark.parametrize(
    "data, fields_args",
    [
        (
                [
                    ["ovaj text", 1],
                    ["ovaj isto", 0],
                    ["ovaj isto takodjer", 0],
                ],
                [
                    ("text", True), ("label", False)
                ]
        ),
    ]
)
def test_finalize_fields(data, fields_args):
    fields = [MockField(*args) for args in fields_args]
    examples = [MockExample(fields, d) for d in data]

    d = Dataset(examples, fields)

    for f in fields:
        # before finalization, no field's dict was updated
        assert f.updated_count == 0
        assert not f.finalized

    d.finalize_fields()

    for f in fields:
        # during finalization, only non-eager field's dict should be updated
        assert f.updated_count == (len(data) if (not f.eager) else 0)

        # all fields should be finalized
        assert f.finalized

    fields = [MockField(*args) for args in fields_args]
    examples = [MockExample(fields, d) for d in data]

    d2 = Dataset(examples, fields)

    d2_train, d2_val, d2_test = d2.split([0.3334, 0.333, 0.333])

    # using only the train set to build the vocabs for non-eager fields
    d2.finalize_fields(d2_train)

    # only the train set examples should have been accessed
    # during finalization
    for example in d2_train.examples:
        assert example.accessed

    for example in d2_val.examples:
        assert not example.accessed

    for example in d2_test.examples:
        assert not example.accessed


@pytest.mark.parametrize(
    "data, fields, float_ratio, train_test_ratio, train_val_test_ratio",
    [
        (
                [
                    ["ovaj text", 1],
                    ["ovaj isto", 2],
                    ["ovaj isto takodjer", 3],
                    ["ovaj text", 4],
                    ["ovaj isto", 5],
                    ["ovaj isto takodjer", 6],
                    ["ovaj text", 7],
                    ["ovaj isto", 8],
                    ["ovaj isto takodjer", 9],
                ],
                [
                    MockField("text", True), MockField("label", False)
                ],
                0.62,
                [0.3, 0.9],
                (0.3, 0.5, 0.3),
        ),
    ]
)
def test_split_not_stratified(data, fields, float_ratio, train_test_ratio,
                              train_val_test_ratio):
    examples = [MockExample(fields, d) for d in data]

    d = Dataset(examples, fields)

    # test with float ratio
    train_d, test_d = d.split(float_ratio)

    expected_train_size = int(round(float_ratio * len(d.examples)))
    expected_test_size = len(d.examples) - expected_train_size

    assert len(train_d.examples) == expected_train_size
    assert len(test_d.examples) == expected_test_size

    # test with length=2 list ratio
    train_d, test_d = d.split(train_test_ratio)

    expected_train_size = int(round(
        (train_test_ratio[0] / sum(train_test_ratio)) * len(d.examples)
    ))
    expected_test_size = len(d.examples) - expected_train_size

    assert len(train_d) == expected_train_size
    assert len(test_d) == expected_test_size

    # test with length=3 list ratio
    train_d, val_d, test_d = d.split(train_val_test_ratio)
    expected_train_size = int(round(
        (train_val_test_ratio[0] / sum(train_val_test_ratio)) * len(d.examples)
    ))
    expected_test_size = int(round(
        (train_val_test_ratio[2] / sum(train_val_test_ratio)) * len(d.examples)
    ))
    expected_val_size = len(
        d.examples) - expected_train_size - expected_test_size

    assert len(train_d) == expected_train_size
    assert len(val_d) == expected_val_size
    assert len(test_d) == expected_test_size

    # test that the sets don't overlap
    train_label_set = set(map(lambda ex: ex.label[0], train_d.examples))
    val_label_set = set(map(lambda ex: ex.label[0], val_d.examples))
    test_label_set = set(map(lambda ex: ex.label[0], test_d.examples))

    assert not train_label_set.intersection(val_label_set)
    assert not train_label_set.intersection(test_label_set)
    assert not val_label_set.intersection(test_label_set)


@pytest.mark.parametrize(
    "data, fields, ratios",
    [
        (
                [
                    ["ovaj text", 1],
                    ["ovaj isto", 2],
                    ["ovaj isto takodjer", 3],
                    ["ovaj text", 4],
                    ["ovaj isto", 5],
                    ["ovaj isto takodjer", 6],
                    ["ovaj text", 7],
                    ["ovaj isto", 8],
                    ["ovaj isto takodjer", 9],
                ],
                [
                    MockField("text", True), MockField("label", False)
                ],
                [
                    [0.62, 0.2, 0.1, 0.7],
                    None,
                    1.5,
                ],
        ),
    ]
)
def test_split_wrong_ratio(data, fields, ratios):
    examples = [MockExample(fields, d) for d in data]

    d = Dataset(examples, fields)

    # all the ratios provided are wrong in their own way
    # (too many elements; wrong type; not in (0.0, 1.0))
    for ratio in ratios:
        with pytest.raises(ValueError):
            d.split(ratio)


def count_labels(dataset):
    label_counter = dict()

    for e in dataset.examples:
        # [0] because we want just the raw value of the label
        label = e.label[0]

        if label in label_counter:
            label_counter[label] += 1
        else:
            label_counter[label] = 1

    return label_counter


@pytest.mark.parametrize(
    "data, fields",
    [
        (
                [
                    ["ovaj text", 0],
                    ["ovaj isto", 1],
                    ["ovaj isto takodjer", 1],
                    ["ovaj text", 1],
                    ["ovaj isto", 0],
                    ["ovaj isto takodjer", 0],
                    ["ovaj text", 0],
                    ["ovaj isto", 0],
                    ["ovaj isto takodjer", 0],
                    ["ovaj isto ne", 0],
                    ["bla bla", 0],
                    ["a a a a ", 0],
                ],
                [
                    MockField("text", True),
                    MockField("label", False, sequential=False)
                ],
        ),
    ]
)
def test_split_stratified_ok(data, fields):
    # should split evenly
    split_ratio = [0.33334, 0.33333, 0.33333]

    examples = [MockExample(fields, d) for d in data]

    # we want a random state for which the regular (not stratified) split
    # creates a class imbalance, to show that the stratified split doesn't
    random_state = get_good_random_state(examples, fields, split_ratio)

    d = Dataset(examples, fields)

    train_d, val_d, test_d = d.split(split_ratio, stratified=True,
                                     random_state=random_state)

    train_label_counter = count_labels(train_d)
    val_label_counter = count_labels(val_d)
    test_label_counter = count_labels(test_d)

    all_labels = set(map(lambda e: e.label[0], d.examples))

    # stratified split preserves the percentage of each class in every split
    # if the splits are the same (1/3), then the number of examples with
    # the same class will be the same in all three splits
    for label in all_labels:
        assert train_label_counter[label] == val_label_counter[label]
        assert train_label_counter[label] == test_label_counter[label]
        assert val_label_counter[label] == test_label_counter[label]


def get_good_random_state(examples, fields, split_ratio):
    random_state = 1
    while True:
        d = Dataset(examples, fields)
        train_d, val_d, test_d = d.split(split_ratio, stratified=False,
                                         random_state=random_state)

        train_label_counter = count_labels(train_d)
        val_label_counter = count_labels(val_d)
        test_label_counter = count_labels(test_d)

        all_labels = set(map(lambda e: e.label[0], d.examples))

        # we want a situation where even though the train, val, test splits
        # are the same (1/3), the number of examples for the same label
        # varies (at least one of eq1, eq2 and eq3 doesn't hold)
        for label in all_labels:
            eq1 = (train_label_counter.get(label, 0) == val_label_counter.get(
                label, 0))
            eq2 = (train_label_counter.get(label, 0) == test_label_counter.get(
                label, 0))
            eq3 = (val_label_counter.get(label, 0) == test_label_counter.get(
                label, 0))

            if not (eq1 and eq2 and eq3):
                return random_state

        random_state += 1


@pytest.mark.parametrize(
    "data, fields, split_ratio, ",
    [
        (
                [
                    ["ovaj text", 0],
                    ["ovaj isto", 1],
                    ["ovaj isto takodjer", 1],
                    ["ovaj text", 1],
                    ["ovaj isto", 0],
                    ["ovaj isto takodjer", 0],
                    ["ovaj text", 0],
                    ["ovaj isto", 0],
                    ["ovaj isto takodjer", 0],
                    ["ovaj isto ne", 0],
                    ["bla bla", 0],
                    ["a a a a ", 0],
                ],
                [
                    MockField("text", True),
                    MockField("label", False, sequential=False)
                ],
                [0.334, 0.333, 0.333],
        ),
    ]
)
def test_split_stratified_exception(data, fields, split_ratio):
    examples = [MockExample(fields, d) for d in data]

    d = Dataset(examples, fields)

    # when field with the given name doesn't exist
    with pytest.raises(ValueError):
        d.split(split_ratio=0.5, stratified=True,
                strata_field_name="NOT_label")


def create_temp_csv(path, delimiter, data):
    df = pd.DataFrame(data)
    df.to_csv(path, sep=delimiter, index=False)


def create_temp_json(path, data):
    df = pd.DataFrame(data)
    lines = (df.loc[i].to_json() + "\n" for i in df.index)

    with open(path, "w") as f:
        f.writelines(lines)


@pytest.fixture()
def create_files(tmpdir, data):
    # tmpdir is a default pytest fixture

    paths = {format: os.path.join(tmpdir, "sample." + format) for format in
             {"csv", "tsv", "json"}}

    create_temp_csv(paths["csv"], ",", data)
    create_temp_csv(paths["tsv"], "\t", data)
    create_temp_json(paths["json"], data)

    yield paths


@pytest.mark.parametrize(
    "data",
    [
        (
                {
                    "text":
                        [
                            "odlicni cevapi",
                            "ma ful, \"odlicni\" cevapi ..",
                            "mozd malo prepikantni cevapi, al inace ok",
                            "nema veganskih cevapa..u kojem stoljecu zivimo?"
                        ],
                    "rating":
                        [1, 0, 1, 0],
                    "source":
                        [
                            "www.volimljuto.hr",
                            "www.mrzimljuto.hr",
                            "www.nekadminepaseljuto.hr",
                            "www.neamideje.hr",
                        ],
                    "should_ignore":
                        [
                            "a", "b", "c", "d"
                        ]
                }
        ),
    ]
)
def test_tabular_dataset_ok(data, create_files):
    # create file is a fixture (see above) that creates the csv, tsv
    # and json files and returns the paths as a dict
    file_paths = create_files

    # the possible combinations of format and use_dict
    combinations = [
        ("csv", True),
        ("csv", False),
        ("tsv", True),
        ("tsv", False),
        ("json", True)
    ]

    for format, use_d in combinations:
        data_path = file_paths[format]

        TEXT = MockField('text', eager=True)
        CHARS = MockField('chars', eager=True)
        RATING = MockField('rating', sequential=False, eager=True)
        SOURCE = MockField('source', sequential=False, eager=False)
        SHOULD_IGNORE = None

        if use_d:
            fields = {"text": (TEXT, CHARS), "rating": RATING,
                      "source": SOURCE}
        else:
            fields = [(TEXT, CHARS), RATING, SOURCE, SHOULD_IGNORE]

        skip_header = (format in {"csv", "tsv"}) and (not use_d)

        d = TabularDataset(data_path, format, fields, skip_header=skip_header)

        # SHOULD_IGNORE was ignored
        assert set(d.fields) == {TEXT, CHARS, RATING, SOURCE}

        d.sort_key = "d_sort_key"
        d.finalize_fields()
        d_train, d_test = d.split(split_ratio=0.5, shuffle=False)

        # the sort key should be passed from the original dataset
        assert d_train.sort_key == d.sort_key
        assert d_test.sort_key == d.sort_key

        # iterating over the dataset like iterating over examples
        field_name = "text"
        for i, example in enumerate(d_train):
            val = data[field_name][i]
            expected_data = val, [val]

            assert getattr(example, field_name) == expected_data

        # iterating over examples
        field_name = "text"
        for i, example in enumerate(d_test.examples):
            index = len(d_train) + i
            val = data[field_name][index]
            expected_data = val, [val]

            assert getattr(example, field_name) == expected_data

        # iterating over a single field of a dataset
        field_val_expected_data_tuples = zip(getattr(d_train, field_name),
                                             data[field_name])

        for i, (field_value, expected_data) in enumerate(
                field_val_expected_data_tuples):

            assert field_value == (expected_data, [expected_data])

        # can get an example by its index in the dataset
        val = data[field_name][0]
        expected_data = val, [val]
        assert getattr(d_train[0], field_name) == expected_data


@pytest.mark.parametrize(
    "data",
    [
        (
                {
                    "text":
                        [
                            "odlicni cevapi",
                            "ma ful, \"odlicni\" cevapi ..",
                            "mozd malo prepikantni cevapi, al inace ok",
                            "nema veganskih cevapa..u kojem stoljecu zivimo?"
                        ],
                    "rating":
                        [1, 0, 1, 0],
                    "source":
                        [
                            "www.volimljuto.hr",
                            "www.mrzimljuto.hr",
                            "www.nekadminepaseljuto.hr",
                            "www.neamideje.hr",
                        ]
                }
        ),
    ]
)
def test_tabular_dataset_exception(data, create_files):
    file_paths = create_files

    format_use_dict_combinations = [
        ("csv", True),
        ("csv", False),
        ("tsv", True),
        ("tsv", False),
        ("json", True)
    ]

    for format, use_d in format_use_dict_combinations:
        data_path = file_paths[format]

        TEXT = MockField('text', eager=True)
        CHARS = MockField('chars', eager=True)
        RATING = MockField('rating', sequential=False, eager=True)
        SOURCE = MockField('source', sequential=False, eager=False)

        if use_d:
            fields = {"text": (TEXT, CHARS), "rating": RATING,
                      "source": SOURCE, }
        else:
            fields = [(TEXT, CHARS), RATING, SOURCE]

        # skip_header True when using a dict
        if use_d:
            with pytest.raises(ValueError):
                TabularDataset(data_path, format, fields, skip_header=True)

        # wrong format given
        with pytest.raises(ValueError):
            TabularDataset(data_path, "wrong_format", fields)

        td = TabularDataset(data_path, format, fields)

        # accessing a non-existing field
        with pytest.raises(AttributeError):
            next(getattr(td, "not_text"))

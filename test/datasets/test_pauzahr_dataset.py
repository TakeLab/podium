import os
import tempfile
import pytest
from takepod.datasets.pauza_dataset import PauzaHRDataset
from takepod.storage.dataset import Dataset

TRAIN_EXAMPLES = [{"Text": r"Izvrstan, ogroman Zagrebački,"
                           r" dostava na vrijeme, ljubazno osoblje ...",
                   "Rating": r"6",
                   "Source": r"http://pauza.hr/menu/pronto-pizza"},
                  {"Text": r"Lazanje su bile uzasne ali sis cevapi su bili"
                           r" odlicni tako da je ocjena za lazanje 2/6"
                           r"a cevape 6/6.",
                   "Rating": r"3.5",
                   "Source": r"http://pauza.hr/menu/pronto-pizza"},
                  {"Text": r"dostava kasnila 20ak minuta od predvidjenog,iako"
                           r" je nedjelja (?nema prometnih guzvi,mozda imaju  "
                           r"vise narudzbi ili nesto..)umjesto pohanog sira s "
                           r"kroketima dobila s pommesom :(ostalo sve ok",
                   "Rating": r"4.5",
                   "Source": r"http://pauza.hr/menu/pronto-pizza"}]

TEST_EXAMPLES = [{"Text": r"Izvrstan, ogroman Zagrebački,"
                          r" dostava na vrijeme, ljubazno osoblje ...",
                  "Rating": r"6",
                  "Source": r"http://pauza.hr/menu/pronto-pizza"},
                 {"Text": r"Fina hrana i zasigurno najpošteniji restoran u"
                          r" Zagrebu!Primjer: Naručila sam jelo kojeg je"
                          r" ponestalo. U kratkom su me roku kontaktirali i"
                          r" pitali želim li što drugo. Izabrala sam jelo"
                          r" skuplje od prvotno naručenog i naplaćena mi je "
                          r"cijena prve narudžbe uz porciju palačinki gratis."
                          r" Stalni sam klijent, a to ću i ostati.Svaka čast",
                  "Rating": r"6",
                  "Source": r"http://pauza.hr/menu/strossmayer"}
                 ]


@pytest.fixture(scope="module")
def mock_dataset_path():
    base_temp = tempfile.mkdtemp()
    assert os.path.exists(base_temp)
    base_dataset_dir = os.path.join(base_temp, "croopinion",
                                    "CropinionDataset", "reviews_original")

    train_dir = os.path.join(base_dataset_dir, "Train")
    os.makedirs(train_dir)
    assert os.path.exists(train_dir)
    test_dir = os.path.join(base_dataset_dir, "Test")
    os.makedirs(test_dir)
    assert os.path.exists(test_dir)

    create_examples(train_dir, TRAIN_EXAMPLES)
    create_examples(test_dir, TEST_EXAMPLES)
    return base_temp


def create_examples(base_dir, examples):
    for i in range(len(examples)):
        file_name = "comment{}.xml".format(i)
        create_mock_xml(base_dir=base_dir, file_name=file_name,
                        example=examples[i])


def create_mock_xml(base_dir, file_name, example):
    with open(file=os.path.join(base_dir, file_name),
              mode='w', encoding="utf8") as fp:
        fp.write('<?xml version="1.0" encoding="utf-8"?>\n')
        fp.write('<CrawlItem xmlns:xsi='
                 '"http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd='
                 '"http://www.w3.org/2001/XMLSchema">\n')
        fp.write('<Text>{}</Text>\n<Rating>{}</Rating>\n<Source>{}</Source>\n'
                 .format(example['Text'], example['Rating'], example['Source'])
                 )
        fp.write("</CrawlItem>")


def test_return_params(mock_dataset_path):
    data = PauzaHRDataset.get_train_test_dataset(mock_dataset_path)
    assert len(data) == 2
    assert isinstance(data[0], Dataset)
    assert isinstance(data[1], Dataset)


def test_default_fields():
    fields = PauzaHRDataset._get_default_fields()
    assert len(fields) == 3
    field_names = ["Text", "Rating", "Source"]
    assert all([name in fields for name in field_names])


def test_loaded_data(mock_dataset_path):
    data = PauzaHRDataset.get_train_test_dataset(mock_dataset_path)
    train_dataset, _ = data
    ratings = list(train_dataset.Rating)
    sources = list(train_dataset.Source)
    texts = list(train_dataset.Text)
    for ex in TRAIN_EXAMPLES:
        assert (ex["Rating"], None) in ratings
        assert (None, ex["Text"].split()) in texts
        assert (ex["Source"], None) in sources
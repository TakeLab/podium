import os
import tempfile
import pytest
from podium.datasets import Dataset
from podium.datasets.impl.squad_dataset import SQuADDataset
from podium.storage import LargeResource

TRAIN_EXAMPLES = [
    '{"version": "v2.0", "data": ['
    '{"title": "Beyoncé", "paragraphs":['
    '{"qas": ['
    '{"question": "When did Beyonce start becoming popular?",'
    '"id": "56be85543aeaaa14008c9063",'
    '"answers": ['
    '{"text": "in the late 1990s", "answer_start": 269}'
    '],'
    '"is_impossible": false}'
    '],'
    '"context": "Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born '
    'September 4, 1981) is an American singer, songwriter, '
    'record producer and actress. Born and raised in Houston, Texas, '
    'she performed in various singing and dancing competitions as a '
    'child, and rose to fame in the late 1990s as lead singer of R&B '
    'girl-group Destiny\'s Child. Managed by her father, Mathew Knowles, '
    'the group became one of the world\'s best-selling girl groups of all '
    'time. Their hiatus saw the release of Beyoncé\'s debut album, '
    'Dangerously in Love (2003), which established her as a solo artist '
    'worldwide, earned five Grammy Awards and featured the Billboard Hot '
    '100 number-one singles \\"Crazy in Love\\" and \\"Baby Boy\\"." '
    '}'
    ']}'
    ']}'
]

TRAIN_EXPECTED_EXAMPLES = [
    {"question": r"When did Beyonce start becoming popular?".split(),
     "answer": r"in the late 1990s".split(),
     "plausible_answer": r"".split(),
     "context": "Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born "
                "September 4, 1981) is an American singer, songwriter, record producer "
                "and actress. Born and raised in Houston, Texas, she performed in "
                "various singing and dancing competitions as a child, and rose to fame "
                "in the late 1990s as lead singer of R&B girl-group Destiny's Child. "
                "Managed by her father, Mathew Knowles, the group became one of the "
                "world's best-selling girl groups of all time. Their hiatus saw the "
                "release of Beyoncé's debut album, Dangerously in Love (2003), "
                "which established her as a solo artist worldwide, earned five Grammy "
                "Awards and featured the Billboard Hot 100 number-one singles \"Crazy "
                "in Love\" and \"Baby Boy\".".split(),
     "is_impossible": False},
]

DEV_EXAMPLES = [
    '{"version": "v2.0", "data": ['
    '{"title": "Normans", "paragraphs":['
    '{"qas": ['
    '{"question": "When were the Normans in Normandy?",'
    '"id": "56ddde6b9a695914005b9629",'
    '"answers": ['
    '{"text": "10th and 11th centuries", "answer_start": 94},'
    '{"text": "in the 10th and 11th centuries", "answer_start": 87}'
    '],'
    '"is_impossible": false},'
    '{"question": "Who gave their name to Normandy in the 1000\'s and 1100\'s",'
    '"id": "5ad39d53604f3c001a3fe8d1",'
    '"answers": [],'
    '"plausible_answers": ['
    '{"text": "Normans", "answer_start": 4}'
    '],'
    '"is_impossible": true}'
    '],'
    '"context": "The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) '
    'were the people who in the 10th and 11th centuries gave their name to Normandy, '
    'a region in France. They were descended from Norse (\\"Norman\\" comes from '
    '\\"Norseman\\") raiders and pirates from Denmark, Iceland and Norway who, '
    'under their leader Rollo, agreed to swear fealty to King Charles III of West '
    'Francia. Through generations of assimilation and mixing with the native Frankish '
    'and Roman-Gaulish populations, their descendants would gradually merge with the '
    'Carolingian-based cultures of West Francia. The distinct cultural and ethnic '
    'identity of the Normans emerged initially in the first half of the 10th century, '
    'and it continued to evolve over the succeeding centuries." '
    '}'
    ']}'
    ']}'
]

DEV_EXPECTED_EXAMPLES = [
    {"question": r"When were the Normans in Normandy?".split(),
     "answer": r"10th and 11th centuries".split(),
     "plausible_answer": r"".split(),
     "context": "The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) "
                "were the people who in the 10th and 11th centuries gave their name to "
                "Normandy, a region in France. They were descended from Norse ("
                "\"Norman\" comes from \"Norseman\") raiders and pirates from Denmark, "
                "Iceland and Norway who, under their leader Rollo, agreed to swear "
                "fealty to King Charles III of West Francia. Through generations of "
                "assimilation and mixing with the native Frankish and Roman-Gaulish "
                "populations, their descendants would gradually merge with the "
                "Carolingian-based cultures of West Francia. The distinct cultural and "
                "ethnic identity of the Normans emerged initially in the first half of "
                "the 10th century, and it continued to evolve over the succeeding "
                "centuries.".split(),
     "is_impossible": False},
    {"question": r"When were the Normans in Normandy?".split(),
     "answer": r"in the 10th and 11th centuries".split(),
     "plausible_answer": r"".split(),
     "context": "The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) "
                "were the people who in the 10th and 11th centuries gave their name to "
                "Normandy, a region in France. They were descended from Norse ("
                "\"Norman\" comes from \"Norseman\") raiders and pirates from Denmark, "
                "Iceland and Norway who, under their leader Rollo, agreed to swear "
                "fealty to King Charles III of West Francia. Through generations of "
                "assimilation and mixing with the native Frankish and Roman-Gaulish "
                "populations, their descendants would gradually merge with the "
                "Carolingian-based cultures of West Francia. The distinct cultural and "
                "ethnic identity of the Normans emerged initially in the first half of "
                "the 10th century, and it continued to evolve over the succeeding "
                "centuries.".split(),
     "is_impossible": False},
    {"question": r"Who gave their name to Normandy in the 1000's and 1100's".split(),
     "answer": r"".split(),
     "plausible_answer": r"Normans".split(),
     "context": "The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) "
                "were the people who in the 10th and 11th centuries gave their name to "
                "Normandy, a region in France. They were descended from Norse ("
                "\"Norman\" comes from \"Norseman\") raiders and pirates from Denmark, "
                "Iceland and Norway who, under their leader Rollo, agreed to swear "
                "fealty to King Charles III of West Francia. Through generations of "
                "assimilation and mixing with the native Frankish and Roman-Gaulish "
                "populations, their descendants would gradually merge with the "
                "Carolingian-based cultures of West Francia. The distinct cultural and "
                "ethnic identity of the Normans emerged initially in the first half of "
                "the 10th century, and it continued to evolve over the succeeding "
                "centuries.".split(),
     "is_impossible": True}
]


@pytest.fixture(scope="module")
def mock_dataset_path():
    base_temp = tempfile.mkdtemp()
    assert os.path.exists(base_temp)

    LargeResource.BASE_RESOURCE_DIR = base_temp

    train_dataset_path = os.path.join(base_temp, SQuADDataset.TRAIN_FILE_NAME)
    create_examples(train_dataset_path, TRAIN_EXAMPLES)

    dev_dataset_path = os.path.join(base_temp, SQuADDataset.DEV_FILE_NAME)
    create_examples(dev_dataset_path, DEV_EXAMPLES)

    return base_temp


def create_examples(file_name, raw_examples):
    with open(file=file_name, mode='w', encoding="utf8") as fpr:
        for example in raw_examples:
            fpr.write(example)


def test_return_params(mock_dataset_path):
    data = SQuADDataset.get_train_dev_dataset()
    assert len(data) == 2
    assert isinstance(data[0], Dataset)
    assert isinstance(data[1], Dataset)


def test_default_fields():
    fields = SQuADDataset.get_default_fields()
    assert len(fields) == 5
    field_names = [SQuADDataset.QUESTION_FIELD_NAME,
                   SQuADDataset.ANSWER_FIELD_NAME,
                   SQuADDataset.PLAUSIBLE_ANSWER_FIELD_NAME,
                   SQuADDataset.CONTEXT_FIELD_NAME,
                   SQuADDataset.IS_IMPOSSIBLE_FIELD_NAME]
    assert all([name in fields for name in field_names])
    assert fields[SQuADDataset.ANSWER_FIELD_NAME].is_target
    assert fields[SQuADDataset.PLAUSIBLE_ANSWER_FIELD_NAME].is_target
    assert not fields[SQuADDataset.QUESTION_FIELD_NAME].is_target
    assert not fields[SQuADDataset.CONTEXT_FIELD_NAME].is_target
    assert not fields[SQuADDataset.IS_IMPOSSIBLE_FIELD_NAME].is_target


def test_load_dataset(mock_dataset_path):
    train_dataset, dev_dataset = SQuADDataset.get_train_dev_dataset()
    assert isinstance(train_dataset, Dataset)
    assert isinstance(dev_dataset, Dataset)

    check_dataset(train_dataset, TRAIN_EXPECTED_EXAMPLES)
    check_dataset(dev_dataset, DEV_EXPECTED_EXAMPLES)


def check_dataset(dataset, expected_datasets):
    for ex in dataset:
        ex_data = {
            SQuADDataset.QUESTION_FIELD_NAME: ex.question[1],
            SQuADDataset.ANSWER_FIELD_NAME: ex.answer[1],
            SQuADDataset.PLAUSIBLE_ANSWER_FIELD_NAME: ex.plausible_answer[1],
            SQuADDataset.CONTEXT_FIELD_NAME: ex.context[1],
            SQuADDataset.IS_IMPOSSIBLE_FIELD_NAME: ex.is_impossible[0]
        }
        assert ex_data in expected_datasets

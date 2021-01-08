import os
import tempfile

import dill
import pytest

from podium.datasets import CatacxDataset


@pytest.fixture(scope="module")
def example_catacx_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "example.json")
        ds_file = open(os.path.join(tmpdir, "example.json"), mode="w")
        ds_file.write(EXAMPLE_CATACX_DATASET_JSON)
        ds_file.close()
        ds = CatacxDataset.load_from_file(file_path)

    return ds


def validate_example_catacx_dataset(dataset):
    root_nodes = dataset._root_nodes

    root_example_1 = root_nodes[0].example

    assert root_example_1["sentiment"][0] == -1.0
    assert root_example_1["likes_cnt"][0] == 0
    assert root_example_1["message"][1] == ["Example", "post", "No.", "1"]
    assert root_example_1["spam"][0] is False
    assert root_example_1["emotions"][1] == ["interest"]

    comment_1 = root_nodes[0].children[0].example

    assert comment_1["likes_cnt"][0] == 0
    assert comment_1["message"][1] == ["Comment", "No.1", "text"]
    assert comment_1["cs"][1] == ["answer"]
    assert comment_1["irony"] == (None, None)

    reply_1 = root_nodes[0].children[0].children[0].example

    assert reply_1["speech_acts"][1] == ["praising", "stating"]
    assert reply_1["irony"][0] is False
    assert reply_1["emotions"][1] == ["gratitude"]
    assert reply_1["pos_tags"][1] == ["pos_tag_1", "pos_tag_2", "pos_tag_3"]
    assert reply_1["lemmas"][1] == ["lemma_1", "lemma_2", "lemma_3"]
    assert reply_1["parent_ids"][1] == ["0", "1", "2"]
    assert reply_1["tokens"][1] == ["token_1", "token_2", "token_3"]
    assert reply_1["dependency_tags"][1] == ["dep_tag_1", "dep_tag_2", "dep_tag_3"]
    assert reply_1["id_tags"][1] == ["1", "2", "3"]
    assert reply_1["topics"][1] == ["opcenito podrska", "dijalog"]


def test_catacx_dataset_loading(example_catacx_dataset):
    validate_example_catacx_dataset(example_catacx_dataset)


def test_catacx_dataset_pickle(example_catacx_dataset, tmpdir):
    pickle_filename = tmpdir.join("catacx_dill.pkl")
    with open(pickle_filename, mode="wb") as file:
        dill.dump(example_catacx_dataset, file)

    with open(pickle_filename, mode="rb") as file:
        loaded_data = dill.load(file)

    validate_example_catacx_dataset(loaded_data)


EXAMPLE_CATACX_DATASET_JSON = """
[
    {
        "reactions": [],
        "sentiment": -1.0,
        "likes_cnt": 0,
        "message": "Example post No. 1",
        "spam": false,
        "emotions": [
            "interest"
        ],
        "sentences" : [[]],
        "comments":[
            {
                "replies": [
                    {
                        "speech_acts": [
                            "praising",
                            "stating"
                        ],
                        "sentiment": 0.666667,
                        "spam": false,
                        "irony": false,
                        "replies": [],
                        "emotions": [
                            "gratitude"
                        ],
                        "author_name": "reply_author_1",
                        "id": "1214916608526573_1214919761859591",
                        "likes_cnt": 0,
                        "smileys": [],
                        "likes": [],
                        "sentences": [
                            [
                                {
                                    "pos_tag": "pos_tag_1",
                                    "lemma": "lemma_1",
                                    "parent_id": "0",
                                    "token": "token_1",
                                    "dependency_tag": "dep_tag_1",
                                    "id": "1"
                                },
                                {
                                    "pos_tag": "pos_tag_2",
                                    "lemma": "lemma_2",
                                    "parent_id": "1",
                                    "token": "token_2",
                                    "dependency_tag": "dep_tag_2",
                                    "id": "2"
                                },
                                {
                                    "pos_tag": "pos_tag_3",
                                    "lemma": "lemma_3",
                                    "parent_id": "2",
                                    "token": "token_3",
                                    "dependency_tag": "dep_tag_3",
                                    "id": "3"
                                }
                            ]
                        ],
                        "created_time": "2016-07-16T10:46:07+0000",
                        "topics": [
                            "opcenito podrska",
                            "dijalog"
                        ],
                        "message": "reply_message_1",
                        "author_id": "10208153438006765"
                    },
                    {
                        "replies": [],
                        "author_name": "reply_author_2",
                        "id": "1214916608526573_1214926101858957",
                        "likes_cnt": 0,
                        "smileys": [],
                        "likes": [],
                        "created_time": "2016-07-16T10:55:38+0000",
                        "message": "reply_message_2",
                        "author_id": "503846199633621",
                        "cs": [
                            "satisfaction"
                        ],
                        "sentences" : [[]]
                    }
                ],
                "author_name": "Comment author No.1 name",
                "id": "Comment_No_1_id",
                "likes_cnt": 0,
                "smileys": [],
                "likes": [],
                "created_time": "2016-07-18T16:19:21+0000",
                "message": "Comment No.1 text",
                "author_id": "Comment_author_No_1_id",
                "cs": [
                    "answer"
                ],
                "sentences" : [[]]
            },
            {
                "replies": [],
                "author_name": "Comment author No.2 name",
                "id": "Comment_author_No_2_id",
                "likes_cnt": 0,
                "smileys": [],
                "likes": [],
                "created_time": "2016-07-18T16:19:21+0000",
                "message": "Comment No.2 text",
                "author_id": "Comment_author_No_2_id",
                "cs": [
                    "answer"
                ],
                "sentences" : [[]]
            }
        ]

    },
     {
        "reactions": [],
        "sentiment": -1.0,
        "likes_cnt": 0,
        "message": "Example post No. 2",
        "spam": false,
        "emotions": [
            "interest"
        ],
        "sentences" : [[]],
        "comments":[
            {
                "replies": [],
                "author_name": "Comment author No.3 name",
                "id": "Comment_No_3_id",
                "likes_cnt": 0,
                "smileys": [],
                "likes": [],
                "created_time": "2016-07-18T16:19:21+0000",
                "message": "Comment No.3 text",
                "author_id": "Comment_author_No_3_id",
                "cs": [
                    "answer"
                ],
                "sentences" : [[]]
            },
            {
                "replies": [],
                "author_name": "Comment author No.4 name",
                "id": "Comment_No_4_id",
                "likes_cnt": 0,
                "smileys": [],
                "likes": [],
                "created_time": "2016-07-18T16:19:21+0000",
                "message": "Comment No.4 text",
                "author_id": "Comment_author_No_4_id",
                "cs": [
                    "answer"
                ],
                "sentences" : [[]]
            }
        ]

    }
]
"""

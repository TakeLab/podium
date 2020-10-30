import os
import tempfile

from podium.datasets.impl.catacx_comments_dataset import CatacxCommentsDataset


def test_dataset_loading():
    tmpfile = tempfile.NamedTemporaryFile(mode="w", delete=False)
    tmpfile.write(SAMPLE_DATASET_RAW_JSON)
    tmpfile.close()

    dataset = CatacxCommentsDataset(tmpfile.name)
    os.remove(tmpfile.name)

    assert len(dataset) == 4

    assert dataset[0]["author_name"][0] == "Comment author No.1 name"
    assert dataset[0]["author_id"][0] == "Comment_author_No_1_id"
    assert dataset[0]["message"][1] == "Comment No.1 text".split()
    assert dataset[0]["likes_cnt"][0] is None
    assert dataset[0]["id"][0] == "Comment_No_1_id"

    assert dataset[2]["author_name"][0] == "Comment author No.3 name"
    assert dataset[2]["author_id"][0] == "Comment_author_No_3_id"
    assert dataset[2]["message"][1] == "Comment No.3 text".split()
    assert dataset[2]["likes_cnt"][0] is None
    assert dataset[2]["id"][0] == "Comment_No_3_id"

    ex = dataset[0]

    assert "replies" not in ex
    assert "smileys" not in ex
    assert "likes" not in ex
    assert "sentences" not in ex
    assert "created_time" not in ex
    assert "cs" not in ex


SAMPLE_DATASET_RAW_JSON = """[
    {
        "reactions": [],
        "sentiment": -1.0,
        "likes_cnt": 0,
        "message": "Example post No. 1",
        "spam": false,
        "emotions": [
            "interest"
        ],
        "comments":[
            {
                "replies": [],
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
                ]
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
                ]
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
        "comments":[
            {
                "replies": [],
                "author_name": "Comment author No.3 name",
                "id": "Comment_No_3_id",
                "likes_cnt": 2,
                "smileys": [],
                "likes": [],
                "created_time": "2016-07-18T16:19:21+0000",
                "message": "Comment No.3 text",
                "author_id": "Comment_author_No_3_id",
                "cs": [
                    "answer"
                ]
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
                ]
            }
        ]

    }
]"""

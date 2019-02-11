from takepod.datasets.catacx_comments_dataset import CatacxCommentsDataset
import os
import tempfile
from takepod.storage.field import Field


def get_default_fields():
    """
    Method returns a dict of default Catacx comment fields.
    fields : likes_cnt, id, likes_cnt, message


    Returns
    -------
    fields : dict(str, Field)
        dict containing all default Catacx fields
    """
    author_name = Field(name='author_name', sequential=False)

    id = Field(name='id', sequential=False)

    likes_cnt = Field(name="likes_cnt", vocab=None,
                      sequential=False,
                      custom_numericalize=int)
    message = Field(name='message', sequential=True, store_raw=True,
                    tokenizer='split', language='hr')

    author_id = Field(name='author_id', sequential=False)

    return {
        "author_name": author_name,
        "author_id": author_id,
        "id": id,
        "likes_cnt": likes_cnt,
        "message": message
    }


def test_dataset_loading():
    tmpfile = tempfile.NamedTemporaryFile(mode="w", delete=False)
    tmpfile.write(SAMPLE_DATASET_RAW_JSON)
    tmpfile.close()

    fields = get_default_fields()
    dataset = CatacxCommentsDataset(tmpfile.name, fields=fields)
    os.remove(tmpfile.name)

    assert dataset[0].author_name[0] == "Comment author No.1 name"
    assert dataset[0].author_id[0] == "Comment_author_No_1_id"
    assert dataset[0].message[0] == "Comment No.1 text"
    assert dataset[0].likes_cnt[0] == 0
    assert dataset[0].id[0] == "Comment_No_1_id"

    assert dataset[2].author_name[0] == "Comment author No.3 name"
    assert dataset[2].author_id[0] == "Comment_author_No_3_id"
    assert dataset[2].message[0] == "Comment No.3 text"
    assert dataset[2].likes_cnt[0] == 2
    assert dataset[2].id[0] == "Comment_No_3_id"


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

from takepod.datasets import CornellMovieDialogsConversationalDataset


def test_default_fields():
    fields = CornellMovieDialogsConversationalDataset.get_default_fields()
    assert len(fields) == 2
    field_names = ["statement", "reply"]
    assert all([name in fields for name in field_names])


def default_dataset():
    pass
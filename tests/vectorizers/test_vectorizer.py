import contextlib
import os
import shutil
import tempfile

import numpy as np
import pytest

from podium.vectorizers import vectorizer
from podium.vectorizers.impl import GloVe


BASIC_VECT_HEADING = b"251518 300"
BASIC_VECT_DATA = [
    b". 0.001134 -0.000058 -0.000668\n",
    b"' -0.079721 -0.074874 -0.184826\n",
    b": -0.144256 -0.169637 -0.044801\n",
    b", -0.031469 -0.107289 -0.182500\n",
]

BASIC_VECT_DATA_PLAIN = [
    ". 0.001134 -0.000058 -0.000668\n",
    "' -0.079721 -0.074874 -0.184826\n",
    ": -0.144256 -0.169637 -0.044801\n",
    ", -0.031469 -0.107289 -0.182500\n",
]

BASIC_VECT_DATA_DICT = {
    ".": np.array([0.001134, -0.000058, -0.000668]),
    "'": np.array([-0.079721, -0.074874, -0.184826]),
    ":": np.array([-0.144256, -0.169637, -0.044801]),
    ",": np.array([-0.031469, -0.107289, -0.182500]),
}

DIFF_DIM_VECT_DATA = [
    b". 0.001134 -0.000058 -0.000668\n",
    b"' -0.079721 -0.074874\n",
    b", -0.031469 -0.107289 -0.182500\n",
]


def test_base_class_abstract():
    with pytest.raises(TypeError):
        vectorizer.VectorStorage(path="/")


def test_basic_not_initialized():
    with create_temp_vect_file(
        vect_file_name="vect1", file_data=BASIC_VECT_DATA
    ) as vect_file_path:
        vect = vectorizer.WordVectors(path=vect_file_path)
        with pytest.raises(RuntimeError):
            vect["."]
        with pytest.raises(RuntimeError):
            vect.token_to_vector(".")


def test_basic_load_all_vectors():
    with create_temp_vect_file(
        vect_file_name="vect1", file_data=BASIC_VECT_DATA
    ) as vect_file_path:
        vect = vectorizer.WordVectors(path=vect_file_path)
        vect.load_all()
        assert len(vect._vectors) == 4
        assert vect["."].shape == (3,)
        assert vect.token_to_vector(",").shape == (3,)
        assert np.allclose(a=vect["."], b=BASIC_VECT_DATA_DICT["."], rtol=0, atol=1.0e-6)
        assert np.allclose(
            a=vect.token_to_vector(","), b=BASIC_VECT_DATA_DICT[","], rtol=0, atol=1.0e-6
        )


def test_get_vector_dimension():
    with create_temp_vect_file(
        vect_file_name="vect1", file_data=BASIC_VECT_DATA
    ) as vect_file_path:
        vect = vectorizer.WordVectors(path=vect_file_path)
        vect.load_all()
        assert vect.get_vector_dim() == vect["."].shape[0]
        assert vect.get_vector_dim() == 3


def test_get_vector_dim_not_initialized_vector_storage():
    with create_temp_vect_file(
        vect_file_name="vect1", file_data=BASIC_VECT_DATA
    ) as vect_file_path:
        vect = vectorizer.WordVectors(path=vect_file_path)
        with pytest.raises(RuntimeError):
            vect.get_vector_dim()


def test_basic_load_with_header():
    with create_temp_vect_file(
        vect_file_name="vect1", file_header=BASIC_VECT_HEADING, file_data=BASIC_VECT_DATA
    ) as vect_file_path:
        vect = vectorizer.WordVectors(path=vect_file_path)
        vect.load_all()
        assert len(vect._vectors) == 4
        assert vect["."].shape == (3,)
        assert vect.token_to_vector(",").shape == (3,)
        assert np.allclose(a=vect["."], b=BASIC_VECT_DATA_DICT["."], rtol=0, atol=1.0e-6)
        assert np.allclose(
            a=vect.token_to_vector(","), b=BASIC_VECT_DATA_DICT[","], rtol=0, atol=1.0e-6
        )


def test_basic_no_token():
    with create_temp_vect_file(
        vect_file_name="vect1", file_data=BASIC_VECT_DATA
    ) as vect_file_path:
        vect = vectorizer.WordVectors(path=vect_file_path, default_vector_function=None)
        vect.load_all()
        with pytest.raises(KeyError):
            print(vect["a"])
        with pytest.raises(KeyError):
            vect.token_to_vector("a")


def test_basic_token_none():
    with create_temp_vect_file(
        vect_file_name="vect1", file_data=BASIC_VECT_DATA
    ) as vect_file_path:
        vect = vectorizer.WordVectors(path=vect_file_path, default_vector_function=None)
        vect.load_all()
        with pytest.raises(ValueError):
            vect[None]
        with pytest.raises(ValueError):
            vect.token_to_vector(None)


def test_basic_token_default():
    with create_temp_vect_file(
        vect_file_name="vect1", file_data=BASIC_VECT_DATA
    ) as vect_file_path:
        vect = vectorizer.WordVectors(
            path=vect_file_path, default_vector_function=vectorizer.zeros_default_vector
        )
        vect.load_all()
        assert "a" not in vect._vectors
        assert vect["a"].shape == (3,)
        assert np.allclose(
            a=vect.token_to_vector("a"), b=np.zeros(3), rtol=0, atol=1.0e-6
        )


def test_basic_load_vocab():
    with create_temp_vect_file(
        vect_file_name="vect1", file_data=BASIC_VECT_DATA
    ) as vect_file_path:
        vect = vectorizer.WordVectors(path=vect_file_path, default_vector_function=None)
        vocab = [".", ":"]
        vect.load_vocab(vocab=vocab)
        assert len(vect._vectors) == 2
        assert vect["."].shape == (3,)
        assert vect.token_to_vector(":").shape == (3,)
        assert np.allclose(a=vect[":"], b=BASIC_VECT_DATA_DICT[":"], rtol=0, atol=1.0e-6)
        assert np.allclose(
            a=vect.token_to_vector("."), b=BASIC_VECT_DATA_DICT["."], rtol=0, atol=1.0e-6
        )

        with pytest.raises(KeyError):
            vect[","]
        with pytest.raises(KeyError):
            vect.token_to_vector(",")


def test_basic_load_vocab_none():
    with create_temp_vect_file(
        vect_file_name="vect1", file_data=BASIC_VECT_DATA
    ) as vect_file_path:
        vect = vectorizer.WordVectors(path=vect_file_path)
        with pytest.raises(ValueError):
            vect.load_vocab(vocab=None)


@pytest.mark.parametrize(
    "tokens, expected_matrix, expected_shape",
    [
        (["."], np.array(BASIC_VECT_DATA_DICT["."]), (1, 3)),
        (
            [",", ":", ".", "'"],
            np.array(
                [
                    BASIC_VECT_DATA_DICT[","],
                    BASIC_VECT_DATA_DICT[":"],
                    BASIC_VECT_DATA_DICT["."],
                    BASIC_VECT_DATA_DICT["'"],
                ]
            ),
            (4, 3),
        ),
    ],
)
def test_get_embedding_matrix(tokens, expected_matrix, expected_shape):
    with create_temp_vect_file(
        vect_file_name="vect1", file_data=BASIC_VECT_DATA
    ) as vect_file_path:
        vect = vectorizer.WordVectors(path=vect_file_path)
        vect.load_all()

        embedding_matrix = vect.get_embedding_matrix(vocab=tokens)
        assert embedding_matrix.shape == expected_shape
        assert np.allclose(a=embedding_matrix, b=expected_matrix, rtol=0, atol=1e-6)


def test_basic_diff_dimensions():
    with create_temp_vect_file(
        vect_file_name="vect1", file_data=DIFF_DIM_VECT_DATA
    ) as vect_file_path:
        vect = vectorizer.WordVectors(path=vect_file_path)
        with pytest.raises(RuntimeError):
            vect.load_all()


def test_default_vectors_zeros():
    vector = vectorizer.zeros_default_vector(token="token", dim=5)
    assert vector.shape == (5,)
    assert np.allclose(a=vector, b=np.zeros(5), rtol=0, atol=1.0e-6)


def test_default_vectors_zeros_none_arguments():
    with pytest.raises(ValueError):
        vectorizer.zeros_default_vector(token="token", dim=None)

    with pytest.raises(ValueError):
        vectorizer.zeros_default_vector(token=None, dim=None)


def test_basic_max_vectors_less_than_num_lines():
    with create_temp_vect_file(
        vect_file_name="vect1", file_data=BASIC_VECT_DATA
    ) as vect_file_path:
        vect = vectorizer.WordVectors(path=vect_file_path, max_vectors=2)
        vect.load_all()
        assert len(vect._vectors) == 2
        contained_elements = [".", "'"]
        assert all(elem in vect._vectors for elem in contained_elements)
        uncontained_elements = [":", ","]
        assert all(elem not in vect._vectors for elem in uncontained_elements)


def test_basic_max_vectors_vocab():
    with create_temp_vect_file(
        vect_file_name="vect1", file_data=BASIC_VECT_DATA
    ) as vect_file_path:
        vect = vectorizer.WordVectors(path=vect_file_path, max_vectors=2)
        vocab = [".", ":", ","]
        vect.load_vocab(vocab)
        assert len(vect._vectors) == 2
        contained_elements = [".", ":"]
        assert all(elem in vect._vectors for elem in contained_elements)
        uncontained_elements = ["'", ","]
        assert all(elem not in vect._vectors for elem in uncontained_elements)


def test_basic_max_vectors_bigger_than_num_lines():
    with create_temp_vect_file(
        vect_file_name="vect1", file_data=BASIC_VECT_DATA
    ) as vect_file_path:
        vect = vectorizer.WordVectors(path=vect_file_path, max_vectors=20)
        vect.load_all()
        assert len(vect._vectors) == 4
        contained_elements = [".", "'", ":", ","]
        assert all(elem in vect._vectors for elem in contained_elements)


def test_basic_both_paths_none():
    vect = vectorizer.WordVectors(path=None, cache_path=None)
    with pytest.raises(ValueError):
        vect.load_all()


def test_basic_both_paths_doesnt_exist(tmpdir):
    base = tmpdir
    assert os.path.exists(base)
    file_path = os.path.join(base, "file.t")
    assert not os.path.exists(file_path)
    cache_path = os.path.join(base, "cache.t")
    assert not os.path.exists(cache_path)

    vect = vectorizer.WordVectors(path=file_path, cache_path=cache_path)
    with pytest.raises(ValueError):
        vect.load_all()


def test_basic_path_none_cache_doesnt_exist(tmpdir):
    base = tmpdir
    assert os.path.exists(base)
    cache_path = os.path.join(base, "cache.t")
    assert not os.path.exists(cache_path)

    vect = vectorizer.WordVectors(path=None, cache_path=cache_path)
    with pytest.raises(ValueError):
        vect.load_all()


def test_basic_cache_max_vectors(tmpdir):
    with create_temp_vect_file(
        vect_file_name="vect1", file_data=BASIC_VECT_DATA, base_dir=tmpdir
    ) as vect_file_path:
        assert os.path.exists(vect_file_path)
        cache_path = os.path.join(tmpdir, "cache.t")
        assert not os.path.exists(cache_path)
        vect = vectorizer.WordVectors(
            path=vect_file_path, max_vectors=2, cache_path=cache_path
        )
        vect.load_all()
        assert os.path.exists(cache_path)
        with open(cache_path, "rb") as cache_file:
            content = cache_file.readlines()
            assert len(content) == 2
            first_line_parts = content[0].split(b" ")
            word, values = first_line_parts[0], first_line_parts[1:]
            assert word == b"."
            assert len(values) == 3


def test_basic_cache_vocab():
    with tempfile.TemporaryDirectory() as base:
        with create_temp_vect_file(
            vect_file_name="vect1", file_data=BASIC_VECT_DATA, base_dir=base
        ) as vect_file_path:
            assert os.path.exists(vect_file_path)
            cache_path = os.path.join(base, "cache.t")
            assert not os.path.exists(cache_path)
            vect = vectorizer.WordVectors(path=vect_file_path, cache_path=cache_path)

            vocab = [".", ":", ","]
            vect.load_vocab(vocab)

            assert os.path.exists(cache_path)

            with open(cache_path, "rb") as cache_file:
                content = cache_file.readlines()
                assert len(content) == 3


def test_load_plain_text():
    filename = "test.txt"
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, filename)
        with open(file_path, mode="w") as file:
            assert os.path.exists(file_path)
            file.writelines(BASIC_VECT_DATA_PLAIN)

        vec_storage = vectorizer.WordVectors(file_path, binary=False)
        vec_storage.load_all()

    assert len(vec_storage) == 4

    for token, vec in BASIC_VECT_DATA_DICT.items():
        assert np.all(vec == vec_storage[token])


@pytest.mark.parametrize(
    "name, dim",
    [
        ("not_valid_glove", 11),  # both name and dimension are invalid
        ("name_not_valid", 300),  # invalid name
        ("glove-wikipedia", 77),  # invalid dimension
    ],
)
def test_glove_wrong_params(name, dim):
    with pytest.raises(ValueError):
        GloVe(name=name, dim=dim)


@contextlib.contextmanager
def create_temp_vect_file(
    vect_file_name, file_data, file_header=None, base_dir=None, binary=True
):
    """
    Helper function that creates temporary vector file with given data.

    Parameters
    ----------
    vect_file_name : str
        temporary vector file file name
    file_data : byte str
        list of lines to be written
    file_header : byte str, optional
        header text, if None nothing is written
    base_dir : str
        path to base directory where to create vector file

    Returns
    -------
    vect_file_path : str
        path to the created vector file
    """
    try:
        is_base_created = False
        if base_dir is None:
            is_base_created = True
            base_dir = tempfile.mkdtemp()

        file_path = os.path.join(base_dir, vect_file_name)

        with open(file_path, "wb") as vect_file:
            if file_header is not None:
                vect_file.write(file_header + b"\n")
            vect_file.writelines(file_data)

        yield file_path
    finally:
        if is_base_created:
            shutil.rmtree(base_dir)

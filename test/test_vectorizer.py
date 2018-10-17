import pytest
from takepod.storage import vectorizer
import numpy as np
import tempfile
import os 


basic_vect_heading = b"251518 300"
basic_vect_data = [b". 0.001134 -0.000058 -0.000668\n",
                   b"' -0.079721 -0.074874 -0.184826\n",
                   b": -0.144256 -0.169637 -0.044801\n",
                   b", -0.031469 -0.107289 -0.182500\n"]
basic_vect_data_dict = {'.': np.array([0.001134, -0.000058, -0.000668]),
                        "'": np.array([-0.079721, -0.074874, -0.184826]),
                        ":": np.array([-0.144256, -0.169637, -0.044801]),
                        ",": np.array([-0.031469, -0.107289, -0.182500])}
diff_dim_vect_data = [b". 0.001134 -0.000058 -0.000668\n",
                      b"' -0.079721 -0.074874\n",
                      b", -0.031469 -0.107289 -0.182500\n"]

def test_base_class_abstract():
    with pytest.raises(TypeError):
        vectorizer.VectorStorage(path="/")


def test_basic_not_initialized():
    vect_file_path = create_temp_vect_file(vect_file_name="vect1",
                                           file_header=basic_vect_heading,
                                           file_data=basic_vect_data)
    vect = vectorizer.BasicVectorStorage(path=vect_file_path)
    with pytest.raises(RuntimeError):
        a = vect['.']
    with pytest.raises(RuntimeError):
        vect.token_to_vector('.')

def test_basic_load_all_vectors():
    vect_file_path = create_temp_vect_file(vect_file_name="vect1",
                                           file_header=basic_vect_heading,
                                           file_data=basic_vect_data)

    vect = vectorizer.BasicVectorStorage(path=vect_file_path)
    vect.load_all()
    assert vect['.'].shape == (3,)
    assert vect.token_to_vector(',').shape == (3,)
    assert np.allclose(a=vect['.'], b=basic_vect_data_dict['.'], rtol=0)
    assert np.allclose(a=vect.token_to_vector(','),
                       b=basic_vect_data_dict[','],
                       rtol=0)

def test_basic_no_token():
    vect_file_path = create_temp_vect_file(vect_file_name="vect1",
                                           file_header=basic_vect_heading,
                                           file_data=basic_vect_data)

    vect = vectorizer.BasicVectorStorage(path=vect_file_path)
    vect.load_all()
    with pytest.raises(KeyError):
        vect['a']
    with pytest.raises(KeyError):
        vect.token_to_vector('a')

def test_basic_token_none():
    vect_file_path = create_temp_vect_file(vect_file_name="vect1",
                                           file_header=basic_vect_heading,
                                           file_data=basic_vect_data)

    vect = vectorizer.BasicVectorStorage(path=vect_file_path)
    vect.load_all()
    with pytest.raises(ValueError):
        vect[None]
    with pytest.raises(ValueError):
        vect.token_to_vector(None)

def test_basic_token_default():
    vect_file_path = create_temp_vect_file(vect_file_name="vect1",
                                           file_header=basic_vect_heading,
                                           file_data=basic_vect_data)

    vect = vectorizer.BasicVectorStorage(path=vect_file_path,
                    default_vector_function=vectorizer.zeros_default_vector)
    vect.load_all()
    assert 'a' not in vect.vectors
    assert vect['a'].shape == (3,)
    assert np.allclose(a=vect.token_to_vector('a'),
                       b=np.zeros(3),
                       rtol=0)

def test_basic_load_vocab():
    vect_file_path = create_temp_vect_file(vect_file_name="vect1",
                                           file_header=basic_vect_heading,
                                           file_data=basic_vect_data)

    vect = vectorizer.BasicVectorStorage(path=vect_file_path)
    vocab = ['.', ':']
    vect.load_vocab(vocab=vocab)
    assert vect['.'].shape == (3,)
    assert vect.token_to_vector(':').shape == (3,)
    assert np.allclose(a=vect[':'], b=basic_vect_data_dict[':'], rtol=0)
    assert np.allclose(a=vect.token_to_vector('.'),
                       b=basic_vect_data_dict['.'],
                       rtol=0)

    with pytest.raises(KeyError):
        vect[',']
    with pytest.raises(KeyError):
        vect.token_to_vector(',')

def test_basic_load_vocab_none():
    vect_file_path = create_temp_vect_file(vect_file_name="vect1",
                                           file_header=basic_vect_heading,
                                           file_data=basic_vect_data)

    vect = vectorizer.BasicVectorStorage(path=vect_file_path)
    with pytest.raises(ValueError):
        vect.load_vocab(vocab=None)
    
def test_basic_diff_dimensions():
    vect_file_path = create_temp_vect_file(vect_file_name="vect1",
                                           file_header=None,
                                           file_data=diff_dim_vect_data)

    vect = vectorizer.BasicVectorStorage(path=vect_file_path)
    with pytest.raises(RuntimeError):
        vect.load_all()



def test_default_vectors_zeros():
    vector = vectorizer.zeros_default_vector(token="token", dim = 5)
    assert vector.shape == (5,)
    assert np.allclose(a=vector, b=np.zeros(5), rtol=0)

def test_default_vectors_zeros_none_arguments():
    with pytest.raises(ValueError):
        vector = vectorizer.zeros_default_vector(token="token", dim=None)

    with pytest.raises(ValueError):
        vector = vectorizer.zeros_default_vector(token=None, dim=5)

    with pytest.raises(ValueError):
        vector = vectorizer.zeros_default_vector(token=None, dim=None)
def test_basic_max_vectors_less_than_num_lines():
    vect_file_path = create_temp_vect_file(vect_file_name="vect1",
                                           file_header=None,
                                           file_data=basic_vect_data)

    vect = vectorizer.BasicVectorStorage(path=vect_file_path, max_vectors=2)
    vect.load_all()
    assert len(vect.vectors) == 2
    assert "." in vect.vectors
    assert "'" in vect.vectors
    assert ":" not in vect.vectors
    assert "," not in vect.vectors

def test_basic_max_vectors_vocab():
    vect_file_path = create_temp_vect_file(vect_file_name="vect1",
                                           file_header=None,
                                           file_data=basic_vect_data)

    vect = vectorizer.BasicVectorStorage(path=vect_file_path, max_vectors=2)
    vocab = ['.', ':', ","]
    vect.load_vocab(vocab)
    assert len(vect.vectors) == 2
    assert "." in vect.vectors
    assert "'" not in vect.vectors
    assert ":" in vect.vectors
    assert "," not in vect.vectors


def test_basic_max_vectors_bigger_than_num_lines():
    vect_file_path = create_temp_vect_file(vect_file_name="vect1",
                                           file_header=None,
                                           file_data=basic_vect_data)

    vect = vectorizer.BasicVectorStorage(path=vect_file_path, max_vectors=20)
    vect.load_all()
    assert len(vect.vectors) == 4
    assert "." in vect.vectors
    assert "'" in vect.vectors
    assert ":" in vect.vectors
    assert "," in vect.vectors

def test_basic_both_paths_none():
    vect_file_path = create_temp_vect_file(vect_file_name="vect1",
                                           file_header=None,
                                           file_data=basic_vect_data)

    vect = vectorizer.BasicVectorStorage(path=None, cache_path=None)
    with pytest.raises(ValueError):
        vect.load_all()

def test_basic_both_paths_doesnt_exist():
    base = tempfile.mkdtemp()
    assert os.path.exists(base)
    file_path = os.path.join(base, 'file.t')
    assert not os.path.exists(file_path)
    cache_path = os.path.join(base, 'cache.t')
    assert not os.path.exists(cache_path)

    vect = vectorizer.BasicVectorStorage(path=file_path, cache_path=cache_path)
    with pytest.raises(ValueError):
        vect.load_all()

def test_basic_path_none_cache_doesnt_exist():
    base = tempfile.mkdtemp()
    assert os.path.exists(base)
    cache_path = os.path.join(base, 'cache.t')
    assert not os.path.exists(cache_path)

    vect = vectorizer.BasicVectorStorage(path=None, cache_path=cache_path)
    with pytest.raises(ValueError):
        vect.load_all()

def test_basic_cache_max_vectors():
    base = tempfile.mkdtemp()
    assert os.path.exists(base)
    vect_file_path = create_temp_vect_file(vect_file_name="vect1",
                                           file_header=None,
                                           file_data=basic_vect_data,
                                           base_dir=base)
    assert os.path.exists(vect_file_path)
    cache_path = os.path.join(base, 'cache.t')
    assert not os.path.exists(cache_path)
    vect = vectorizer.BasicVectorStorage(path=vect_file_path,
                                         max_vectors=2,
                                         cache_path=cache_path)
    vect.load_all()
    assert os.path.exists(cache_path)
    with open(cache_path, 'rb') as cache_file:
        content = cache_file.readlines()
        assert len(content) == 2
        first_line_parts = content[0].split(b" ")
        word, values = first_line_parts[0], first_line_parts[1:]
        assert word == b"."
        assert len(values) == 3


def test_basic_cache_vocab():
    base = tempfile.mkdtemp()
    assert os.path.exists(base)
    vect_file_path = create_temp_vect_file(vect_file_name="vect1",
                                           file_header=None,
                                           file_data=basic_vect_data,
                                           base_dir=base)
    assert os.path.exists(vect_file_path)
    cache_path = os.path.join(base, 'cache.t')
    assert not os.path.exists(cache_path)
    vect = vectorizer.BasicVectorStorage(path=vect_file_path,
                                         cache_path=cache_path)

    vocab = ['.', ':', ","]
    vect.load_vocab(vocab)

    assert os.path.exists(cache_path)

    with open(cache_path, 'rb') as cache_file:
        content = cache_file.readlines()
        assert len(content) == 3

def create_temp_vect_file(vect_file_name, file_header, 
                          file_data, base_dir=None):
    if base_dir is None:
        base_dir = tempfile.mkdtemp()
        assert os.path.exists(base_dir)

    file_path = os.path.join(base_dir, vect_file_name)

    with open(file_path, 'wb') as vect_file:
        if file_header is not None:
            vect_file.write(file_header+b"\n")
        vect_file.writelines(file_data)

    assert os.path.exists(file_path)
    return file_path

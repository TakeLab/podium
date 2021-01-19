"""
Module contains example on how to use vectorizer class by using GloVe concrete
vectors class.
"""
from podium import Vocab
from podium.storage import LargeResource
from podium.vectorizers import GloVe


if __name__ == "__main__":
    LargeResource.BASE_RESOURCE_DIR = "downloaded_datasets"
    # we use vocab so that we don't need to load all vectors
    vocab = Vocab.from_itos(["frog", "load", "lizard", "company", "city"])

    vectorizer = GloVe()
    vectorizer.load_vocab(vocab=vocab)

    # token to vector
    token = "frog"
    vector = vectorizer.token_to_vector(token=token)
    print(token, vector)

    # embedding matrix
    embedding = vectorizer.get_embedding_matrix(vocab)
    print(vocab.itos)
    print(embedding)

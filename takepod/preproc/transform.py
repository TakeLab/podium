"""Module for transforming word to vector or index."""


def make_bow_vector(document, word_to_ix):
    '''
    Creates a bag-of-words vector from a document and
    word to index dict.

    Parameters
    ----------
    document : str
        Sentence to tokenize and transform each word
    word_to_ix : dict str => int
        Word to index dictionary

    Returns
    -------
    vec : list of int
        Vector of word-to-index indices, each element is
        an index lookup from word_to_ix
    '''

    vec = [0] * len(word_to_ix)
    # TODO split on proper tokenizer
    for word in document.split(' '):
        vec[word_to_ix[word]] += 1
    return vec


def create_word_to_index(data):
    '''
    From a list of text strings outputs a word to int dict

    Assumes splitting on words

    Parameters
    ----------
    data : list of str
        List of sentences to tokenize to words
        and created an index for

    Returns
    -------
    word_to_ix : dict str => int
        Word to index number.
        Each word has a unique index.

    '''
    word_to_ix = {}
    for document in data:
        # TODO: use a proper tokenizer and allow user-defined one
        for word in document.split(' '):
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    return word_to_ix


def categories_to_int(categories, mapping=None):
    '''
    Transforms categorial variables to numerical

    Parameters
    ----------
    categories : list of str
        List of categories
    '''
    if not mapping:
        mapping = create_word_to_index(categories)
    return [mapping[c] for c in categories]

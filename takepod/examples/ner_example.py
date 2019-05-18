"""Example how to use BLCC model on Croatian NER dataset for NER task."""

import sys
import logging
from collections import namedtuple
from functools import partial

import numpy as np

from takepod.datasets.croatian_ner_dataset import CroatianNERDataset
from takepod.metrics import multiclass_f1_metric
from takepod.models.blcc_model import BLCCModel
from takepod.models.simple_trainers import SimpleTrainer
from takepod.storage import TokenizedField, Vocab, SpecialVocabSymbols
from takepod.storage.iterator import BucketIterator
from takepod.storage.large_resource import LargeResource
from takepod.storage.vectorizer import BasicVectorStorage

_LOGGER = logging.getLogger(__name__)

max_word_length = 25

label_numericalized = {}

# using the same label set as original CroNER
label_mapping = {
    'Organization': 'Organization',
    'Person': 'Person',
    'Location': 'Location',
    'Date': 'Date',
    'Time': 'Time',
    'Money': 'Money',
    'Percent': 'Percent',
    'Etnic': 'Etnic',

    # remapped or unused types
    'PersonPossessive': 'Person',
    'Product': None,
    'OrganizationAsLocation': 'Organization',
    'LocationAsOrganization': 'Location'
}


def label_mapper_hook(data, tokens):
    """Function maps the labels to a reduced set."""
    for i in range(len(tokens)):
        if tokens[i] == 'O':
            continue

        prefix, value = tokens[i].split('-')

        mapped_token = label_mapping[value]
        if mapped_token is None:
            tokens[i] = 'O'
        else:
            tokens[i] = prefix + '-' +  mapped_token

    return data, tokens


def casing_mapper_hook(data, tokens):
    """Hook for generating the casing feature from the tokenized text."""
    tokens_casing = []

    for token in tokens:
        token_casing = 'other'
        if token.isdigit():
            token_casing = 'numeric'
        elif token.islower():
            token_casing = 'lowercase'
        elif token.isupper():
            token_casing = 'uppercase'
        elif token[0].isupper():
            token_casing = 'initial_uppercase'

        tokens_casing.append(token_casing)

    return data, tokens_casing


def token_to_characters(token):
    characters = list(token)

    if len(characters) > max_word_length:  # truncate
        return characters[:max_word_length]

    # pad
    padding_length = max_word_length - len(token)
    return characters + [SpecialVocabSymbols.PAD] * padding_length


def character_mapper_hook(data, tokens):
    characters = []
    for token in tokens:
        characters += token_to_characters(token)
    return data, characters


def batch_transform_fun(x_batch, y_batch, embedding_matrix):
    """Function transforms iterator batches to a form acceptable by
    the model."""
    # TODO remove casting to int after fixing the numericalization
    tokens_numericalized = x_batch.tokens.astype(int)
    casing_numericalized = x_batch.casing.astype(int)
    characters_numericalized = x_batch.characters.astype(int)
    y = y_batch.labels.astype(int)

    X = [
        np.take(embedding_matrix, tokens_numericalized, axis=0).astype(int),
        characters_numericalized,
        casing_numericalized
    ]
    return X, y


def example_word_count(example):
    """Function returns the number of tokens in an Example."""
    return len(example.tokens[1])


def print_to_file(inputs_str, expected_str, actual_str):
    value = ''

    for token, correct, guessed in zip(inputs_str, expected_str, actual_str):
        value += f'{token}\t{correct}\t{guessed}\n'

    with open('results.txt', 'w') as f:
        f.write(value)
        f.flush()


def ner_croatian_blcc_example(fields, dataset, batch_transform_function):
    """Example of training the BLCCModel with Croatian NER dataset"""
    output_size = len(fields['labels'].vocab.itos)
    casing_feature_size = len(fields['inputs'].casing.vocab.itos)
    characters_feature_size = len(fields['inputs'].characters.vocab.itos)

    train_set, test_set = dataset.split(split_ratio=0.8)

    train_iter = BucketIterator(train_set, 32, sort_key=example_word_count)
    test_iter = BucketIterator(test_set, 32, sort_key=example_word_count)

    model = BLCCModel(**{
        BLCCModel.OUTPUT_SIZE: output_size,
        BLCCModel.CLASSIFIER: 'CRF',
        BLCCModel.EMBEDDING_SIZE: 300,
        BLCCModel.LSTM_SIZE: (100, 100),
        BLCCModel.DROPOUT: (0.25, 0.25),
        BLCCModel.FEATURE_NAMES: ('casing',),
        BLCCModel.FEATURE_INPUT_SIZES: (casing_feature_size,),
        # set to a high value because of a tensorflow-cpu bug
        BLCCModel.FEATURE_OUTPUT_SIZES: (30,),

        BLCCModel.CHAR_EMBEDDINGS_ENABLED: True,
        BLCCModel.CHAR_EMBEDDINGS_NUM_OF_CHARS: characters_feature_size,
        BLCCModel.CHAR_EMBEDDINGS_INPUT_SIZE: 25
    })
    trainer = SimpleTrainer(model=model)

    _LOGGER.info('Training started')
    trainer.train(iterator=train_iter, **{
        trainer.MAX_EPOCH_KEY: 25,
        trainer.BATCH_TRANSFORM_FUN_KEY: batch_transform_function})
    _LOGGER.info('Training finished')

    pad_symbol = fields['labels'].vocab.pad_symbol()

    inputs = []
    actual = []
    expected = []

    for x_batch, y_batch in test_iter:
        x_test, y_test = batch_transform_function(x_batch, y_batch)

        prediction = model.predict(X=x_test)[BLCCModel.PREDICTION_KEY]
        prediction_filtered, y_test_filtered = filter_out_padding(
            pad_symbol,
            prediction,
            y_test
        )

        inputs.append(np.ravel(x_batch.tokens.astype(int)))
        actual.append(prediction_filtered)
        expected.append(y_test_filtered)

    inputs = np.concatenate(inputs)
    actual = np.concatenate(actual)
    expected = np.concatenate(expected)

    vocab_tokens = fields['inputs'].tokens.vocab.itos
    vocab_labels = fields['labels'].vocab.itos

    inputs_str = [vocab_tokens[x] for x in inputs
                  if vocab_tokens[x] != SpecialVocabSymbols.PAD]
    actual_str = [vocab_labels[x] for x in actual]
    expected_str = [vocab_labels[x] for x in expected]

    print_to_file(inputs_str, expected_str, actual_str)

    _LOGGER.info('Expected:')
    _LOGGER.info(expected)

    _LOGGER.info('Actual:')
    _LOGGER.info(actual)

    f1 = multiclass_f1_metric(
        expected,
        actual,
        average='weighted'
    )
    _LOGGER.info(f'F1: {f1}')


def filter_out_padding(pad_symbol, prediction, y_test):
    """Filters out padding from the predictiopytn and test arrays. The
     resulting arrays are flattened."""
    indices_to_leave = np.where(np.ravel(y_test) != pad_symbol)
    y_test_filtered = np.ravel(y_test)[indices_to_leave]
    prediction_filtered = np.ravel(prediction)[indices_to_leave]
    return prediction_filtered, y_test_filtered


def ner_dataset_classification_fields():
    """Function creates fields to use with the Croatian NER dataset on
    NER task."""
    tokens = TokenizedField(name='tokens',
                            vocab=Vocab())
    casing = TokenizedField(name='casing',
                            vocab=Vocab(specials=(SpecialVocabSymbols.PAD,)))
    characters = TokenizedField(name='characters',
                                prevent_padding=True,
                                vocab=Vocab(
                                    specials=(SpecialVocabSymbols.PAD,),
                                    keep_freqs=False))
    labels = TokenizedField(name='labels',
                            is_target=True,
                            vocab=Vocab(specials=(SpecialVocabSymbols.PAD,)))

    casing.add_posttokenize_hook(casing_mapper_hook)
    characters.add_posttokenize_hook(character_mapper_hook)
    labels.add_posttokenize_hook(label_mapper_hook)

    Inputs = namedtuple('Inputs', ['tokens', 'casing', 'characters'])

    return {'inputs': Inputs(tokens, casing, characters), 'labels': labels}


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    vectors_path = sys.argv[1]
    LargeResource.BASE_RESOURCE_DIR = 'downloaded_datasets'

    # global fields
    fields = ner_dataset_classification_fields()
    dataset = CroatianNERDataset.get_dataset(fields=fields)

    vectorizer = BasicVectorStorage(path=vectors_path)
    vectorizer.load_vocab(vocab=fields['inputs'].tokens.vocab)
    embedding_matrix = vectorizer.get_embedding_matrix(
        fields['inputs'].tokens.vocab)

    batch_transform = partial(
        batch_transform_fun,
        embedding_matrix=embedding_matrix)

    ner_croatian_blcc_example(fields=fields,
                              dataset=dataset,
                              batch_transform_function=batch_transform)

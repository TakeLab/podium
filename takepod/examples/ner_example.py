"""Example how to use BLCC model on Croatian NER dataset for NER task."""

import sys
import logging
from collections import namedtuple
from functools import partial
import pickle

import numpy as np

from takepod.datasets.impl.croatian_ner_dataset import CroatianNERDataset
from takepod.metrics import multiclass_f1_metric
from takepod.models.impl.blcc_model import BLCCModel
from takepod.models import FeatureTransformer
from takepod.models.impl.simple_trainers import SimpleTrainer
from takepod.storage import TokenizedField, Vocab, SpecialVocabSymbols
from takepod.datasets.iterator import BucketIterator
from takepod.storage.resources.large_resource import LargeResource
from takepod.storage.vectorizers.vectorizer import BasicVectorStorage

_LOGGER = logging.getLogger(__name__)

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
    new_tokens = []
    for i in range(len(tokens)):
        if tokens[i] == 'O':
            new_tokens.append('O')
            continue

        prefix, value = tokens[i].split('-')

        mapped_token = label_mapping[value]
        if not mapped_token:
            new_tokens.append('O')
        else:
            new_tokens.append(prefix + '-' + mapped_token)

    return data, new_tokens


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


def feature_extraction_fn(x_batch, embedding_matrix):
    """Function transforms iterator batches to a form acceptable by
    the model."""
    tokens_numericalized = x_batch.tokens.astype(int)
    casing_numericalized = x_batch.casing.astype(int)
    X = [
        np.take(embedding_matrix, tokens_numericalized, axis=0),
        casing_numericalized
    ]
    return X


def label_transform_fun(y_batch):
    return y_batch.labels.astype(int)


def example_word_count(example):
    """Function returns the number of tokens in an Example."""
    return len(example.tokens[1])


def ner_croatian_blcc_example(fields, dataset, feature_transform):
    """Example of training the BLCCModel with Croatian NER dataset"""
    output_size = len(fields['labels'].vocab.itos)
    casing_feature_size = len(fields['inputs'].casing.vocab.itos)

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
        BLCCModel.FEATURE_OUTPUT_SIZES: (30,)
    })
    trainer = SimpleTrainer()
    feature_transformer = FeatureTransformer(feature_transform)

    _LOGGER.info('Training started')
    trainer.train(
        model=model,
        iterator=train_iter,
        feature_transformer=feature_transformer,
        label_transform_fun=label_transform_fun,
        **{trainer.MAX_EPOCH_KEY: 1}
    )
    _LOGGER.info('Training finished')

    X_test_batch, y_test_batch = next(test_iter.__iter__())
    X_test = feature_transformer.transform(X_test_batch)
    y_test = label_transform_fun(y_test_batch)

    prediction = model.predict(X=X_test)[BLCCModel.PREDICTION_KEY]
    # pickle for later use
    pickle.dump(model, open('ner_model.pkl', 'wb'))

    pad_symbol = fields['labels'].vocab.pad_symbol()
    prediction_filtered, y_test_filtered = filter_out_padding(
        pad_symbol,
        prediction,
        y_test
    )

    _LOGGER.info('Expected:')
    _LOGGER.info(y_test_filtered)

    _LOGGER.info('Actual:')
    _LOGGER.info(prediction_filtered)

    f1 = multiclass_f1_metric(
        y_test_filtered,
        prediction_filtered,
        average='weighted'
    )
    info_msg = "F1: {}".format(f1)
    _LOGGER.info(info_msg)


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
    labels = TokenizedField(name='labels',
                            is_target=True,
                            vocab=Vocab(specials=(SpecialVocabSymbols.PAD,)))

    casing.add_posttokenize_hook(casing_mapper_hook)
    labels.add_posttokenize_hook(label_mapper_hook)

    Inputs = namedtuple('Inputs', ['tokens', 'casing'])

    return {'inputs': Inputs(tokens, casing), 'labels': labels}


if __name__ == '__main__':
    vectors_path = sys.argv[1]
    LargeResource.BASE_RESOURCE_DIR = 'downloaded_datasets'

    fields = ner_dataset_classification_fields()
    dataset = CroatianNERDataset.get_dataset(fields=fields)

    vocab = fields['inputs'].tokens.vocab
    embedding_matrix = BasicVectorStorage(path=vectors_path).load_vocab(vocab)

    feature_transform = partial(
        feature_extraction_fn,
        embedding_matrix=embedding_matrix)

    ner_croatian_blcc_example(
        fields=fields,
        dataset=dataset,
        feature_transform=feature_transform
    )

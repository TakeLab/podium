import sys
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


def batch_transform_fun(x_batch, y_batch, embedding_matrix):
    x_numericalized = x_batch.tokens.astype(int)
    X = np.take(embedding_matrix, x_numericalized, axis=0).astype(int)
    y = y_batch.labels.astype(int)
    return X, y


def label_mapper_hook(data, tokens):
    for i in range(len(tokens)):
        if tokens[i] == 'O':
            continue

        prefix = tokens[i][:2]
        value = tokens[i][2:]

        mapped_token = label_mapping[value]
        if mapped_token is None:
            tokens[i] = 'O'
        else:
            tokens[i] = prefix + mapped_token

    return data, tokens


def sort_by_word_count_key(example):
    return len(example.tokens[1])


def ner_croatian_blcc_example(fields, dataset, batch_transform_function):
    output_dim = len(fields['labels'].vocab.itos)

    train_set, test_set = dataset.split(split_ratio=0.8)

    train_iter = BucketIterator(train_set, 32, sort_key=sort_by_word_count_key)
    test_iter = BucketIterator(test_set, 32, sort_key=sort_by_word_count_key)

    model = BLCCModel(**{
        'output_dim': output_dim,
        'classifier': 'CRF',
        'embedding_size': 300,
        'LSTM-Size': [100, 100],
        'dropout': (0.25, 0.25)
    })
    trainer = SimpleTrainer(model=model)

    print('Training started')
    trainer.train(iterator=train_iter, **{
        trainer.MAX_EPOCH_KEY: 25,
        trainer.BATCH_TRANSFORM_FUN_KEY: batch_transform_function})
    print('Training finished')

    x_test, y_test = batch_transform_function(*next(test_iter.__iter__()))
    prediction = model.predict(X=x_test)

    pad_symbol = fields['labels'].vocab.pad_symbol()
    prediction_filtered, y_test_filtered = filter_out_padding(
        pad_symbol,
        prediction,
        y_test
    )

    print('Expected:')
    print(y_test_filtered)

    print('Actual:')
    print(prediction_filtered)

    print(f'F1: {multiclass_f1_metric(y_test_filtered, prediction_filtered)}')


def filter_out_padding(pad_symbol, prediction, y_test):
    indices_to_leave = np.where(np.ravel(y_test) != pad_symbol)
    y_test_filtered = np.ravel(y_test)[indices_to_leave]
    prediction_filtered = np.ravel(prediction)[indices_to_leave]
    return prediction_filtered, y_test_filtered


def ner_dataset_classification_fields():
    tokens = TokenizedField(name="tokens",
                            vocab=Vocab())
    labels = TokenizedField(name="labels",
                            is_target=True,
                            vocab=Vocab(specials=(SpecialVocabSymbols.PAD,)))

    labels.add_posttokenize_hook(label_mapper_hook)

    fields = {"tokens": tokens, "labels": labels}
    return fields


if __name__ == "__main__":
    vectors_path = sys.argv[1]
    LargeResource.BASE_RESOURCE_DIR = "downloaded_datasets"

    fields = ner_dataset_classification_fields()
    dataset = CroatianNERDataset.get_dataset(fields=fields)

    vectorizer = BasicVectorStorage(path=vectors_path)
    vectorizer.load_vocab(vocab=fields['tokens'].vocab)
    embedding_matrix = vectorizer.get_embedding_matrix(
        fields['tokens'].vocab)

    batch_transform = partial(
        batch_transform_fun,
        embedding_matrix=embedding_matrix)

    ner_croatian_blcc_example(fields=fields,
                              dataset=dataset,
                              batch_transform_function=batch_transform)

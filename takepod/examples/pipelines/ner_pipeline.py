"""Example how to use BLCC model on Croatian NER dataset for NER task."""

import sys
import logging
from functools import partial
import pickle

from takepod.dataload.ner_croatian import (
    convert_sequence_to_entities
)
from takepod.datasets.impl.croatian_ner_dataset import CroatianNERDataset
from takepod.models.impl.blcc_model import BLCCModel
from takepod.models import FeatureTransformer
from takepod.models.impl.simple_trainers import SimpleTrainer
from takepod.storage import ExampleFormat
from takepod.datasets.iterator import BucketIterator
from takepod.storage.resources.large_resource import LargeResource
from takepod.storage.vectorizers.vectorizer import BasicVectorStorage
from takepod.pipeline import Pipeline
from takepod.models import Experiment
from takepod.examples.ner_example import (
    feature_extraction_fn,
    label_transform_fun,
    example_word_count,
    ner_dataset_classification_fields
)


_LOGGER = logging.getLogger()


def map_iterable(iterable, mapping):
    return [
        mapping[i]
        for i in iterable
    ]


class Input:
    def __init__(self, items):
        self.items = items
        self.is_target = False

    def __iter__(self):
        for item in self.items:
            yield item


class CroatianNER:

    def __init__(self):
        self.pipeline = None

    def fit(self, vector_path):
        LargeResource.BASE_RESOURCE_DIR = 'downloaded_datasets'
        fields = ner_dataset_classification_fields()
        dataset = CroatianNERDataset.get_dataset(fields=fields)

        vectorizer = BasicVectorStorage(path=vector_path)
        vectorizer.load_vocab(vocab=fields['inputs'].tokens.vocab)
        embedding_matrix = vectorizer.get_embedding_matrix(
            fields['inputs'].tokens.vocab
        )
        feature_transform = partial(
            feature_extraction_fn,
            embedding_matrix=embedding_matrix)

        output_size = len(fields['labels'].vocab.itos)
        casing_feature_size = len(fields['inputs'].casing.vocab.itos)

        train_set, test_set = dataset.split(split_ratio=0.8)
        trainer = SimpleTrainer()
        feature_transformer = FeatureTransformer(feature_transform)

        _LOGGER.info('Training started')
        model_params = {
            BLCCModel.OUTPUT_SIZE: output_size,
            BLCCModel.CLASSIFIER: 'CRF',
            BLCCModel.EMBEDDING_SIZE: 300,
            BLCCModel.LSTM_SIZE: (20, 20),
            BLCCModel.DROPOUT: (0.25, 0.25),
            BLCCModel.FEATURE_NAMES: ('casing',),
            BLCCModel.FEATURE_INPUT_SIZES: (casing_feature_size,),
            # set to a high value because of a tensorflow-cpu bug
            BLCCModel.FEATURE_OUTPUT_SIZES: (30,)
        }
        experiment = Experiment(
            BLCCModel,
            trainer=trainer,
            feature_transformer=feature_transformer,
            label_transform_fn=label_transform_fun
        )
        iterator = BucketIterator(
            batch_size=32, sort_key=example_word_count
        )
        experiment.set_default_model_args(**model_params)
        trainer_args = {
            'iterator': iterator,
            'max_epoch': 2
        }
        experiment.set_default_trainer_args(**trainer_args)
        experiment.fit(train_set)

        # model has been fit and is ready to use
        dataset_fields = {
            "tokens":
                (
                    train_set.field_dict["tokens"],
                    train_set.field_dict["casing"]
                )
        }
        self.pipeline = Pipeline(
            dataset_fields, ExampleFormat.DICT,
            feature_transformer=feature_transformer,
            model=experiment.model,
            output_transform_fn=partial(
                map_iterable, mapping=train_set.field_dict["labels"].vocab.itos
            )
        )

    def save_pipeline(self, path):
        pickle.dump(self.pipeline, open(path, 'wb'))

    def load_pipeline(self, path):
        self.pipeline = pickle.load(open(path, 'rb'))

    def predict(self, text, tokenizer=str.split):
        tokenized_text = tokenizer(text)
        example_to_predict = {'tokens': tokenized_text}
        tags = self.pipeline.predict_raw(example_to_predict)
        return convert_sequence_to_entities(tags, tokenized_text)


if __name__ == '__main__':
    model_path = 'ner_pipeline_entire_model.pkl'
    ner = CroatianNER()
    # one can download from fasttext cc.hr.300.vec
    ner.fit(vector_path=sys.argv[1])
    ner.save_pipeline(model_path)

    ner = CroatianNER()
    ner.load_pipeline(model_path)
    ner.predict(
        "U Hrvatskoj državi žive mala bića . Velika bića žive u Graškogradu ."
    )

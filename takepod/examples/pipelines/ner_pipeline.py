"""Example how to use BLCC model on Croatian NER dataset for NER task."""

import sys
import logging
from collections import namedtuple
from functools import partial
import numpy as np
import pickle

from takepod.dataload.ner_croatian import (
    NERCroatianXMLLoader,
    convert_sequence_to_entities
)
from takepod.datasets.impl.croatian_ner_dataset import CroatianNERDataset
from takepod.metrics import multiclass_f1_metric
from takepod.models.impl.blcc_model import BLCCModel
from takepod.models import AbstractSupervisedModel, FeatureTransformer
from takepod.models.impl.simple_trainers import SimpleTrainer
from takepod.storage import TokenizedField, Vocab, SpecialVocabSymbols, ExampleFormat
from takepod.datasets.iterator import BucketIterator
from takepod.storage.resources.large_resource import LargeResource
from takepod.storage.vectorizers.vectorizer import BasicVectorStorage
from takepod.pipeline import Pipeline
from takepod.models import Experiment
from takepod.examples.ner_example import (
    label_mapper_hook,
    label_mapping,
    casing_mapper_hook,
    feature_extraction_fn,
    label_transform_fun,
    example_word_count,
    ner_dataset_classification_fields
)


_LOGGER = logging.getLogger(__name__)


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
        train_iterator_callable = partial(
            BucketIterator, batch_size=32, 
            sort_key=example_word_count
        )
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
            training_iterator_callable=train_iterator_callable,
            feature_transformer=feature_transformer,
            label_transform_fun=label_transform_fun
        )
        experiment.set_default_model_args(**model_params)
        train_params = {SimpleTrainer.MAX_EPOCH_KEY: 1}
        experiment.set_default_trainer_args(**train_params)
        experiment.fit(train_set)

        dataset_fields = {
            "tokens": 
            (
                train_set.field_dict["tokens"], 
                train_set.field_dict["casing"]
            ),
        }
        # model has been fit and is ready to use
        self.pipeline = Pipeline(
            dataset_fields, ExampleFormat.DICT,
            feature_transformer=feature_transformer,
            model=experiment.model,
            label_itos=train_set.field_dict["labels"].vocab.itos
        )

    def save_pipeline(self, path):
        pickle.dump(self.pipeline, open(path, 'wb'))

    def load_pipeline(self, path):
        self.pipeline = pickle.load(open(path, 'rb'))

    def predict(self, text, tokenizer=str.split):
        tokenized_text = tokenizer(text)
        example_to_predict = {'tokens': tokenized_text}
        tags_numericalized = self.pipeline.predict_raw(example_to_predict).ravel()
        tags = [
            self.pipeline.label_itos[tag]
            for tag in tags_numericalized
        ]
        return convert_sequence_to_entities(tags)

path = 'ner_pipeline_entire_model.pkl'
ner = CroatianNER()
ner.fit(vector_path="cc.hr.300.vec")
ner.predict(
    "U Hrvatskoj državi žive mala bića . Velika bića žive u Graškogradu ."
)
ner.save_pipeline(path)

ner = CroatianNER()
ner.load_pipeline(path)

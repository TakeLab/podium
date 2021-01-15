"""
Example how to use BLCC model on Croatian NER dataset for NER task.
"""

import os
import pickle
import sys
import time
from functools import partial

from ner_example import (
    example_word_count,
    feature_extraction_fn,
    label_transform_fun,
    ner_dataset_classification_fields,
)

from podium.dataload.ner_croatian import convert_sequence_to_entities
from podium.datasets.impl.croatian_ner_dataset import CroatianNERDataset
from podium.datasets.iterator import BucketIterator
from podium.models.impl.blcc_model import BLCCModel
from podium.models.impl.simple_trainers import SimpleTrainer
from podium.pipeline import Pipeline
from podium.storage import ExampleFormat
from podium.storage.resources.large_resource import LargeResource
from podium.storage.vectorizers.vectorizer import BasicVectorStorage


class CroatianNER(Pipeline):
    """
    Pipeline used to train named entity recognition for Croatian.

    It is designed to work on the croopinion dataset, but makes no assumptions
    on the underlying data, except that the data is tokenized before hand and
    labeled with BIO tags.
    """

    def __init__(self, vector_path):
        """
        Creates a new CroatianNER pipeline. Initializes fields and croopinion
        dataset. Defines feature transform (word vector lookup) and output
        transform (mapping labels back to BIO labels). Uses the ```BLCCModel```
        (BiLSTM + linear chain CRF) Expects examples in DICT field format.

        Parameters
        ----------
        vector_path : str
            Path to Croatian word vector file

        Raises
        ------
        ValueError
            If vector_path is None or is not a valid file path.
        """
        if vector_path is None or not os.path.exists(vector_path):
            raise ValueError(
                f"Provided path {vector_path} is None or does not exist. "
                "Path to word Croatian vectors must be defined. "
                "You can use fastText vectors available at "
                "https://fasttext.cc/docs/en/crawl-vectors.html"
            )

        self.fields = ner_dataset_classification_fields()
        self.dataset = CroatianNERDataset.get_dataset(fields=self.fields)
        self.feature_transform = self._define_feature_transform(vector_path)
        self.output_transform_fn = partial(
            CroatianNER.map_iterable, mapping=self.fields["labels"].vocab.itos
        )

        super().__init__(
            fields={
                # tokens is a feature field
                "tokens": self.fields["inputs"].tokens,
                # casing is a feature field
                "casing": self.fields["inputs"].casing,
                # labels is a target field
                "labels": self.fields["labels"],
            },
            example_format=ExampleFormat.DICT,
            model=BLCCModel,
            feature_transformer=self.feature_transform,
            output_transform_fn=self.output_transform_fn,
            label_transform_fn=label_transform_fun,
        )

    def fit(
        self,
        dataset,
        model_kwargs=None,
        trainer_kwargs=None,
        feature_transformer=None,
        trainer=None,
    ):
        """
        Fits the CroatianNER pipeline on a dataset using provided parameters.

        Parameters
        ----------
        dataset : Dataset
            Dataset to train on

        model_kwargs : dict
            model keyword argument dict forwarded to the model instance

        trainer_kwargs : dict
            trainer keyword argument dict forwarded to the trainer instance

        trainer : AbstractTrainer
            trainer instance (must inherit from `AbstractTrainer`)
        """

        if model_kwargs is None:
            model_kwargs = self._define_model_params()
            print(f"Using default model parameters {model_kwargs}")

        if trainer_kwargs is None:
            # use bucket iterator to minimize padding in batch
            iterator = BucketIterator(batch_size=32, sort_key=example_word_count)
            trainer_kwargs = {"max_epoch": 10, "iterator": iterator}
            print(f"Using default trainer parameters {trainer_kwargs}")

        trainer = SimpleTrainer() if trainer is None else trainer

        start = time.time()
        print("Starting training")
        super().fit(
            dataset=dataset,
            model_kwargs=model_kwargs,
            trainer_kwargs=trainer_kwargs,
            trainer=trainer,
        )
        print(f"Training took {time.time() - start} seconds")

    def predict_raw(self, raw_example, tokenizer=str.split):
        """
        Predicts target Fields for raw_example.

        Parameters
        ----------
        raw_example : str
            Sentence in Croatian.

        Returns
        ------
        list (dict)
            List of dicts of recognized named entities,
            where each dict has keys:
            name, type, start, end.
        """
        tokenized_text = tokenizer(raw_example)
        example_to_predict = {"tokens": tokenized_text, "casing": tokenized_text}
        tags = super().predict_raw(example_to_predict)
        return convert_sequence_to_entities(tags, tokenized_text)

    def _define_feature_transform(self, vector_path):
        vectorizer = BasicVectorStorage(path=vector_path)
        embedding_matrix = vectorizer.load_vocab(vocab=self.fields["inputs"].tokens.vocab)
        return partial(feature_extraction_fn, embedding_matrix=embedding_matrix)

    def _define_model_params(self):
        output_size = len(self.fields["labels"].vocab.itos)
        casing_feature_size = len(self.fields["inputs"].casing.vocab.itos)
        return {
            BLCCModel.OUTPUT_SIZE: output_size,
            BLCCModel.CLASSIFIER: "CRF",
            BLCCModel.EMBEDDING_SIZE: 300,
            BLCCModel.LSTM_SIZE: (20, 20),
            BLCCModel.DROPOUT: (0.25, 0.25),
            BLCCModel.FEATURE_NAMES: ("casing",),
            BLCCModel.FEATURE_INPUT_SIZES: (casing_feature_size,),
            # set to a high value because of a tensorflow-cpu bug
            BLCCModel.FEATURE_OUTPUT_SIZES: (30,),
        }

    @staticmethod
    def map_iterable(iterable, mapping):
        return [mapping[i] for i in iterable]


if __name__ == "__main__":
    model_path = "ner_pipeline_entire_model.pkl"
    LargeResource.BASE_RESOURCE_DIR = "downloaded_datasets"

    ner_pipeline = CroatianNER(sys.argv[1])
    ner_pipeline.fit(ner_pipeline.dataset)

    pickle.dump(ner_pipeline, open(model_path, "wb"))

    loaded_ner = pickle.load(open(model_path, "rb"))
    text = "U Hrvatskoj državi žive mala bića . Velika bića žive u Graškogradu ."
    print(loaded_ner.predict_raw(text))

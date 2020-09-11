# flake8: noqa
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from podium.datasets import IMDB, Iterator
from podium.models import Experiment
from podium.models.impl.pytorch import TorchTrainer, TorchModel
from podium.pipeline import Pipeline
from podium.storage import Field, LabelField, Vocab


def create_fields():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def text_to_tokens(string):
        input_ids = tokenizer(string,
                              max_length=128,
                              padding=False,
                              truncation=True,
                              return_attention_mask=False
                              )['input_ids']

        return tokenizer.convert_ids_to_tokens(input_ids)

    def token_to_input_id(token):
        return tokenizer.convert_tokens_to_ids(token)

    text = Field(name='text',
                 tokenizer=text_to_tokens,
                 custom_numericalize=token_to_input_id,
                 padding_token=0)

    label = LabelField(name='label', vocab=Vocab(specials=()))

    return {
        'text': text,
        'label': label
    }


class BertModelWrapper(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',
                                                                         return_dict=True)

    def forward(self, x):
        attention_mask = (x != 0).long()
        return_dict = self.model(x, attention_mask)
        return_dict['pred'] = return_dict['logits']
        return return_dict


def main():
    # loading the IMDB dataset
    fields = create_fields()
    imdb_train, imdb_test = IMDB.get_dataset_splits(fields)

    # setting up the experiment for fine-tuning the model
    model_config = {
        'lr': 1e-5,
        'clip': float('inf'),  # disable gradient clipping
        'num_epochs': 3,
    }
    model_config['num_classes'] = len(fields['label'].vocab)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    iterator = Iterator(batch_size=32)
    trainer = TorchTrainer(model_config['num_epochs'], device, iterator, imdb_test)

    # we have to swap axes to nullify the effect of swapping axes afterwards
    # because we work with the batch-first model (we should add this option to Podium!!!)
    feature_transformer = lambda feature_batch: feature_batch[0].astype(np.int64).swapaxes(0, 1)
    label_transformer = lambda label_batch: label_batch[0].astype(np.int64)

    experiment = Experiment(TorchModel,
                            trainer=trainer,
                            feature_transformer=feature_transformer,
                            label_transform_fn=label_transformer)

    experiment.fit(imdb_train,
                   model_kwargs={
                       'model_class': BertModelWrapper,
                       'criterion': nn.CrossEntropyLoss(),
                       'optimizer': optim.AdamW,
                       'device': device,
                       **model_config
                   })

    # utilities for saving/loading the model
    def save_model(model, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(fitted_model, f)

    def load_model(file_path):
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model

    fitted_model = experiment.model

    model_file = 'bert_model.pt'
    save_model(fitted_model, model_file)
    loaded_model = load_model(model_file)

    # here we show how to use the raw model to make predictions,
    # this is how you can use the model that is already fine-tuned
    cast_to_torch_transformer = lambda t: torch.from_numpy(t[0].astype(np.int64)).to(device)

    def make_predictions(raw_model, dataset):
        raw_model.eval()

        # we call `.batch()` on the dataset to get numericalized examples
        X, _ = dataset.batch()
        with torch.no_grad():
            predictions = raw_model(cast_to_torch_transformer(X))['pred']
            return predictions.cpu().numpy()

    raw_model = loaded_model.model
    predictions = make_predictions(raw_model, imdb_test[:4])

    _, y = imdb_test[:4].batch()
    y_pred = predictions.argmax(axis=1)
    print('y_pred == y_true:', (y_pred == y[0].ravel()).all())

    # we use `Pipeline` to make predictions on raw data
    pipe = Pipeline(fields=list(fields.values()),
                    example_format='list',
                    feature_transformer=cast_to_torch_transformer,
                    model=loaded_model)

    instances = [
        ['This movie is horrible'],
        ['This movie is great!']
    ]

    for instance in instances:
        predictions = pipe.predict_raw(instance)
        print(f'instance: {instance}, predicted label: '
              f'{fields["label"].vocab.itos[predictions.argmax()]}, '
              f'predictions: {predictions}')


if __name__ == '__main__':
    main()

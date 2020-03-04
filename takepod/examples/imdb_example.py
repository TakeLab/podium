import math
import time
import pickle
import takepod

from takepod.datasets import BucketIterator, Iterator, BasicSupervisedImdbDataset
from takepod.storage import Field, Vocab
from takepod.storage.vectorizers.impl import GloVe
from takepod.models import Experiment, AbstractSupervisedModel
from takepod.models.trainer import AbstractTrainer
from takepod.pipeline import Pipeline

from takepod.models.impl.sequence_classification import *

import torch
import torch.nn as nn
import torch.nn.functional as F

class Config(dict):
    def __init__(self, *args, **kwargs): 
        dict.__init__(self, *args, **kwargs)
            
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

def lowercase(raw, data):
    return raw, [d.lower() for d in data]

def max_length(raw, data, length=200):
    return raw, data[:length]

def create_fields():
    # Define the vocabulary
    vocab = Vocab(max_size=10000, min_freq=5)
    text = Field(name='text', vocab=vocab, tokenizer='spacy', store_as_raw=False)
    # Add preprpocessing hooks to model
    # 1. Lowercase
    text.add_posttokenize_hook(lowercase)
    text.add_posttokenize_hook(max_length)
    # Improve readability: LabelField
    label = Field(name='label', vocab=Vocab(specials=()), is_target=True, tokenize=False)
    return {text.name : text, label.name: label}

def main():
    fields = create_fields()
    imdb_train, imdb_test = BasicSupervisedImdbDataset.get_train_test_dataset(fields)

    # Construct vectoziter based on vocab
    vocab = fields['text'].vocab
    #embeddings = GloVe().load_vocab(vocab)

    criterion = nn.CrossEntropyLoss()
    label_vocab = fields['label'].vocab
    # Ugly but just to check
    config_dict = {
        'rnn_type': 'LSTM',
        'embed_dim': 300,
        'hidden_dim': 150,
        'nlayers': 1,
        'lr': 1e-3,
        'clip': 5,
        'epochs': 1,
        'batch_size': 32,
        'dropout': 0.,
        'bidirectional': True,
        'cuda': False,
        'vocab_size': len(vocab),
        'num_classes': len(label_vocab),
    }

    device = torch.device('cuda:0')
    config = Config(config_dict)

    data_iterator = Iterator(batch_size=32)

    trainer = TorchTrainer(config.epochs, device, data_iterator, imdb_test)

    experiment = Experiment(MyTorchModel, trainer=trainer)
    model = experiment.fit(
        imdb_train,
        model_kwargs={
            'model_class': AttentionRNN, 
            'config': config, 
            'criterion': criterion,
            'optimizer': torch.optim.Adam,
            'device': device
            #'embedding': embedding
        },
    )

    # Check serialization for _model_ only (should be for experiment as well)
    fitted_model = experiment.model

    model_save_file = 'model.pt'
    with open(model_save_file, 'wb') as dump_file:
      pickle.dump(fitted_model, dump_file)

    with open(model_save_file, 'rb') as load_file:
      loaded_model = pickle.load(load_file)

    ft = experiment.feature_transformer
    cast_to_torch_transformer = lambda t: torch.from_numpy(ft.transform(t).swapaxes(0,1)).to(device)

    pipe = Pipeline(
      fields = list(fields.values()),
      example_format = 'list',
      feature_transformer = cast_to_torch_transformer,
      model = fitted_model
      )

    prediction = pipe.predict_raw(['This movie is horrible'])
    print(prediction)

if __name__ == '__main__':
  main()
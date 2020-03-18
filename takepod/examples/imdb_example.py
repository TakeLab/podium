import math
import time
import takepod
import pickle

import torch
import torch.nn as nn
import numpy as np

from takepod.models import Experiment
from takepod.storage import LabelField, Field, Vocab
from takepod.pipeline import Pipeline
from takepod.datasets import IMDB, Iterator
from takepod.storage.vectorizers.impl import GloVe
from takepod.models.impl.pytorch import TorchTrainer, TorchModel, AttentionRNN


def lowercase(raw, tokenized):
    """Applies lowercasing as a post-tokenization hook
    
    Parameters
    ----------
    Raw : str
        the untokenized input data
    Tokenized: list(str)
        list of tokens.
    Returns
    -------
    Raw: str 
        unmodified input
    Tokenized: list(str) 
        lowercased tokenized data
    """
    return raw, [token.lower() for token in tokenized]


def max_length(raw, data, length=200):
    """Applies lowercasing as a post-tokenization hook
    
    Parameters
    ----------
    Raw : str
        the untokenized input data
    Tokenized: list(str)
        list of tokens.
    Length: int
        maximum length for each instance 
    Returns
    -------
    Raw: str 
        unmodified input
    Tokenized: list(str) 
        tokenized data truncated to `length`
    """
    return raw, data[:length]


def create_fields():
    # Define the vocabulary
    max_vocab_size = 10000
    min_frequency = 5
    vocab = Vocab(max_size=max_vocab_size, min_freq=min_frequency)

    text = Field(name='text', vocab=vocab, tokenizer='spacy', store_as_raw=False)
    # Add preprpocessing hooks to model
    # 1. Lowercase
    text.add_posttokenize_hook(lowercase)
    # 2. Truncate to length
    text.add_posttokenize_hook(max_length)

    label = LabelField(name='label', vocab = Vocab(specials=()))
    return {text.name : text, label.name: label}


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
    imdb_train, imdb_test = IMDB.get_dataset_splits(fields)

    # Construct vectoziter based on vocab
    vocab = fields['text'].vocab
    embeddings = GloVe().load_vocab(vocab)
    print(f"For vocabulary of size: {len(vocab)} loaded embedding matrix of shape: {embeddings.shape}")

    # First, we will define the hyperparameters for our model. 
    # These are only used when a concrete model is trained, and can be changed between calls.
    model_config = {
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
        'gpu': -1
    }

    # Task-specific metadata
    label_vocab = fields['label'].vocab
    model_config['num_classes'] = len(label_vocab)
    model_config['vocab_size'] = len(vocab)
    model_config['pretrained_embedding'] = embeddings
    # Run on CPU since we don't have a GPU on this machine
    device = torch.device('cpu:0')
    # Define the model criterion
    criterion = nn.CrossEntropyLoss()

    data_iterator = Iterator(batch_size=32)

    trainer = TorchTrainer(model_config['epochs'], device, data_iterator, imdb_test)
    experiment = Experiment(TorchModel, trainer=trainer)

    model = experiment.fit(
        imdb_train,  # Data on which to fit the model
        model_kwargs={  # Arguments passed to the model constructor
            'model_class': AttentionRNN,  # The wrapped concrete model
            'criterion': criterion,  # The loss for the concrete model
            'optimizer': torch.optim.Adam,  # Optimizer _class_
            'device': device,  # The device to store the data on
            **model_config  # Delegated to the concrete model
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

    instances = [
            ['This movie is horrible'], 
            ['This movie is great!']
    ]

    for instance in instances:
        prediction = pipe.predict_raw(instance)
        print(f"For instance: {instance}, the prediction is: "
              f"{fields['label'].vocab.itos[prediction.argmax()]},"
              f" with logits: {prediction}")

if __name__ == '__main__':
  main()
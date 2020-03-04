import math
import time

from takepod.datasets import BucketIterator, Iterator, BasicSupervisedImdbDataset
from takepod.storage import Field, Vocab
from takepod.storage.vectorizers.impl import GloVe
from takepod.models import Experiment, AbstractSupervisedModel
from takepod.models.trainer import AbstractTrainer

import torch
import torch.nn as nn
import torch.nn.functional as F

RNNS = ['LSTM', 'GRU']

class Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, nlayers=1, dropout=0.,
                   bidirectional=True, rnn_type='GRU'):
        super(Encoder, self).__init__()
        
        self.bidirectional = bidirectional
        assert rnn_type in RNNS, 'Use one of the following: {}'.format(str(RNNS))
        rnn_cell = getattr(nn, rnn_type) # fetch constructor from torch.nn, cleaner than if
        self.rnn = rnn_cell(embedding_dim, hidden_dim, nlayers, 
                                dropout=dropout, bidirectional=bidirectional)

    def forward(self, input, hidden=None):
        return self.rnn(input, hidden)


class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super(Attention, self).__init__()
        self.scale = 1. / math.sqrt(query_dim)

    def forward(self, query, keys, values):
        # Query = [BxQ]
        # Keys = [TxBxK]
        # Values = [TxBxV]
        # Outputs = a:[TxB], lin_comb:[BxV]

        # Here we assume q_dim == k_dim (dot product attention)

        query = query.unsqueeze(1) # [BxQ] -> [Bx1xQ]
        keys = keys.transpose(0,1).transpose(1,2) # [TxBxK] -> [BxKxT]
        energy = torch.bmm(query, keys) # [Bx1xQ]x[BxKxT] -> [Bx1xT]
        energy = F.softmax(energy.mul_(self.scale), dim=2) # scale, normalize

        values = values.transpose(0,1) # [TxBxV] -> [BxTxV]
        linear_combination = torch.bmm(energy, values).squeeze(1) #[Bx1xT]x[BxTxV] -> [BxV]
        return energy, linear_combination

class AttentionRNN(nn.Module):
    def __init__(self, cfg):
        super(AttentionRNN, self).__init__()
        self.config = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.encoder = Encoder(cfg.embed_dim, cfg.hidden_dim, cfg.nlayers, 
                               cfg.dropout, cfg.bidirectional, cfg.rnn_type)
        attention_dim = cfg.hidden_dim if not cfg.bidirectional else 2 * cfg.hidden_dim
        self.attention = Attention(attention_dim, attention_dim, attention_dim)
        self.decoder = nn.Linear(attention_dim, cfg.num_classes)

        size = 0
        for p in self.parameters():
            size += p.nelement()
        print('Total param size: {}'.format(size))


    def forward(self, input):
        outputs, hidden = self.encoder(self.embedding(input))
        if isinstance(hidden, tuple): # LSTM
            hidden = hidden[1] # take the cell state

        if self.encoder.bidirectional: # need to concat the last 2 hidden layers
            hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden = hidden[-1]

        energy, linear_combination = self.attention(hidden, outputs, outputs) 
        logits = self.decoder(linear_combination)
        return_dict = {
            'pred': logits,
            'attention_weights':energy
        }

        return return_dict


class MyTorchModel(AbstractSupervisedModel):
    def __init__(self, model_class, config, criterion, optimizer, device='cpu'):
        self.model_class = model_class
        self.config = config
        self.device = device
        self._model = model_class(config).to(device)
        self.optimizer_class = optimizer
        self.optimizer = optimizer(self.model.parameters(), config.lr)
        self.criterion = criterion

    @property
    def model(self):
        return self._model
        
    def __call__(self, x):
        return self._model(x)

    def fit(self, X, y, **kwargs):
        # This is a _step_ in the iteration process.
        # Should assume model is in training mode
        
        # Train-specific code
        self.model.train()
        self.model.zero_grad()

        return_dict = self(X)
        logits = return_dict['pred']
        #print(logits.view(-1, self.config.num_classes), y.squeeze())
        loss = self.criterion(logits.view(-1, self.config.num_classes), y.squeeze())
        return_dict['loss'] = loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
        self.optimizer.step()
        return return_dict

    def predict(self, X, **kwargs):
        # Assumes that the model is in _eval_ mode
        self.model.eval()
        with torch.no_grad():
            return_dict = self(X)
            return return_dict
    
    def evaluate(self, X, y, **kwargs):
        self.model.eval()
        with torch.no_grad():
            return_dict = self(X)
            logits = return_dict['pred']
            loss = self.criterion(logits.view(-1, self.config.num_classes), y.squeeze())
            return_dict['loss'] = loss
            return return_dict

    def reset(self, **kwargs):
        # Restart model
        self._model = self.model_class(self.config).to(self.config.device)

    def __setstate__(self, state):
        print("Restoring model from state")
        self.model_class = state['model_class']
        self.config = Config(state['config'])
        self.device = state['device']
        # Deserialize model
        model = self.model_class(self.config)
        model.load_state_dict(state['model_state'])
        self._model = model

        # Deserialize optimizer
        self.optimizer_class = state['optimizer_class']
        self.optimizer = self.optimizer_class(self.model.parameters(), self.config.lr)
        self.optimizer.load_state_dict(state['optimizer_state'])

        # Deserialize loss
        loss_class = state['loss_class']
        self.criterion = loss_class()
        self.criterion.load_state_dict(state['loss_state'])

    def __getstate__(self):
        state = {
            'model_class': self.model_class,
            'config': dict(self.config),
            'model_state': self.model.state_dict(),
            'optimizer_class': self.optimizer_class,
            'optimizer_state': self.optimizer.state_dict(),
            'loss_class': self.criterion.__class__,
            'loss_state': self.criterion.state_dict(),
            'device': self.device
        }
        return state

class TorchTrainer(AbstractTrainer):
    def __init__(self, num_epochs, device, valid_iterator=None):
        self.epochs = num_epochs
        self.valid_iterator = valid_iterator
        self.device = device

    def train(self,
              model: AbstractSupervisedModel,
              iterator: Iterator,
              feature_transformer,
              label_transform_fun,
              **kwargs):
        # Actual training loop
        # Single training epoch

        for _ in range(self.epochs):
            total_time = time.time()
            for batch_num, (batch_x, batch_y) in enumerate(iterator):
                t = time.time()
                X = torch.from_numpy(
                    feature_transformer.transform(batch_x).swapaxes(0,1) # swap batch_size and T
                    ).to(self.device)
                y = torch.from_numpy(
                    label_transform_fun(batch_y)
                    ).to(self.device)

                return_dict = model.fit(X, y)

                print("[Batch]: {}/{} in {:.5f} seconds, loss={:.5f}".format(
                       batch_num, len(iterator), time.time() - t, return_dict['loss']), 
                       end='\r', flush=True)

            print(f"\nTotal time for train epoch: {time.time() - total_time}")

            total_time = time.time()
            for batch_num, (batch_x, batch_y) in enumerate(self.valid_iterator):
                t = time.time()
                X = torch.from_numpy(
                    feature_transformer.transform(batch_x).swapaxes(0,1) # swap batch_size and T
                    ).to(self.device)
                y = torch.from_numpy(
                    label_transform_fun(batch_y)
                    ).to(self.device)


                return_dict = model.evaluate(X, y)
                loss = return_dict['loss']
                print("[Valid]: {}/{} in {:.5f} seconds, loss={:.5f}".format(
                       batch_num, len(self.valid_iterator), time.time() - t, loss), 
                       end='\r', flush=True)

            print(f"\nTotal time for valid epoch: {time.time() - total_time}")


class Config(dict):
    def __init__(self, *args, **kwargs): 
        dict.__init__(self, *args, **kwargs)
            
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

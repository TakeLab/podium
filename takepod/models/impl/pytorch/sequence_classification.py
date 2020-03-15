import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

RNNS = ['LSTM', 'GRU']

class RNNEncoder(nn.Module):
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
    def __init__(self, pretrained_embedding=None, **config):
        super(AttentionRNN, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config['vocab_size'], config['embed_dim'])

        # Copy the pretrained embeddings if they exist
        if pretrained_embedding is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))

        self.encoder = RNNEncoder(config['embed_dim'], config['hidden_dim'], config['nlayers'], 
                               config['dropout'], config['bidirectional'], config['rnn_type'])
        attention_dim = config['hidden_dim'] if not config['bidirectional'] else 2 * config['hidden_dim']
        self.attention = Attention(attention_dim, attention_dim, attention_dim)
        self.decoder = nn.Linear(attention_dim, config['num_classes'])

        size = 0
        for p in self.parameters():
            size += p.nelement()
        print('Total parameter size: {}'.format(size))


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

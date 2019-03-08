from __future__ import print_function

# noinspection PyUnresolvedReferences
import keras
from keras.optimizers import *
from keras.models import Model
from keras.layers import *
import numpy as np

from .ChainCRF import ChainCRF


class BiLSTM:

    def __init__(self, **kwargs):
        default_hyperparameters = {
            'dropout': (0.5, 0.5),
            'classifier': 'CRF',
            'LSTM-Size': (100,),
            'optimizer': 'adam',
            'clipvalue': 0,
            'clipnorm': 1,
            'learning_rate': 0.01
        }

        if kwargs:
            default_hyperparameters.update(kwargs)
        self.params = default_hyperparameters

        self.model = None

    def build_model(self):
        embedding_size = self.params.get('embedding_size')
        output_dim = self.params.get('output_dim')

        tokens_input = Input(shape=(None, embedding_size),
                             dtype='float32',
                             name='embeddings_input')
        input_nodes = [tokens_input]

        # todo add character embeddings

        shared_layer = input_nodes
        cnt = 0

        for size in self.params['LSTM-Size']:
            cnt += 1
            if isinstance(self.params['dropout'], (list, tuple)):
                shared_layer = Bidirectional(
                    LSTM(
                        size,
                        return_sequences=True,
                        dropout=self.params['dropout'][0],
                        recurrent_dropout=self.params['dropout'][1]
                    ),
                    name='shared_varLSTM_' + str(cnt))(shared_layer)
            else:
                # Naive dropout
                shared_layer = Bidirectional(
                    LSTM(size, return_sequences=True),
                    name='shared_LSTM_' + str(cnt))(shared_layer)

                if self.params['dropout'] > 0.0:
                    shared_layer = TimeDistributed(
                        Dropout(self.params['dropout']),
                        name=f'shared_dropout_{self.params["dropout"]}'
                    )(shared_layer)

        n_class_labels = output_dim
        classifier = self.params['classifier']
        output = shared_layer

        if classifier == 'Softmax':
            output = TimeDistributed(
                Dense(n_class_labels, activation='softmax'),
                name='Softmax')(output)
            loss_fct = 'sparse_categorical_crossentropy'
        elif classifier == 'CRF':
            output = TimeDistributed(
                Dense(n_class_labels, activation=None),
                name='hidden_lin_layer')(output)
            crf = ChainCRF(name='CRF')
            output = crf(output)
            loss_fct = crf.sparse_loss
        else:
            raise ValueError('Unsupported classifier')

        optimizerParams = {}
        if self.params.get('clipnorm', 0) > 0:
            optimizerParams['clipnorm'] = self.params['clipnorm']
        if self.params.get('clipvalue', 0) > 0:
            optimizerParams['clipvalue'] = self.params['clipvalue']

        if self.params['optimizer'].lower() == 'adam':
            opt = Adam(**optimizerParams)
        elif self.params['optimizer'].lower() == 'nadam':
            opt = Nadam(**optimizerParams)
        elif self.params['optimizer'].lower() == 'rmsprop':
            opt = RMSprop(**optimizerParams)
        elif self.params['optimizer'].lower() == 'adadelta':
            opt = Adadelta(**optimizerParams)
        elif self.params['optimizer'].lower() == 'adagrad':
            opt = Adagrad(**optimizerParams)
        elif self.params['optimizer'].lower() == 'sgd':
            opt = SGD(lr=0.1, **optimizerParams)
        else:
            raise ValueError('Unsupported optimizer')

        model = Model(inputs=input_nodes, outputs=[output])
        model.compile(loss=loss_fct, optimizer=opt)
        model.summary(line_length=125)

        self.model = model

        learning_rate = self.params['learning_rate']
        K.set_value(self.model.optimizer.SHlr, learning_rate)

    def train_model(self, X, y):
        self.model.train_on_batch(X, y)

    def fit(self, X, y):
        if self.model is None:
            self.build_model()

        self.train_model(X, np.expand_dims(y, -1))

    def predict(self, X):
        predictions = self.model.predict(X, verbose=False)
        return predictions.argmax(axis=-1)  # can be solved by the Reshape layer

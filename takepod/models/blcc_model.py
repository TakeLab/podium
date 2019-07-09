"""Module contains deep learning based sequence labelling model.."""
import numpy as np
from keras import backend as K
from keras.layers import Bidirectional, concatenate, Conv1D, Dense, Dropout
from keras.layers import Embedding, GlobalMaxPool1D, Input, LSTM
from keras.layers import Reshape, TimeDistributed
from keras.models import Model
from keras.optimizers import Adadelta, Adagrad, Adam, Nadam, RMSprop, SGD

from takepod.models import AbstractSupervisedModel
from takepod.models.blcc.chain_crf import ChainCRF


class BLCCModel(AbstractSupervisedModel):
    """
    Deep learning model for sequence labelling tasks.

    Originally proposed in the following paper:
    https://arxiv.org/pdf/1603.01354.pdf

    Attributes
    ----------
    EMBEDDING_SIZE : int
        Size of the word embeddings
    OUTPUT_SIZE : int
        Number of output classes
    FEATURE_NAMES : iterable(str)
        Names of the custom features
    FEATURE_INPUT_SIZES : iterable(str)
        Input sizes of the custom features
    FEATURE_OUTPUT_SIZES : iterable(str)
        Output sizes of the custom features
    LEARNING_RATE : float
        Learning rate
    CLIPNORM : float
        Optimizer clip norm
    CLIPVALUE : float
        Optimizer clip value
    OPTIMIZER : str
        Optimizer name
        Supported optimizers: ['adam', 'nadam', 'rmsprop', 'adadelta',
        'adagrad', 'sgd']
    CLASSIFIER : str
        Classifier name
        Supported classifiers: ['Softmax', 'CRF']
    LSTM_SIZE = iterable(int)
        Sizes of the bidirectional LSTM layers.
    DROPOUT : float | iterable(float)
        If the single float value is given, a naive dropout with the given value
        is applied after each LSTM layer.
        If the iterable value is given, the first element is applied as a
        standard dropout and a second is applied as a recurrent dropout within
        a LSTM layer.
    """

    # TODO : model expects keras=2.2.4 to be installed

    EMBEDDING_SIZE = 'embedding_size'
    OUTPUT_SIZE = 'output_size'

    CHAR_EMBEDDINGS_ENABLED = 'char_embeddings_enabled'
    CHAR_EMBEDDINGS_INPUT_SIZE = 'char_embeddings_input_size'
    CHAR_EMBEDDINGS_OUTPUT_SIZE = 'char_embeddings_output_size'
    CHAR_EMBEDDINGS_NUM_OF_CHARS = 'char_embeddings_num_of_chars'
    CHAR_EMBEDDINGS_NUM_OF_FILTERS = 'char_embeddings_num_of_filters'
    CHAR_EMBEDDINGS_KERNEL_SIZE = 'char_embeddings_kernel_size'

    FEATURE_NAMES = 'feature_names'
    FEATURE_INPUT_SIZES = 'feature_input_sizes'
    FEATURE_OUTPUT_SIZES = 'feature_output_sizes'

    LEARNING_RATE = 'learning_rate'
    CLIPNORM = 'clipnorm'
    CLIPVALUE = 'clipvalue'
    OPTIMIZER = 'optimizer'
    CLASSIFIER = 'classifier'
    LSTM_SIZE = 'LSTM-Size'
    DROPOUT = 'dropout'

    def __init__(self, **kwargs):
        default_hyperparameters = {
            self.EMBEDDING_SIZE: None,
            self.OUTPUT_SIZE: None,

            self.CHAR_EMBEDDINGS_ENABLED: False,
            self.CHAR_EMBEDDINGS_INPUT_SIZE: 25,
            self.CHAR_EMBEDDINGS_OUTPUT_SIZE: 30,
            self.CHAR_EMBEDDINGS_NUM_OF_CHARS: None,  # TODO make mandatory parameter
            self.CHAR_EMBEDDINGS_NUM_OF_FILTERS: 30,
            self.CHAR_EMBEDDINGS_KERNEL_SIZE: 3,

            self.FEATURE_NAMES: (),
            self.FEATURE_INPUT_SIZES: (),
            self.FEATURE_OUTPUT_SIZES: (),

            self.DROPOUT: (0.5, 0.5),
            self.CLASSIFIER: 'CRF',
            self.LSTM_SIZE: (100,),
            self.OPTIMIZER: 'adam',
            self.CLIPVALUE: 0.0,
            self.CLIPNORM: 1.0,
            self.LEARNING_RATE: 0.01
        }

        if kwargs:
            default_hyperparameters.update(kwargs)

        self.params = default_hyperparameters
        self.model = self._build_model()

    def _build_model(self):
        """Method initializes and compiles the model."""
        embedding_size = self.params.get(self.EMBEDDING_SIZE)

        tokens_input = Input(shape=(None, embedding_size),
                             dtype='float32',
                             name='embeddings_input')

        input_nodes = [tokens_input]
        input_layers_to_concatenate = [tokens_input]

        # Character embeddings
        char_embeddings_enabled = self.params.get(self.CHAR_EMBEDDINGS_ENABLED)
        char_embeddings_input_size = \
            self.params.get(self.CHAR_EMBEDDINGS_INPUT_SIZE)
        num_of_chars = self.params.get(self.CHAR_EMBEDDINGS_NUM_OF_CHARS)
        char_embedding_output_size = \
            self.params.get(self.CHAR_EMBEDDINGS_OUTPUT_SIZE)
        char_num_of_filters = \
            self.params.get(self.CHAR_EMBEDDINGS_NUM_OF_FILTERS)
        char_kernel_size = self.params.get(self.CHAR_EMBEDDINGS_KERNEL_SIZE)

        if char_embeddings_enabled:
            char_input = Input(
                shape=(None,),
                dtype='int32',
                name='char_flat_input'
            )

            char_input_reshape = Reshape(
                target_shape=(-1, char_embeddings_input_size),
                name='char_input_reshape'
            )(char_input)

            char_embedding = TimeDistributed(
                Embedding(
                    input_dim=num_of_chars,
                    output_dim=char_embedding_output_size,
                    trainable=True
                ),
                name='char_embeddings')(char_input_reshape)

            char_embedding = TimeDistributed(
                Conv1D(
                    char_num_of_filters,
                    char_kernel_size,
                    padding='same'
                ),
                name="char_cnn"
            )(char_embedding)

            char_embedding = TimeDistributed(
                GlobalMaxPool1D(), name="char_pooling"
            )(char_embedding)

            input_nodes.append(char_input)
            input_layers_to_concatenate.append(char_embedding)

        # Custom features
        custom_feature_properties = zip(
            self.params.get(self.FEATURE_NAMES),
            self.params.get(self.FEATURE_INPUT_SIZES),
            self.params.get(self.FEATURE_OUTPUT_SIZES)
        )
        for name, input_size, output_size in custom_feature_properties:
            feature_input = Input(shape=(None,), dtype='int32',
                                  name=f'{name}_input')

            feature_embedding = Embedding(
                input_dim=input_size,
                output_dim=output_size,
                name=f'{name}_embeddings')(feature_input)

            input_nodes.append(feature_input)
            input_layers_to_concatenate.append(feature_embedding)

        if len(input_layers_to_concatenate) > 1:
            shared_layer = concatenate(input_layers_to_concatenate)
        else:
            shared_layer = input_layers_to_concatenate[0]

        # Core LSTM layer(s)
        cnt = 1
        for size in self.params[self.LSTM_SIZE]:
            if isinstance(self.params[self.DROPOUT], (list, tuple)):
                shared_layer = Bidirectional(
                    LSTM(
                        size,
                        return_sequences=True,
                        dropout=self.params[self.DROPOUT][0],
                        recurrent_dropout=self.params[self.DROPOUT][1]
                    ),
                    name=f'shared_varLSTM_{cnt}')(shared_layer)
            else:
                # Naive dropout
                shared_layer = Bidirectional(
                    LSTM(size, return_sequences=True),
                    name=f'shared_LSTM_{cnt}')(shared_layer)

                if self.params[self.DROPOUT] > 0.0:
                    shared_layer = TimeDistributed(
                        Dropout(self.params[self.DROPOUT]),
                        name=f'shared_dropout_{self.params[self.DROPOUT]}'
                    )(shared_layer)
            cnt += 1

        # Classifier layer
        n_class_labels = self.params.get(self.OUTPUT_SIZE)
        output = shared_layer
        classifier = self.params[self.CLASSIFIER]
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
            raise ValueError(f'Unsupported classifier: {classifier}')

        optimizer_params = {}
        if self.params.get(self.CLIPNORM, 0) > 0:
            optimizer_params[self.CLIPNORM] = self.params[self.CLIPNORM]
        if self.params.get(self.CLIPVALUE, 0) > 0:
            optimizer_params[self.CLIPVALUE] = self.params[self.CLIPVALUE]

        optimizer = self.params[self.OPTIMIZER].lower()
        if optimizer == 'adam':
            opt = Adam(**optimizer_params)
        elif optimizer == 'nadam':
            opt = Nadam(**optimizer_params)
        elif optimizer == 'rmsprop':
            opt = RMSprop(**optimizer_params)
        elif optimizer == 'adadelta':
            opt = Adadelta(**optimizer_params)
        elif optimizer == 'adagrad':
            opt = Adagrad(**optimizer_params)
        elif optimizer == 'sgd':
            opt = SGD(lr=0.1, **optimizer_params)
        else:
            raise ValueError(f'Unsupported optimizer: {optimizer}')

        model = Model(inputs=input_nodes, outputs=[output])
        model.compile(loss=loss_fct, optimizer=opt)
        model.summary(line_length=125)

        learning_rate = self.params[self.LEARNING_RATE]
        K.set_value(model.optimizer.lr, learning_rate)

        return model

    def fit(self, X, y, **kwargs):
        """
        Method calls fit on BLCC model with the given batch.
        It is supposed to be used as online learning.
        """
        self.model.train_on_batch(X, np.expand_dims(y, -1))

    def predict(self, X, **kwargs):
        predictions = self.model.predict(X, verbose=False)
        y_pred = predictions.argmax(axis=-1)
        return {AbstractSupervisedModel.PREDICTION_KEY: y_pred}

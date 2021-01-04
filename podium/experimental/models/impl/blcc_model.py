"""
Module contains deep learning based sequence labelling model.
"""
import tempfile

import numpy as np

from podium.experimental.models import AbstractSupervisedModel
from podium.experimental.models.impl.blcc.chain_crf import ChainCRF, create_custom_objects


try:
    from keras import backend as K
    from keras.layers import (
        LSTM,
        Bidirectional,
        Dense,
        Dropout,
        Embedding,
        Input,
        TimeDistributed,
        concatenate,
    )
    from keras.models import Model, load_model
    from keras.optimizers import SGD, Adadelta, Adagrad, Adam, Nadam, RMSprop
except ImportError:
    print(
        "Problem occured while trying to import keras. If the "
        "library is not installed visit https://keras.io/ "
        "for more details."
    )
    raise


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

    EMBEDDING_SIZE = "embedding_size"
    OUTPUT_SIZE = "output_size"

    FEATURE_NAMES = "feature_names"
    FEATURE_INPUT_SIZES = "feature_input_sizes"
    FEATURE_OUTPUT_SIZES = "feature_output_sizes"

    LEARNING_RATE = "learning_rate"
    CLIPNORM = "clipnorm"
    CLIPVALUE = "clipvalue"
    OPTIMIZER = "optimizer"
    CLASSIFIER = "classifier"
    LSTM_SIZE = "LSTM-Size"
    DROPOUT = "dropout"

    def __init__(self, **kwargs):
        self.reset(**kwargs)

    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as fd:
            self.model.save(fd.name, overwrite=True)
            model_str = fd.read()
            odict = self.__dict__.copy()
            del odict["model"]
        return {"model_str": model_str, "rest": odict}

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as fd:
            fd.write(state["model_str"])
            fd.flush()
            model = load_model(fd.name, custom_objects=create_custom_objects())
        self.__dict__ = state["rest"]
        self.model = model

    def reset(self, **kwargs):
        default_hyperparameters = {
            self.EMBEDDING_SIZE: None,
            self.OUTPUT_SIZE: None,
            self.FEATURE_NAMES: (),
            self.FEATURE_INPUT_SIZES: (),
            self.FEATURE_OUTPUT_SIZES: (),
            self.DROPOUT: (0.5, 0.5),
            self.CLASSIFIER: "CRF",
            self.LSTM_SIZE: (100,),
            self.OPTIMIZER: "adam",
            self.CLIPVALUE: 0.0,
            self.CLIPNORM: 1.0,
            self.LEARNING_RATE: 0.01,
        }

        if kwargs:
            default_hyperparameters.update(kwargs)

        self.params = default_hyperparameters
        self.model = self._build_model()

    def _build_model(self):
        """
        Method initializes and compiles the model.

        Raises
        ------
        ValueError
            If the given classifier is not supported.
            If the given optimizer is not supported.
        """
        embedding_size = self.params.get(self.EMBEDDING_SIZE)

        output_size = self.params.get(self.OUTPUT_SIZE)

        tokens_input = Input(
            shape=(None, embedding_size), dtype="float32", name="embeddings_input"
        )

        input_nodes = [tokens_input]
        input_layers_to_concatenate = [tokens_input]

        # TODO add character embeddings

        custom_feature_properties = zip(
            self.params.get(self.FEATURE_NAMES),
            self.params.get(self.FEATURE_INPUT_SIZES),
            self.params.get(self.FEATURE_OUTPUT_SIZES),
        )
        for name, input_size, output_size in custom_feature_properties:
            feature_input = Input(shape=(None,), dtype="int32", name=f"{name}_input")

            feature_embedding = Embedding(
                input_dim=input_size,
                output_dim=output_size,
                name=f"{name}_embeddings",
            )(feature_input)

            input_nodes.append(feature_input)
            input_layers_to_concatenate.append(feature_embedding)

        if len(input_layers_to_concatenate) > 1:
            shared_layer = concatenate(input_layers_to_concatenate)
        else:
            shared_layer = input_layers_to_concatenate[0]

        cnt = 1

        for size in self.params[self.LSTM_SIZE]:
            if isinstance(self.params[self.DROPOUT], (list, tuple)):
                shared_layer = Bidirectional(
                    LSTM(
                        size,
                        return_sequences=True,
                        dropout=self.params[self.DROPOUT][0],
                        recurrent_dropout=self.params[self.DROPOUT][1],
                    ),
                    name=f"shared_varLSTM_{cnt}",
                )(shared_layer)
            else:
                # Naive dropout
                shared_layer = Bidirectional(
                    LSTM(size, return_sequences=True), name=f"shared_LSTM_{cnt}"
                )(shared_layer)

                if self.params[self.DROPOUT] > 0.0:
                    shared_layer = TimeDistributed(
                        Dropout(self.params[self.DROPOUT]),
                        name=f"shared_dropout_{self.params[self.DROPOUT]}",
                    )(shared_layer)
            cnt += 1

        n_class_labels = output_size
        output = shared_layer
        classifier = self.params[self.CLASSIFIER]

        if classifier == "Softmax":
            output = TimeDistributed(
                Dense(n_class_labels, activation="softmax"), name="Softmax"
            )(output)
            loss_fct = "sparse_categorical_crossentropy"
        elif classifier == "CRF":
            output = TimeDistributed(
                Dense(n_class_labels, activation=None), name="hidden_lin_layer"
            )(output)
            crf = ChainCRF(name="CRF")
            output = crf(output)
            loss_fct = crf.sparse_loss
        else:
            raise ValueError(f"Unsupported classifier: {classifier}")

        optimizerParams = {}
        if self.params.get(self.CLIPNORM, 0) > 0:
            optimizerParams[self.CLIPNORM] = self.params[self.CLIPNORM]
        if self.params.get(self.CLIPVALUE, 0) > 0:
            optimizerParams[self.CLIPVALUE] = self.params[self.CLIPVALUE]

        optimizer = self.params[self.OPTIMIZER].lower()
        if optimizer == "adam":
            opt = Adam(**optimizerParams)
        elif optimizer == "nadam":
            opt = Nadam(**optimizerParams)
        elif optimizer == "rmsprop":
            opt = RMSprop(**optimizerParams)
        elif optimizer == "adadelta":
            opt = Adadelta(**optimizerParams)
        elif optimizer == "adagrad":
            opt = Adagrad(**optimizerParams)
        elif optimizer == "sgd":
            opt = SGD(lr=0.1, **optimizerParams)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

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

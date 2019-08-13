"""Module contains ULMFiT implementations."""
import logging
import numpy as np
import pandas as pd
from takepod.models.model import AbstractFrameworkModel, AbstractSupervisedModel
from fastai.core import partial
from fastai.text import *
_LOGGER = logging.getLogger(__name__)
ENCODER_NAME = 'ft_enc'


def train_valid_split(df, valid_pct):
    """Method that imitates train-validation split
    inside fastai library
    """
    df = df.iloc[np.random.permutation(len(df))]
    cut = int(valid_pct * len(df)) + 1
    train_df, valid_df = df[cut:], df[:cut]
    return train_df, valid_df


class UlmfitCroatian(AbstractFrameworkModel, AbstractSupervisedModel):
    """ULMFiT implementation for Croatian language"""
    def __init__(self, model_file, itos_file):
        """Constructor that initialized ULMFiT with given paths to
               responding files for pretrained ULMFiT language model.
               """
        self.reset(model_file=model_file, itos_file=itos_file)

    def train_lm(self, text_feature):
        """Method that performs fine-tuning of pretrained language model
        on target dataset.
        """
        df = pd.DataFrame({'text': text_feature})
        train_df, valid_df = train_valid_split(df, valid_pct=0.2)
        self.data_lm = TextLMDataBunch.from_df(train_df=train_df, valid_df=valid_df, path='.',
                                          text_cols=0, bs=30)
        pretrained_fnames = ['hr-100-best', 'itos']
        learner = language_model_learner(self.data_lm, AWD_LSTM, pretrained_fnames=pretrained_fnames, drop_mult=0.9,
                                         model_dir='./')
        learner.freeze()
        learner.opt_func = partial(optim.Adam, betas=(0.8, 0.99))
        learner.fit_one_cycle(1, 1e-2)
        learner.unfreeze()
        learner.fit_one_cycle(1, 1e-3, moms=(0.8, 0.7))
        learner.save_encoder(ENCODER_NAME)

    def train_classifier(self, X, y):
        """Method that trains classifier for
        target task.
        """
        df = pd.DataFrame({'text': X, 'target': y})
        train_df, valid_df = train_valid_split(df, valid_pct=0.2)
        data_class = TextClasDataBunch.from_df(path='.',
                                               train_df=train_df,
                                               valid_df=valid_df,
                                               vocab=self.data_lm.vocab,
                                               text_cols='text', label_cols='target', bs=30)
        learner = text_classifier_learner(data_class, AWD_LSTM, drop_mult=0.5, model_dir='./')
        learner.load_encoder(ENCODER_NAME)
        learner.fit_one_cycle(1, 1e-2, moms=(0.8, 0.7))
        learner.freeze_to(-2)
        learner.fit_one_cycle(1, slice(1e-2 / (2.6 ** 4), 1e-2), moms=(0.8, 0.7))
        learner.freeze_to(-3)
        learner.fit_one_cycle(1, slice(5e-3 / (2.6 ** 4), 5e-3), moms=(0.8, 0.7))
        learner.unfreeze()
        learner.fit_one_cycle(2, slice(1e-3 / (2.6 ** 4), 1e-3), moms=(0.8, 0.7))
        return learner

    def load(self, **kwargs):
        """TODO: add load function which will load files from somewhere - server/dropbox/google-drive, for now add
        files manually
        """
        self.model_file = kwargs.get('model_file', None)
        self.itos_file = kwargs.get('itos_file', None)

    def save(self, file_path, **kwargs):
        self._model.export(file_path)

    def fit(self, X, y, **kwargs):
        self.train_lm(X)
        self._model = self.train_classifier(X, y)

    def predict(self, X, **kwargs):
        return self._model.predict(X)

    def reset(self, **kwargs):
        self.load(**kwargs)



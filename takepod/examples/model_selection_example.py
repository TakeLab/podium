from takepod.storage import Field, Vocab
from takepod.datasets.impl import BasicSupervisedImdbDataset
from takepod.storage.vectorizers.vectorizer import GloVe
from takepod.datasets import SingleBatchIterator
from takepod.models.impl.fc_model import ScikitMLPClassifier
from takepod.models.impl.sklearn_models import SklearnModels, ModelType
from takepod.models import Experiment, FeatureTransformer, SklearnTensorTransformerWrapper
from takepod.models.impl.simple_trainers import SimpleTrainer
from takepod.model_selection import grid_search
from takepod.preproc.stop_words import get_croatian_stop_words_removal_hook

from sklearn.metrics import accuracy_score
from nltk.tokenize import WordPunctTokenizer
from functools import partial
import numpy as np
import pickle
import os
from itertools import chain, combinations
from recordclass import recordclass


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def combine_all(*args, names=None):

    lists_to_combine = len(args)
    if lists_to_combine <= 1:
        return args
    
    combination_number = 1
    for i in range(lists_to_combine):
        combination_number *= len(args[i])

    if names:
        SingleCombination = recordclass('SingleCombination', ' '.join(names))
        combinations = combination_number * [SingleCombination(*([None] * len(names)))]
    else:
        combinations = combination_number * [len(args) * [None]]

    for i in range(lists_to_combine):
        repeat_times = combination_number // len(args[i])
        for j in range(repeat_times * len(args[i])):
            combinations[j][i] = args[i][j % len(args[i])]
    return combinations


def lowercase_hook(raw, data):
    return raw, [d.lower() for d in data]


def remove_br_html_tag_hook(raw, data):
    return data, [
        d.replace(d.replace('<br', ''), 'br>', '')
        for d in data 
        if '<br' not in d or 'br>' not in d
    ]


def remove_non_alphas(raw, data):
    return raw, [
        d for d in data
        if d.isalpha()
    ]


def feature_transform_fn(x_batch, embedding_matrix):
    """Batch transform function that returns a mean of embedding vectors for every
        token in an Example"""
    x_tensor = np.take(embedding, x_batch.text.astype(int), axis=0)
    x = np.mean(x_tensor, axis=1)
    return x


def label_transform_fun(y_batch):
    return y_batch.label.ravel()


def load_dataset(combination):
    fields = BasicSupervisedImdbDataset.get_default_fields()
    text = Field(
        name=BasicSupervisedImdbDataset.TEXT_FIELD_NAME, vocab=Vocab(),
        # get tokenizer from combinations
        tokenizer=combination.tokenizer, language="en", tokenize=True,
        store_as_raw=False
    )
    fields['text'] = text
    for hook in combination.posttokenize_hook:
        # get set of hooks to try out 
        fields['text'].add_posttokenize_hook(lowercase_hook)
 
    ds = BasicSupervisedImdbDataset.get_train_test_dataset(fields)
 
    # get vectorizer from combination
    vectorizer = combination.vectorizer

    voc = fields['text'].vocab
    vectorizer.load_vocab(vocab=fields['text'].vocab)
    embedding = vectorizer.get_embedding_matrix(voc)
    return ds, embedding


HOOKS_TO_TRY = [
    get_croatian_stop_words_removal_hook, 
    lowercase_hook, 
    remove_br_html_tag_hook, remove_non_alphas
]
# we wish to try hooks, they should be order invariant, therefore
# no permutation is needed
HOOK_COMBINATIONS = powerset(HOOKS_TO_TRY)
TOKENIZERS_TO_TRY = [
    'split', WordPunctTokenizer().tokenize, 'spacy'
]
VECTORIZER_TO_TRY = [
    GloVe(),
]

MODEL_PARAM_GRID = {
    'model': [
        {
            'model_type': ModelType.SUPPORT_VECTOR_MACHINE,
            'model_specific': {'C': 0.1, 'kernel': 'linear'},
        },
        {
            'model_type': ModelType.SUPPORT_VECTOR_MACHINE,
            'model_specific': {'C': 1, 'kernel': 'linear'},
        },
        {
            'model_type': ModelType.SUPPORT_VECTOR_MACHINE,
            'model_specific': {'C': 10, 'kernel': 'linear'},
        },
        {
            'model_type': ModelType.SUPPORT_VECTOR_MACHINE,
            'model_specific': {'C': 100, 'kernel': 'linear'},
        },
        {
            'model_type': ModelType.SUPPORT_VECTOR_MACHINE,
            'model_specific': {'C': 1, 'gamma': 0.001, 'kernel': 'rbf'},
        },
        {
            'model_type': ModelType.LOGISTIC_REGRESSION,
            'model_specific': {'penalty': 'none'},
        },
        {
            'model_type': ModelType.LOGISTIC_REGRESSION,
            'model_specific': {'C': 1},
        },
        {
            'model_type': ModelType.LOGISTIC_REGRESSION,
            'model_specific': {'C': 0.1},
        },
        {
            'model_type': ModelType.STOCHASTIC_GRADIENT_DESCENT,
            'model_specific': {},
        }
    ]
}


if __name__ == '__main__':
    combinations = combine_all(
        list(HOOK_COMBINATIONS), TOKENIZERS_TO_TRY, VECTORIZER_TO_TRY,
        names=['posttokenize_hook', 'tokenizer', 'vectorizer']
    )

    for combination in combinations:
        print('using preprocessing combination of {}'.format(combination))

        print('loading data set and embeddings')
        ds, embedding = load_dataset(combination)
        print('finished loading data set and embeddings')

        train_set, test_set = ds
     
        trainer = SimpleTrainer()
        iterator = SingleBatchIterator 

        feature_transformer = FeatureTransformer(
            partial(feature_transform_fn, embedding_matrix=embedding), None
        )
        
        experiment = Experiment(SklearnModels,
                                trainer=trainer,
                                training_iterator_callable=iterator,
                                prediction_iterator_callable=iterator,
                                feature_transformer=feature_transformer,
                                label_transform_fun=label_transform_fun)
        
        print('performing grid search')
        best_score, best_model_params, best_train_params = grid_search(experiment,
            train_set,
            accuracy_score,
            model_param_grid=MODEL_PARAM_GRID,
            trainer_param_grid={SimpleTrainer.MAX_EPOCH_KEY: [1]},
            n_splits=5
        )

        print(
            'combination {} has best score of {} with params {}'
            .format(combination, best_score, best_model_params)
        )

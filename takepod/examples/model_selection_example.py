from takepod.storage import Field, Vocab
from takepod.datasets.impl import BasicSupervisedImdbDataset
from takepod.storage.vectorizers.vectorizer import GloVe
from takepod.datasets import SingleBatchIterator
from takepod.models.impl.fc_model import ScikitMLPClassifier
from takepod.models.impl.sklearn_models import SklearnModels, ModelType
from takepod.models import Experiment, FeatureTransformer, SklearnTensorTransformerWrapper
from takepod.models.impl.simple_trainers import SimpleTrainer
from takepod.model_selection import grid_search

from sklearn.metrics import accuracy_score
from nltk.tokenize import WordPunctTokenizer
from functools import partial
import numpy as np
import pickle
import os


def lowercase_hook(raw, data):
    lowercase_data = [d.lower() for d in data]
    return raw, lowercase_data


if os.path.isfile('imdb.pkl') and os.path.isfile('imdb_embed.pkl'):
    print('loading pickled')
    train_set, test_set = pickle.load(open('imdb.pkl', 'rb'))
    embedding = pickle.load(open('imdb_embed.pkl', 'rb'))
else:
    tokenizers_to_try = ['split', WordPunctTokenizer().tokenize, 'spacy']

    fields = BasicSupervisedImdbDataset.get_default_fields()
    text = Field(name=BasicSupervisedImdbDataset.TEXT_FIELD_NAME, vocab=Vocab(),
                 tokenizer=tokenizers_to_try[0], language="en", tokenize=True,
                 store_as_raw=False)
    fields['text'] = text
    fields['text'].add_posttokenize_hook(lowercase_hook)

    ds = BasicSupervisedImdbDataset.get_train_test_dataset(fields)
    print('pickling to reuse later')
    pickle.dump(ds, open('imdb.pkl', 'wb'))

    train_set, test_set = ds
    vectorizer = GloVe()
    # train_vocab = train_set.field_dict['text'].vocab
    voc = fields['text'].vocab
    vectorizer.load_vocab(vocab=fields['text'].vocab)
    embedding = vectorizer.get_embedding_matrix(voc)

    print('pickling embeddings')
    pickle.dump(embedding, open('imdb_embed.pkl', 'wb'))

def feature_transform_fn(x_batch):
    """Batch transform function that returns a mean of embedding vectors for every
        token in an Example"""
    global embedding

    x_tensor = np.take(embedding, x_batch.text.astype(int), axis=0)
    x = np.mean(x_tensor, axis=1)
    return x

def label_transform_fun(y_batch):
    return y_batch.label.ravel()

trainer = SimpleTrainer()
iterator = SingleBatchIterator

# tensor_transformer = SklearnTensorTransformerWrapper(StandardScaler())
feature_transformer = FeatureTransformer(
    feature_transform_fn, None
)

num_of_classes = len(train_set.field_dict["label"].vocab.itos)

experiment = Experiment(SklearnModels,
                        trainer=trainer,
                        training_iterator_callable=iterator,
                        prediction_iterator_callable=iterator,
                        feature_transformer=feature_transformer,
                        label_transform_fun=label_transform_fun)

model_param_grid = {
    'classes': [np.arange(num_of_classes)], 
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
            'model_specific': {},
        },
        {
            'model_type': ModelType.STOCHASTIC_GRADIENT_DESCENT,
            'model_specific': {},
        }
    ]
}

best_score, best_model_params, best_train_params = grid_search(experiment,
    test_set,
    accuracy_score,
    model_param_grid=model_param_grid,
    trainer_param_grid={SimpleTrainer.MAX_EPOCH_KEY: [1]},
    n_splits=5
)

print(best_model_params)
predictions = experiment.predict(test_set)

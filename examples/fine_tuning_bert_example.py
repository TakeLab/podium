# flake8: noqa
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from podium.datasets import IMDB, Iterator
from podium.models import Experiment
from podium.models.impl.pytorch import TorchTrainer, TorchModel
from podium.pipeline import Pipeline
from podium.storage import Field, LabelField, Vocab


def create_fields():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def text_to_tokens(string):
        input_ids = tokenizer(string,
                              max_length=128,
                              padding=False,
                              truncation=True,
                              return_attention_mask=False
                              )['input_ids']

        return tokenizer.convert_ids_to_tokens(input_ids)

    text = Field(name='text',
                 tokenizer=text_to_tokens,
                 custom_numericalize=tokenizer.convert_tokens_to_ids,
                 padding_token=0)

    label = LabelField(name='label', vocab=Vocab(specials=()))

    return {
        'text': text,
        'label': label
    }

def bert_initializer():
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', 
                                                                return_dict=True)
    def get_bert_model():
        return model
    
    return get_bert_model

get_bert_model = bert_initializer()


class BertModelWrapper(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.model = get_bert_model()
    
    def forward(self, x):
        attention_mask = (x != 0).long()
        return_dict = self.model(x, attention_mask)
        return_dict['pred'] = return_dict['logits']
        return return_dict


def main():
    # loading the IMDB dataset
    fields = create_fields()
    imdb_train, imdb_test = IMDB.get_dataset_splits(fields)

    # setting up the experiment for fine-tuning the model
    model_config = {
        'lr': 1e-5,
        'clip': float('inf'),  # disable gradient clipping
        'num_epochs': 3,
    }
    model_config['num_classes'] = len(fields['label'].vocab)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    iterator = Iterator(batch_size=32)
    trainer = TorchTrainer(model_config['num_epochs'], device, iterator, imdb_test)

    # we have to swap axes to nullify the effect of swapping axes afterwards
    # because we work with the batch-first model (we should add this option to Podium!!!)
    feature_transformer = lambda feature_batch: feature_batch[0].astype(np.int64).swapaxes(0, 1)
    label_transformer = lambda label_batch: label_batch[0].astype(np.int64)

    experiment = Experiment(TorchModel,
                            trainer=trainer,
                            feature_transformer=feature_transformer,
                            label_transform_fn=label_transformer)

    experiment.fit(imdb_train,
                   model_kwargs={
                       'model_class': BertModelWrapper,
                       'criterion': nn.CrossEntropyLoss(),
                       'optimizer': optim.AdamW,
                       'device': device,
                       **model_config
                   })

    # utilities for saving/loading the model
    def save_model(model, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(fitted_model, f)

    def load_model(file_path):
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model

    fitted_model = experiment.model

    model_file = 'bert_model.pt'
    save_model(fitted_model, model_file)
    loaded_model = load_model(model_file)

    # here we show how to use the raw model to make predictions,
    # this is how you can use the model that is already fine-tuned
    cast_to_torch_transformer = lambda t: torch.from_numpy(t[0].astype(np.int64)).to(device)

    @torch.no_grad()
    def make_predictions(raw_model, dataset, batch_size=64):
        raw_model.eval()
        
        def predict(batch):
            predictions = raw_model(cast_to_torch_transformer(batch))['pred']
            return predictions.cpu().numpy()

        iterator = Iterator(batch_size=batch_size, 
                            shuffle=False)
        
        predictions = []
        for x_batch, _ in iterator(dataset):
            batch_prediction = predict(x_batch)
            predictions.append(batch_prediction)
        
        return np.concatenate(predictions)
    
    # model comparison: pretrained BERT vs pretrained + fine-tuned BERT
    _, y_true = imdb_test.batch()
    y_true = y_true[0].ravel()

    predictions = make_predictions(BertModelWrapper(), imdb_test)
    y_pred = predictions.argmax(axis=1)

    print('pretrained model')
    print('accuracy score:', accuracy_score(y_true, y_pred))
    print('precision score:', precision_score(y_true, y_pred))
    print('recall score:', recall_score(y_true, y_pred))
    print('f1 score:', f1_score(y_true, y_pred))
    
    loaded_model_raw = loaded_model.model
    predictions = make_predictions(loaded_model_raw, imdb_test)
    y_pred = predictions.argmax(axis=1)

    print('pretrained + fine-tuned model')
    print('accuracy score:', accuracy_score(y_true, y_pred))
    print('precision score:', precision_score(y_true, y_pred))
    print('recall score:', recall_score(y_true, y_pred))
    print('f1 score:', f1_score(y_true, y_pred))
    
    # we use `Pipeline` to make predictions on raw data
    pipe = Pipeline(fields=list(fields.values()),
                    example_format='list',
                    feature_transformer=cast_to_torch_transformer,
                    model=loaded_model)

    instances = [
        ['This movie is horrible'],
        ['This movie is great!']
    ]

    for instance in instances:
        predictions = pipe.predict_raw(instance)
        print(f'instance: {instance}, predicted label: '
              f'{fields["label"].vocab.itos[predictions.argmax()]}, '
              f'predictions: {predictions}')


if __name__ == '__main__':
    main()

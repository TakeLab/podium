from takepod.models.base_model import SupervisedModel
import takepod.preproc.transform as trans
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import sklearn.metrics
from takepod.metrics import standard_metrics


import sklearn.svm
import numpy as np

class PauzaModelSVMBow (SupervisedModel):
    def __init__(self, word_to_ix=None):
        self.model = sklearn.svm.SVC()
        self.word_to_ix = word_to_ix

    def train(self, X, y):
        if self.word_to_ix == None:
            self.word_to_ix = trans.create_word_to_index(X)
        X_bow = [trans.make_bow_vector(i, self.word_to_ix) for i in X]
        kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        np.random.seed(1)
        grid_svm = GridSearchCV(self.model, param_grid={'C':[0.01, 0.1,1]}, cv=kfolds, scoring="f1", verbose=1, n_jobs=-1)
        grid_svm.fit(X_bow, y)
        print("Best params ",grid_svm.best_params_, grid_svm.best_score_)
        self.model = sklearn.svm.SVC(C=grid_svm.best_params_['C'])
        self.model.fit(X_bow, y)

    def test(self, X):
        X_bow = [trans.make_bow_vector(i, self.word_to_ix) for i in X]
        return self.model.predict(X_bow)
   

class PauzaModelSVMW2V(SupervisedModel):
    def __init__(self, w2v_path="tweeterVectors.bin"):
        self.model = sklearn.svm.SVR()
        self.w2v=KeyedVectors.load_word2vec_format(fname=w2v_path, binary=True, unicode_errors='ignore')

    def train(self, X, y):
        X_train = [self._line_to_vector(i) for i in X]
        self.model.fit(X=X_train, y=y)

    def test(self, X):
        X_test = [self._line_to_vector(i) for i in X]
        return self.model.predict(X_test)

    def _line_to_vector(self,x):
        assert(type(x)==str)
        lineParts = x.split(' ')
        vectors = []
        for part in lineParts:
            try:
                vectors.append(self.w2v[part])
            except Exception:
                vectors.append(np.zeros(shape=(300,)))
        return np.mean(vectors)


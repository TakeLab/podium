from sklearn.metrics import f1_score


def f1(true, pred):
    return f1_score(true, pred)

def f1_multiclass(true, pred):
    ##TODO:uncomplete, missing one argument to be able to use in gridsearchcv
    return f1_score(y_true = true, y_pred=pred, average='macro')
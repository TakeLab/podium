from sklearn.metrics import f1_score


def f1(true, pred):
    return f1_score(true, pred)

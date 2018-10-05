import takepod.preproc.stemmer.croatian_stemmer as stem
import takepod.preproc.transform as trans
import takepod.dataload.pauzahr
import takepod.models.pauza_model as pauza_model
import sklearn.svm
import sklearn.metrics

def determine_label(ye):
    if ye<2.5:
        return -1
    if ye>4:
        return 1
    return 0
if __name__ == "__main__":
    ##LOAD DATA
    dataloader = takepod.dataload.pauzahr.PauzaHR()
    dataloader.download_and_extract()
    X_train, y_train = dataloader.load_data() ## shape (2277, 2) and (2277,)
    X_test, y_test = dataloader.load_data(train=False) ## shape (1033, 2) and (1033,)
    y_train = [determine_label(i) for i in y_train]
    y_test = [determine_label(i) for i in y_test]

    #remove data url
    X_train = [i[0] for i in X_train]
    X_test = [i[0] for i in X_test]

    X_train_bin = []
    y_train_bin = []
    for i in range(len(X_train)):
        if y_train[i]==-1 or y_train[i] == 1:
            X_train_bin.append(X_train[i])
            y_train_bin.append(y_train[i])

    X_test_bin = []
    y_test_bin = []
    for i in range(len(X_test)):
        if y_test[i]==-1 or y_test[i] == 1:
            X_test_bin.append(X_test[i])
            y_test_bin.append(y_test[i])

    input_data = list()
    input_data.extend(X_train)
    input_data.extend(X_test)

    word_to_ix = trans.create_word_to_index(input_data) ## it should be performed only on training set

    model = pauza_model.PauzaModelSVMBow(word_to_ix=word_to_ix)
    model.train(X_train_bin,y_train_bin)

    print(sklearn.metrics.accuracy_score(y_train_bin, model.test(X_train_bin )))
    print(sklearn.metrics.accuracy_score(y_test_bin, model.test(X_test_bin)))

    print(sklearn.metrics.f1_score(y_train_bin, model.test(X_train_bin)))
    print(sklearn.metrics.f1_score(y_test_bin, model.test(X_test_bin)))
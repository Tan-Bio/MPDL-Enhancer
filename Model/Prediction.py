from sklearn.metrics import accuracy_score, matthews_corrcoef, recall_score, roc_auc_score, average_precision_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def show_Performance(label, score):
    acc = accuracy_score(y_true=ind_Y, y_pred=label)
    mcc = matthews_corrcoef(y_true=ind_Y, y_pred=label)
    sn = recall_score(y_true=ind_Y, y_pred=label)
    sp = (acc * len(ind_Y) - sn * sum(ind_Y)) / (len(ind_Y) - sum(ind_Y))
    auc = roc_auc_score(y_true=ind_Y, y_score=score)
    aupr = average_precision_score(y_true=ind_Y, y_score=score)
    print('ACC: ' + str(acc))
    print('MCC: ' + str(mcc))
    print('SN: ' + str(sn))
    print('SP: ' + str(sp))
    print('AUC: ' + str(auc))
    print('AUPR: ' + str(aupr))


if __name__ == '__main__':
    file_path = 'optimum_train_feature.csv'
    ind_file_path = 'test_feature.csv'
    selected_index_path = 'optimal_feature_index.txt'
    dna2vec_path = 'DL_output_train.npy'
    ind_dna2vec_path = 'DL_output_test.npy'

    np.random.seed(46)

    w_vector = np.load(dna2vec_path)
    print("dna2vec vector dimension:", w_vector.shape)
    feature = pd.read_csv(file_path, header=None)
    print("structural feature vector dimension:", feature.shape)
    X = feature.iloc[:, :-1].values
    Y = feature.iloc[:, -1].values
    scale = StandardScaler()
    X = scale.fit_transform(X)
    biological_X = X
    X = np.concatenate((X, w_vector), axis=1)
    indexes = np.random.choice(X.shape[0], X.shape[0], replace=False)
    X = X[indexes]
    Y = Y[indexes]
    print("Dimension of merged feature vectors:", X.shape)

    ind_w_vector = np.load(ind_dna2vec_path)
    ind_feature = pd.read_csv(ind_file_path, header=None)
    ind_X = ind_feature.iloc[:, :-1].values
    ind_Y = ind_feature.iloc[:, -1].values
    F = open(selected_index_path, 'r')
    v = F.readlines()
    v = [int(i.strip()) for i in v]
    ind_X = ind_X[:, v]

    scale = StandardScaler()
    ind_X = scale.fit_transform(ind_X)
    ind_X = np.concatenate((ind_X, ind_w_vector), axis=1)
    ind_indexes = np.random.choice(ind_X.shape[0], ind_X.shape[0], replace=False)
    ind_X = ind_X[ind_indexes]
    ind_Y = ind_Y[ind_indexes]
    model = SVC(random_state=0, C=22, kernel='poly', gamma=0.001)
    model.fit(X, Y)
    pre_label = model.predict(ind_X)
    pre_score = model.decision_function(ind_X)
    show_Performance(pre_label, pre_score)

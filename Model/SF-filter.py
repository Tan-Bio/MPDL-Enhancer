import itertools
import numpy as np
from sklearn.preprocessing import LabelEncoder
import argparse
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd


def read_Fastas(fileName):
    with open(fileName, 'r') as file:
        v = []
        genome = ''
        for line in file:
            if line[0] != '>':
                genome += line.strip()
            else:
                v.append(genome.upper())
                genome = ''
        v.append(genome.upper())
        del v[0]
        return v


def save_CSV(X, Y, type, selected_features=None, feature_indices=None):
    if type == 'test':
        F = open('test_feature.csv', 'w')
    elif type == 'optimum':
        F = open('optimum_train_feature.csv', 'w')
        if feature_indices:
            with open(feature_indices, 'w') as idx_file:
                for idx in selected_features:
                    idx_file.write(str(idx) + '\n')

    if selected_features is not None:
        X_selected = X[:, selected_features]
    else:
        X_selected = X

    for x, y in zip(X_selected, Y):
        for each in x:
            F.write(str(each) + ',')
        F.write(str(int(y)) + '\n')
    F.close()


def read_Labels(fileName):
    with open(fileName, 'r') as file:
        v = []
        for line in file:
            if line != '\n':
                v.append((line.replace('\n', '')).replace(' ', ''))
        return v


def kmers(seq, k):
    v = []
    for i in range(len(seq) - k + 1):
        v.append(seq[i:i + k])
    return v


DNAelements = 'ACGT'
elements = DNAelements
m2 = list(itertools.product(elements, repeat=2))
m3 = list(itertools.product(elements, repeat=3))
m4 = list(itertools.product(elements, repeat=4))
m5 = list(itertools.product(elements, repeat=5))


def pseudoKNC(x, k):
    t = []
    for i in range(1, k + 1, 1):
        v = list(itertools.product(elements, repeat=i))
        for i in v:
            t.append(x.count(''.join(i)))
    return t


def monoMonoKGap(x, g):  # k=1, m=1
    m = m2
    t = []
    for i in range(1, g + 1, 1):
        V = kmers(x, i + 2)
        for gGap in m:
            C = 0
            for v in V:
                if v[0] == gGap[0] and v[-1] == gGap[1]:
                    C += 1
            t.append(C)
    return t


def monoDiKGap(x, g):  # k=1, m=2
    t = []
    m = m3
    for i in range(1, g + 1, 1):
        V = kmers(x, i + 3)
        for gGap in m:
            C = 0
            for v in V:
                if v[0] == gGap[0] and v[-2] == gGap[1] and v[-1] == gGap[2]:
                    C += 1
            t.append(C)
    return t


def diMonoKGap(x, g):  # k=2, m=1
    t = []
    m = m3
    for i in range(1, g + 1, 1):
        V = kmers(x, i + 3)
        for gGap in m:
            C = 0
            for v in V:
                if v[0] == gGap[0] and v[1] == gGap[1] and v[-1] == gGap[2]:
                    C += 1
            t.append(C)
    return t


def monoTriKGap(x, g):  # k=1, m=3
    t = []
    m = m4
    for i in range(1, g + 1, 1):
        V = kmers(x, i + 4)
        for gGap in m:
            C = 0
            for v in V:
                if v[0] == gGap[0] and v[-3] == gGap[1] and v[-2] == gGap[2] and v[-1] == gGap[3]:
                    C += 1
            t.append(C)
    return t


def triMonoKGap(x, g):  # k=3, m=1
    t = []
    m = m4
    for i in range(1, g + 1, 1):
        V = kmers(x, i + 4)
        for gGap in m:
            C = 0
            for v in V:
                if v[0] == gGap[0] and v[1] == gGap[1] and v[2] == gGap[2] and v[-1] == gGap[3]:
                    C += 1
            t.append(C)
    return t


def diDiKGap(x, g):  # k=2, m=2
    m = m4
    t = []
    for i in range(1, g + 1, 1):
        V = kmers(x, i + 4)
        for gGap in m:
            C = 0
            for v in V:
                if v[0] == gGap[0] and v[1] == gGap[1] and v[-2] == gGap[2] and v[-1] == gGap[3]:
                    C += 1
            t.append(C)
    return t


def diTriKGap(x, g):  # k=2, m=3
    m = m5
    t = []
    for i in range(1, g + 1, 1):
        V = kmers(x, i + 5)
        for gGap in m:
            C = 0
            for v in V:
                if v[0] == gGap[0] and v[1] == gGap[1] and v[-3] == gGap[2] and v[-2] == gGap[3] and v[-1] == gGap[4]:
                    C += 1
            t.append(C)
    return t


def triDiKGap(x, g):  # k=3, m=2
    m = m5
    t = []
    for i in range(1, g + 1, 1):
        V = kmers(x, i + 5)
        for gGap in m:
            C = 0
            for v in V:
                if v[0] == gGap[0] and v[1] == gGap[1] and v[2] == gGap[2] and v[-2] == gGap[3] and v[-1] == gGap[4]:
                    C += 1
            t.append(C)
    return t


def gen_Features(kGap, kTuple, x, y):
    feature = []
    t = pseudoKNC(x, kTuple)
    for item in t:
        feature.append(item)
    t = monoMonoKGap(x, kGap)
    for item in t:
        feature.append(item)
    t = monoDiKGap(x, kGap)
    for item in t:
        feature.append(item)
    t = monoTriKGap(x, kGap)
    for item in t:
        feature.append(item)
    t = diMonoKGap(x, kGap)
    for item in t:
        feature.append(item)
    t = diDiKGap(x, kGap)
    for item in t:
        feature.append(item)
    t = diTriKGap(x, kGap)
    for item in t:
        feature.append(item)
    t = triMonoKGap(x, kGap)
    for item in t:
        feature.append(item)
    t = triDiKGap(x, kGap)
    for item in t:
        feature.append(item)
    feature.append(y)
    return feature


def main(args):
    if args.testDataset == 0:
        dnashape_path = '../DNAshape/train_shape.csv'
        X = read_Fastas('../Dataset/train.txt')
        print('Loading sequence data complete')
        Y = read_Labels('../Dataset/train_label.txt')
        print('Loading label data complete')
    else:
        dnashape_path = '../DNAshape/test_shape.csv'
        X = read_Fastas('../Dataset/test.txt')
        print('Loading sequence data complete')
        Y = read_Labels('../Dataset/test_label.txt')
        print('Loading label data complete')

    Y = LabelEncoder().fit_transform(Y)
    dnashape = pd.read_csv(dnashape_path, low_memory=False, usecols=range(1, 999))
    print('Loading DNAshape data complete')

    assert len(X) == len(Y)
    T = []
    print('Start generating features')
    for x, y in zip(X, Y):
        feature = gen_Features(5, 3, x, y)
        T.append(feature)
        print(f'Current feature vectors: {len(T)}/{len(X)}')
    print('Feature generation complete')
    print('Merging DNAShape Features')
    T = np.array(T)
    X = T[:, :-1]
    X = np.concatenate((X, dnashape), axis=1)
    Y = T[:, -1]
    print('Feature merge complete')

    if args.testDataset == 1:
        print('Saving the test_feature.csv file')
        save_CSV(X, Y, type='test')
        print('Test dataset saved')
        return
    else:
        print('Start selecting the best feature subset')
        model = AdaBoostClassifier(algorithm='SAMME.R', n_estimators=470, learning_rate=1.0)    # 475
        model.fit(X, Y)
        feature_importance = model.feature_importances_
        top_features = np.argsort(feature_importance)[::-1][:355]
        save_CSV(X, Y, type='optimum', selected_features=top_features, feature_indices='optimal_feature_index.txt')
        print(f'Select the top {len(top_features)} important features and save them as optimum_train_feature.csv')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--testDataset', type=int, help='whether the generated set is a test set', default=0, choices=[0, 1])
    args = p.parse_args()
    main(args)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA


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


X = read_Fastas('../Dataset/train.txt')
train_label = np.array([1] * 1484 + [0] * 1484).astype(np.float32)
scatter_size = 12


def one_hot_encode(seq):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    return np.array([mapping[s] for s in seq])


X_encoded = np.array([one_hot_encode(seq) for seq in X])
X_flattened = np.array([x.flatten() for x in X_encoded])
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_flattened)
ica = FastICA(n_components=2)
X_ica = ica.fit_transform(X_flattened)
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_pca[train_label == 1, 0], X_pca[train_label == 1, 1], c='#FA7F6F', label='Enhancer', s=scatter_size)
plt.scatter(X_pca[train_label == 0, 0], X_pca[train_label == 0, 1], c='#82B0D2', label='Non-Enhancer', s=scatter_size)
plt.title('PCA of DNA Sequences')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_ica[train_label == 1, 0], X_ica[train_label == 1, 1], c='#FA7F6F', label='Enhancer', s=scatter_size)
plt.scatter(X_ica[train_label == 0, 0], X_ica[train_label == 0, 1], c='#82B0D2', label='Non-Enhancer', s=scatter_size)
plt.title('ICA of DNA Sequences')
plt.xlabel('Independent Component 1')
plt.ylabel('Independent Component 2')
plt.legend()
plt.savefig('PCA-ICA.png', dpi=300)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
warnings.filterwarnings("ignore")

original_features = np.load('original_features.npy')
cnn_output = np.load('cnn_output.npy')
lstm_output = np.load('lstm_output.npy')
softmax_output = np.load('softmax_output.npy')
train_label = np.array([1] * 1484 + [0] * 1484)

custom_cmap = ListedColormap(['#82B0D2', '#FA7F6F'])

tsne = TSNE(n_components=2, random_state=42)
original_tsne = tsne.fit_transform(original_features.reshape(original_features.shape[0], -1))
cnn_tsne = tsne.fit_transform(cnn_output.reshape(cnn_output.shape[0], -1))
lstm_tsne = tsne.fit_transform(lstm_output)
softmax_tsne = tsne.fit_transform(softmax_output)

plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 1)
plt.scatter(original_tsne[:, 0], original_tsne[:, 1], c=train_label, cmap=custom_cmap, marker='.')
plt.title('Input Layer Information')
plt.colorbar()

plt.subplot(1, 4, 2)
plt.scatter(cnn_tsne[:, 0], cnn_tsne[:, 1], c=train_label, cmap=custom_cmap, marker='.')
plt.title('CNN Implicit Information')
plt.colorbar()

plt.subplot(1, 4, 3)
plt.scatter(lstm_tsne[:, 0], lstm_tsne[:, 1], c=train_label, cmap=custom_cmap, marker='.')
plt.title('BiLSTM Implicit Information')
plt.colorbar()

plt.subplot(1, 4, 4)
plt.scatter(softmax_tsne[:, 0], softmax_tsne[:, 1], c=train_label, cmap=custom_cmap, marker='.')
plt.title('Softmax Output Layer Information')
plt.colorbar()

plt.tight_layout()
plt.savefig('t-SNE.png')
plt.show()
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
import shap


def SF_SHAP():
    file_path = '../Model/optimum_train_feature.csv'
    np.random.seed(46)
    feature = pd.read_csv(file_path, header=None)
    print("feature vector dimension:", feature.shape)
    X = feature.iloc[:, :-1].values
    Y = feature.iloc[:, -1].values
    scale = StandardScaler()
    X = scale.fit_transform(X)
    indexes = np.random.choice(X.shape[0], X.shape[0], replace=False)
    X = X[indexes]
    Y = Y[indexes]
    model = SVC(random_state=0, C=22, kernel='poly', gamma=0.001, probability=True)
    model.fit(X, Y)

    explainer = shap.KernelExplainer(model.predict_proba, X[:10])
    shap_values = explainer.shap_values(X[:10])
    shap.summary_plot(shap_values[1], X, plot_type="bar", plot_size=(12, 12))


def DL_SHAP():
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)
    lstm_output = np.load('lstm_output.npy')
    lstm_output_input = Input(shape=lstm_output.shape[1:])
    x = Dense(units=16, activation='relu')(lstm_output_input)
    x_output = Dense(units=2, activation='softmax', name='w2v_output')(x)
    model = Model(inputs=lstm_output_input, outputs=x_output)
    model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])

    explainer = shap.GradientExplainer(model, lstm_output)
    shap_values = explainer.shap_values(lstm_output)
    specific_output_shap_values = shap_values[0]
    shap.summary_plot(specific_output_shap_values, lstm_output, plot_type='dot', plot_size=(12, 12))


if __name__ == '__main__':
    SF_SHAP()
    DL_SHAP()

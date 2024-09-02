import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import Input, Conv1D, Dense, Concatenate, LSTM, Bidirectional, Lambda, add, MaxPooling1D
from keras.regularizers import l1, l2
from keras.models import Model
from keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.model_selection import train_test_split
import math
import warnings
from sklearn.metrics import accuracy_score, matthews_corrcoef, recall_score, average_precision_score, roc_auc_score
import random
import tensorflow as tf

warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)
tf.random.set_seed(48)


def step_decay(epoch):
    initial_lrate = 0.0005
    drop = 0.7
    epochs_drop = 5.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    print(lrate)
    return lrate


def multihead_Attention(input_feature, num_heads=6):
    input_dim = K.int_shape(input_feature)[-1]
    head_dim = input_dim // num_heads
    attention_heads = []
    for _ in range(num_heads):
        query = Dense(head_dim)(input_feature)
        key = Dense(head_dim)(input_feature)
        value = Dense(head_dim)(input_feature)
        attention_weights = Lambda(lambda x: K.softmax(
            K.batch_dot(x[0], K.permute_dimensions(x[1], (0, 2, 1))) / K.sqrt(K.cast(head_dim, dtype='float32'))))(
            [query, key])
        attention_output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attention_weights, value])
        attention_heads.append(attention_output)
    multihead_output = Concatenate(axis=-1)(attention_heads)
    output_feature = Dense(input_dim, activation='relu')(multihead_output)
    return output_feature


def show_Performance(pred, pred_class, true_class):
    acc = accuracy_score(true_class, pred_class)
    mcc = matthews_corrcoef(true_class, pred_class)
    sn = recall_score(true_class, pred_class)
    sp = (acc * len(true_class) - sn * sum(true_class)) / (len(true_class) - sum(true_class))
    auc = roc_auc_score(val_true_class, val_pred_class)
    aupr = average_precision_score(true_class, pred[:, 1])
    print('ACC: ' + str(acc))
    print('MCC: ' + str(mcc))
    print('SN: ' + str(sn))
    print('SP: ' + str(sp))
    print('AUC: ' + str(auc))
    print('AUPR: ' + str(aupr))


if __name__ == '__main__':
    train_label = np.array([1] * 1484 + [0] * 1484).astype(np.float32)
    train_label = to_categorical(train_label, num_classes=2)
    test_label = np.array([1] * 200 + [0] * 200).astype(np.float32)
    test_label = to_categorical(test_label, num_classes=2)

    train_dna2vec_path = '../Embedding/train_dna2vec_4mers.npy'
    test_dna2vec_path = '../Embedding/test_dna2vec_4mers.npy'
    train_dna2vec = np.load(train_dna2vec_path)
    test_dna2vec = np.load(test_dna2vec_path)

    w2v_input = Input(shape=(train_dna2vec.shape[1], train_dna2vec.shape[2]), name='w2v_input')
    w2v = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(0.01))(w2v_input)
    w2v2 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu', kernel_regularizer=l2(0.01))(w2v_input)
    w2v = add([w2v, w2v2])
    w2v = multihead_Attention(w2v)
    w2v = MaxPooling1D(pool_size=2)(w2v)
    w2v = Bidirectional(LSTM(units=16, return_sequences=False), name='before_Dense')(w2v)
    w2v = Dense(units=16, activation='relu')(w2v)
    w2v_output = Dense(units=2, name='w2v_output', activation='softmax')(w2v)
    w2v_model = Model(inputs=w2v_input, outputs=w2v_output)
    w2v_model.summary()
    w2v_model.compile(optimizer='nadam', loss={'w2v_output': 'categorical_crossentropy'}, metrics=['accuracy'])
    callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=2, mode='min', restore_best_weights=True),
                 LearningRateScheduler(step_decay)]
    train_X, val_X, train_Y, val_Y = train_test_split(train_dna2vec, train_label, test_size=0.135, stratify=train_label)
    w2v_model.fit(x=train_X, y=train_Y, epochs=50, batch_size=16, verbose=1, shuffle=True, callbacks=callbacks,
                  validation_data=({'w2v_input': val_X}, {'w2v_output': val_Y}))
    final_model = Model(inputs=w2v_model.input, outputs=w2v_model.get_layer('before_Dense').output)
    final_model.summary()

    val_pred = w2v_model.predict(val_X)
    val_pred_class = np.argmax(val_pred, axis=1)
    val_true_class = np.argmax(val_Y, axis=1)
    show_Performance(val_pred, val_pred_class, val_true_class)

    train_vector = final_model.predict(train_dna2vec)
    test_vector = final_model.predict(test_dna2vec)
    np.save('DL_output_train.npy', train_vector)
    np.save('DL_output_test.npy', test_vector)

import numpy as np
from keras.layers import TextVectorization, Embedding
import tensorflow as tf


def read_Fasta(fasta_file_name):
    seqs = []
    seqs_num = 0
    file = open(fasta_file_name)
    for line in file.readlines():
        if line.strip() == '':
            continue
        if line.startswith('>'):
            seqs_num = seqs_num + 1
            continue
        else:
            seq = line.strip()
            result1 = 'N' in seq
            result2 = 'n' in seq
            if result1 == False and result2 == False:
                seqs.append(seq)
    return seqs


def split_Sequence(seq, num):
    t_list = []
    for i in range(len(seq) - num + 1):
        t_list.append(seq[i:i + num])
    return " ".join(t_list)


def to_Text(seqs, k):
    dna_text = []
    for i in seqs:
        t = split_Sequence(i, k)
        dna_text.append(np.array(t))
    dna_text = np.array(dna_text)
    return dna_text


if __name__ == '__main__':
    np.random.seed(0)
    tf.random.set_seed(1)

    train_pos_seqs = np.array(read_Fasta('../Dataset/train_enhancers.fa'))
    train_neg_seqs = np.array(read_Fasta('../Dataset/train_nonenhancers.fa'))
    train_seqs = np.concatenate((train_pos_seqs, train_neg_seqs), axis=0)
    train_text = np.array(to_Text(train_seqs, 4))

    test_pos_seqs = np.array(read_Fasta('../Dataset/test_enhancers.fa'))
    test_neg_seqs = np.array(read_Fasta('../Dataset/test_nonenhancers.fa'))
    test_seqs = np.concatenate((test_pos_seqs, test_neg_seqs), axis=0)
    test_text = np.array(to_Text(test_seqs, 4))

    embedding_matrix = np.load('./embedding_matrix_4mers.npy')
    encoder = TextVectorization(max_tokens=256)
    encoder.adapt(train_text)
    train_vec = encoder(train_text)
    test_vec = encoder(test_text)

    embedding_layer = Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=embedding_matrix.shape[1],
        weights=[embedding_matrix],
        input_length=1,
        trainable=True
    )

    y_train = embedding_layer(train_vec)
    y_test = embedding_layer(test_vec)
    train_save_path = 'train_dna2vec_4mers.npy'
    test_save_path = 'test_dna2vec_4mers.npy'
    np.save(train_save_path, y_train)
    np.save(test_save_path, y_test)

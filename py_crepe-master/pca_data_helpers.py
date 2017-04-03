import string
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.decomposition import PCA


def load_ag_data():
    #train = pd.read_csv('data/ag_news_csv/train.csv', header=None)

    train = pd.read_csv("data/labeledTrainData.tsv", header= None, delimiter="\t", quoting=3)
    train= train[:2200]
    train = train.dropna()
    train = train[1:]


    x_train = train[2]
    x_train = np.array(x_train)

    y_train = train[1]
    y_train = to_categorical(y_train)

    #test = pd.read_csv('data/ag_news_csv/test.csv', header=None)
    test = pd.read_csv("data/labeledTrainData.tsv", header= None, delimiter="\t", quoting=3)
    test = test[:2000]
    test = test.dropna()
    test = test[1:]
    #test = test[:500]
    x_test =   test[2]
    x_test = np.array(x_test)

    y_test = test[1]
    y_test = to_categorical(y_test)


    return (x_train, y_train), (x_test, y_test)
def apply_pca(mydata):
    pca = PCA(n_components = 16)
    return pca.fit_transform(mydata)

def mini_batch_generator(x, y, vocab, vocab_size, vocab_check, maxlen,
                         batch_size=128):

    for i in xrange(0, len(x), batch_size):
        x_sample = x[i:i + batch_size]
        y_sample = y[i:i + batch_size]

        input_data = encode_data(x_sample, maxlen, vocab, vocab_size,
                                 vocab_check)
        pca_input = []
        for data in input_data:
            pca_input.append(apply_pca(data))

        pca_input = np.array(pca_input)
        yield(pca_input,y_sample)
        #yield (input_data, y_sample)


def encode_data(x, maxlen, vocab, vocab_size, check):
    #Iterate over the loaded data and create a matrix of size maxlen x vocabsize
    #In this case that will be 1014x69. This is then placed in a 3D matrix of size
    #data_samples x maxlen x vocab_size. Each character is encoded into a one-hot
    #array. Chars not in the vocab are encoded into an all zero vector.

    input_data = np.zeros((len(x), maxlen, vocab_size))
    for dix, sent in enumerate(x):
        counter = 0
        sent_array = np.zeros((maxlen, vocab_size))
        chars = list(sent.lower().replace(' ', ''))
        for c in chars:
            if counter >= maxlen:
                pass
            else:
                char_array = np.zeros(vocab_size, dtype=np.int)
                if c in check:
                    ix = vocab[c]
                    char_array[ix] = 1
                sent_array[counter, :] = char_array
                counter += 1
        input_data[dix, :, :] = sent_array

    return input_data


def shuffle_matrix(x, y):
    stacked = np.hstack((np.matrix(x).T, y))
    np.random.shuffle(stacked)
    xi = np.array(stacked[:, 0]).flatten()
    yi = np.array(stacked[:, 1:])

    return xi, yi


def create_vocab_set():
    #This alphabet is 69 chars vs. 70 reported in the paper since they include two
    # '-' characters. See https://github.com/zhangxiangxiao/Crepe#issues.

    alphabet = (list(string.ascii_lowercase) + list(string.digits) +
                list(string.punctuation) + ['\n'])
    vocab_size = len(alphabet)
    check = set(alphabet)

    vocab = {}
    reverse_vocab = {}
    for ix, t in enumerate(alphabet):
        vocab[t] = ix
        reverse_vocab[ix] = t

    return vocab, reverse_vocab, vocab_size, check

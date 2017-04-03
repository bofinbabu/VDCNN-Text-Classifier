from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.layers import Input, Dense, Dropout, Flatten, normalization,   Activation
from keras.layers.convolutional import Convolution1D, MaxPooling1D


def model(filter_kernels, dense_outputs, maxlen, nb_filter,
          cat_output):
    #Define what the input shape looks like
    inputs = Input(shape=(maxlen, 16), name='input', dtype='float32')

    #All the convolutional layers...
    conv = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[0],
                         border_mode='same', activation='relu',
                         input_shape=(maxlen, 16))(inputs)
    conv = MaxPooling1D(pool_length=3)(conv)

    conv1 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[1],  border_mode='same')(conv)

    conv1 = normalization.BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    conv1 = MaxPooling1D(pool_length=3)(conv1)

    #conv2 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[2],   border_mode='valid', activation='relu')(conv1)

    #conv3 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[3], border_mode='valid', activation='relu')(conv2)

    #conv4 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[4],    border_mode='valid', activation='relu')(conv3)

    conv5 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[5], border_mode='same')(conv1)

    conv5 = normalization.BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)

    #conv5 = MaxPooling1D(pool_length=3)(conv5)
    conv5 = Flatten()(conv5)

    #Two dense layers with dropout of .5
    #z = Dropout(0.7)(Dense(dense_outputs, activation='relu')(conv5))
    z = Dropout(0.5)(Dense(dense_outputs, activation='relu')(conv5))

    #Output dense layer with softmax activation
    pred = Dense(cat_output, activation='softmax', name='output')(z)

    model = Model(input=inputs, output=pred)

    sgd = SGD(lr=0.0081, momentum=0.9)
    adam  = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(loss='categorical_crossentropy', optimizer=adam,
                  metrics=['accuracy'])

    return model

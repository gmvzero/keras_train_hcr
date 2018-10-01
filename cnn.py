from keras.layers import Activation, Convolution2D, Dropout
from keras.layers import MaxPooling2D, Flatten, Dense
from keras.models import Sequential


def implement_cnn(width, height, depth, num_classes):
    model = Sequential()
    shape = (height, width, depth)
    model.add(Convolution2D(filters=16, kernel_size=(5, 5), padding='same',
                            name='image_array', input_shape=shape))
    model.add(Convolution2D(filters=16, kernel_size=(5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(.2))

    model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
    model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(.2))

    model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(.2))

    model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
    model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(.2))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model



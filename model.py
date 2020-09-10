import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from train import load_training_data, create_training_data
import pickle


def create_model(training_data):
    (features, labels) = training_data
    features = features / 255.0

    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu',
                     input_shape=features.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(1))
    print(model.summary)

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer='adam', metrics=['accuracy'])

    return model


def train_model(model, training_data, batch_size, epochs):
    (features, labels) = training_data
    features = features / 255.0

    model.fit(features, labels, batch_size=batch_size, epochs=epochs)


training_data = load_training_data()
train_model(create_model(training_data), training_data, 32, 2)

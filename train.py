import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv
import random as rand
import numpy as np
import pickle
import os

TRAINING_DIR = 'training-data/pre-editied-images/'
EDITIED_DIR = 'training-data/editied-images/'
CATEGORIES = ['covid', 'normal', 'pneumonia']
IMG_SIZE = 180

def create_training_data():
    training_data = []
    for category in CATEGORIES:
        path = os.path.join(TRAINING_DIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path): # maybe do try and except
            img_array = cv.imread(os.path.join(path, img), cv.IMREAD_GRAYSCALE)
            new_array = cv.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, class_num])
    rand.shuffle(training_data)
    features = []
    labels = []
    for feature, label in training_data:
        features.append(feature)
        labels.append(label)

    features = np.array(features).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    pickle_out_features = open(f'{EDITIED_DIR}/features.pickle', 'wb')
    pickle.dump(features, pickle_out_features)
    pickle_out_features.close()

    pickle_out_labels = open(f'{EDITIED_DIR}/labels.pickle', 'wb')
    pickle.dump(labels, pickle_out_labels)
    pickle_out_labels.close()
    return training_data

def load_training_data():
    pickle_in_features = open(f'{EDITIED_DIR}/features.pickle', 'wb')
    features = pickle.load(pickle_in_features)

    pickle_in_labels = open(f'{EDITIED_DIR}/labels.pickle', 'wb')
    labels = pickle.load(pickle_in_labels)

    training_data = []

    for feature, label in zip(features, labels):
        

    return []

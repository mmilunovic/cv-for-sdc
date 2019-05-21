import matplotlib.pyplot as plt
import pickle
import numpy as np
import csv
import random
import cv2
import tensorflow as tf
from sklearn.utils import shuffle
%matplotlib inline




def load_traffic_sign_data(training_file, testing_file, validation_file):
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    return train, test, valid

# Load pickled data
train, test, valid = load_traffic_sign_data("./train.p", "./test.p", "./valid.p")
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']
X_valid, y_valid = valid['features'], valid['labels']


n_train, n_test = X_train.shape[0], X_test.shape[0]
image_shape = X_train[0].shape
n_classes = np.unique(y_train).shape[0]

id_to_name = dict()

with open('./signnames.csv', 'r') as names_csv:
  reader = csv.reader(names_csv, delimiter=',')
  next(reader)
  id_to_name = {int(row[0]):row[1] for row in reader}
  names_csv.close()
 
train_distribution, test_distribution = np.zeros(n_classes), np.zeros(n_classes)

for c in range(n_classes):
  train_distribution[c] = np.sum(y_train == c) / n_train
  test_distribution[c] = np.sum(y_test == c) / n_test
  


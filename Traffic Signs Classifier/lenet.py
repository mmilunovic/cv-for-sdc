import matplotlib.pyplot as plt
import pickle
import numpy as np
import csv
import random
import tensorflow as tf
from sklearn.utils import shuffle
%matplotlib inline
from tensorflow.contrib.layers import flatten

initializer = tf.contrib.layers.xavier_initializer()

def LeNet(x, dropout=False):
  
  # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
  # 5x5 filter, input depth=1, output depth=6
  conv_w1 = tf.Variable(initializer(shape=(5, 5, 1, 6)))
  conv_b1 = tf.Variable(tf.zeros(6))
  conv_1 = tf.nn.conv2d(x, conv_w1, strides=[1,1,1,1], padding='VALID') + conv_b1
  
  # Activation
  conv_1 = tf.nn.relu(conv_1)
  
  # Pooling 1: Input = 28x28x6. Output = 14x14x6
  conv_1 = tf.nn.max_pool(conv_1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
  if dropout:
    conv_1 = tf.nn.dropout(conv_1, keep_probs)
    
  # Layer 2: Convolution. Input = 14x14x6. Output = 10x10x16
  conv_w2 = tf.Variable(initializer(shape=(5, 5, 6, 16)))
  conv_b2 = tf.Variable(tf.zeros(16))
  conv_2 = tf.nn.conv2d(conv_1, conv_w2, strides=[1,1,1,1], padding='VALID') + conv_b2
  
  # Activation
  conv_2 = tf.nn.relu(conv_2)
  
  # Polling 2: Input = 10x10x16. Output = 5x5x16
  conv_2 = tf.nn.max_pool(conv_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
  if dropout:
    conv_2 = tf.nn.dropout(conv_2, keep_probs)
    
    
  # Flatten. Input = 5x5x16. Output = 400
  fc0 = flatten(conv_2)
  #fc0 = keras.layers.flatten(conv_2)
  
  # Layer 3: Fully connected. Input = 400. Output = 120
  fc1_w = tf.Variable(initializer(shape=(400, 120)))
  fc1_b = tf.Variable(tf.zeros(120))
  fc1 = tf.add(tf.matmul(fc0, fc1_w),fc1_b)
  
  # Activation
  fc1 = tf.nn.relu(fc1)
  if dropout:
    fc1 = tf.nn.dropout(fc1, keep_probs)
    
  # Layer 4: Fully connected. Input = 120. Output = 84
  fc2_w = tf.Variable(initializer(shape=(120, 84)))
  fc2_b = tf.Variable(tf.zeros(84))
  fc2 = tf.add(tf.matmul(fc1, fc2_w), fc2_b)
  
  # Activation
  fc2 = tf.nn.relu(fc2)
  if dropout:
    fc2 = tf.nn.dropout(fc2, keep_probs)
    
  # Layer 5: Fully connected. Input = 84. Output = n_classes
  fc3_W  = tf.Variable(initializer(shape=(84, n_classes)))
  fc3_b  = tf.Variable(tf.zeros(n_classes))
  logits = tf.add(tf.matmul(fc2, fc3_W), fc3_b)
  
  
  return logits
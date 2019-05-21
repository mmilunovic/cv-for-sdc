import matplotlib.pyplot as plt
import pickle
import numpy as np
import csv
import random
import cv2
import tensorflow as tf
from sklearn.utils import shuffle
%matplotlib inline

def normalization(img):
  return (img - np.mean(img)) / np.std(img)

def to_gray(img):
  return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def histagram_equalize(img):
  return cv2.equalizeHist(img)

def clahe(img):
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  return clahe.apply(img)


def preprocess(imgs, labels):
  gray_imgs = list(map(to_gray, imgs))
  eq_imgs = list(map(clahe, gray_imgs))
  final_preprocessed_imgs = list(map(normalization, eq_imgs))
  x_imgs, x_labels = shuffle(final_preprocessed_imgs, labels)
  
  return x_imgs, x_labels


def display_mult(img_list, label_list, pred_list=[], columns=4, rows=5, cmap=''):        
    fig = plt.figure(figsize=(16,16));
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)       
        img = img_list[i-1]
        title =''
        if len(pred_list) > 0 :
            title = 'A:' + str(id_to_name[label_list[i-1]])
            title += '\n/P:' + str(id_to_name[pred_list[i-1]])
        else:
            title = id_to_name[label_list[i-1]]
        plt.title(title)
        plt.axis('off')
        if cmap != '':
            plt.imshow(img, cmap=cmap)
        else:
            plt.imshow(img)
    plt.show()
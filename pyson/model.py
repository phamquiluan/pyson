import cv2
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from glob import glob
import argparse
import os
import matplotlib.pyplot as plt
from utils import *
import keras

class multi_digit:
    def __init__(self, save_path='./model_4_number.h5'):
        self.model = keras.models.load_model(save_path)
    def pred(self, image):
        return predict_multi_number(image, self.model)


        
def padding(image, output_shape=(82,310,1)):
    image = 1-image / 255
    p = 40/image.shape[0]
    image = cv2.resize(image, (0, 0), fx=p, fy=p)
    zero_pad = np.zeros(shape=output_shape)
    h, w = image.shape[:2]
    l = 50#np.random.choice((output_shape[1] - w))
    t = (output_shape[0] - h) // 2
    zero_pad[t:t+h, l:l+w, 0]  = image
    return zero_pad

def slice_padding_image(img, stride = 82//2):
    h, w = img.shape[:2]
    images = []
    for i in range(h, w, stride):
        images.append(img[:,i-h:i])
    return np.array(images)

def combine(img):
    pad = np.zeros((82,310))
    ii = 0
    for i in range(82,310,41):
        pad[:,i-82:i]  = img[ii,:,:,0]
        ii+=1
    return pad

def find_numbers(img, min_w=2, min_h=15, max_h=100, check_small_number=False):

    img = 255 - img
    def sort_contours(cnts, method="left-to-right"):
        reverse = False
        i = 0
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True

        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))
        return (cnts, boundingBoxes)
    
    thresh = img.copy()
    cnts = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    import imutils
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sort_contours(cnts, "left-to-right")[0]
    small_i = 0
    rv = []
    
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if w >= min_w and (h >= min_h and h <= max_h):
            small_number = img[y:y + h, x:x + w]
            rv.append(255-small_number)
    return rv


def predict_multi_number(img, model):  
    def get_pred(result, conf):
        flatten = result.reshape([-1])
        conf = conf.reshape([-1])
        rv = []
        rv_conf = []
        for _,__ in zip(flatten, conf):
            if _ != 10:
                rv.append(_)
                rv_conf.append(__)
        return rv, rv_conf
    
    def conf_array(conf):
        rv = []
        pred_arg = np.argmax(conf, axis=-1)
        for _ in range(len(conf)):
            rv.append(conf[_][np.arange(5), pred_arg[_]])
        rv = np.array(rv)
        return rv
    
    origin = img.copy()
    small_images = find_numbers(img)
    padding_images = np.array([padding(img) for img in small_images])
    splitting_images = np.array([slice_padding_image(img) for img in padding_images])
    output = model.predict(splitting_images)
    conf = np.swapaxes(output, 1, 0)
    result = np.argmax(conf, axis=-1)
    conf = conf_array(conf)
    pred, conf = get_pred(result, conf)
    return pred, conf

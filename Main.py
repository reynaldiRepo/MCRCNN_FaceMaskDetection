from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import math
import cv2
import copy
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
import os
import json

from sklearn.metrics import average_precision_score

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.objectives import categorical_crossentropy

from keras.models import Model
from keras.utils import generic_utils
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers

#import custom modul
from GetData import get_data, createDataset, GetImageByIndex
from RoiPolling import RoiPoolingConv
from VGG import nn_base
from RPN import rpn_layer
from Iou import intersection, union, iou
from Loss import class_loss_cls, class_loss_regr, rpn_loss_cls, rpn_loss_regr
from Classifier import classifier_layer
from MCFE import DecoupledClassifier
from UTILS import *


class MCRCNN():
    def __init__(self):
        st = time.time()
        #rcnn model config
        configpath = os.path.join(os.getcwd(), "model", "model_config_MCRCNN.pickle")
        config_output_filename = configpath
        with open(config_output_filename, 'rb') as f_in:
            C = pickle.load(f_in)
        num_features = 512
        input_shape_img = (None, None, 3)
        input_shape_features = (None, None, num_features)
        img_input = Input(shape=input_shape_img)
        roi_input = Input(shape=(C.num_rois, 4))
        feature_map_input = Input(shape=input_shape_features)
        # define the base network (VGG here, can be Resnet50, Inception, etc)
        shared_layers = nn_base(img_input, trainable=True)
        # define the RPN, built on the base layers
        num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
        rpn_layers = rpn_layer(shared_layers, num_anchors)
        classifier = DecoupledClassifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(C.class_mapping))
        #Init Keras Model from spesific layer
        model_rpn = Model(img_input, rpn_layers)
        model_classifier = Model([feature_map_input, roi_input], classifier)
        #load weight
        print('Loading weights from {}'.format(C.model_path))
        model_rpn.load_weights(C.model_path, by_name=True)
        model_classifier.load_weights(C.model_path, by_name=True)
        #compile model
        model_rpn.compile(optimizer='sgd', loss='mse')
        model_classifier.compile(optimizer='sgd', loss='mse')
        self.rpn = model_rpn
        self.classifier = model_classifier
        self.rpn.summary()
        self.classifier.summary()

class FRCNN():
    def __init__(self):
        st = time.time()
        #rcnn model config
        configpath = os.path.join(os.getcwd(), "model", "model_v1_config.pickle")
        config_output_filename = configpath
        with open(config_output_filename, 'rb') as f_in:
            C = pickle.load(f_in)
        num_features = 512
        input_shape_img = (None, None, 3)
        input_shape_features = (None, None, num_features)
        img_input = Input(shape=input_shape_img)
        roi_input = Input(shape=(C.num_rois, 4))
        feature_map_input = Input(shape=input_shape_features)
        # define the base network (VGG here, can be Resnet50, Inception, etc)
        shared_layers = nn_base(img_input, trainable=True)
        # define the RPN, built on the base layers
        num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
        rpn_layers = rpn_layer(shared_layers, num_anchors)
        classifier = classifier_layer(feature_map_input, roi_input, C.num_rois, nb_classes=len(C.class_mapping))
        #Init Keras Model from spesific layer
        model_rpn = Model(img_input, rpn_layers)
        model_classifier = Model([feature_map_input, roi_input], classifier)
        #load weight
        print('Loading weights from {}'.format(C.model_path))
        model_rpn.load_weights(C.model_path, by_name=True)
        model_classifier.load_weights(C.model_path, by_name=True)
        #compile model
        model_rpn.compile(optimizer='sgd', loss='mse')
        model_classifier.compile(optimizer='sgd', loss='mse')
        self.rpn = model_rpn
        self.classifier = model_classifier
        self.rpn.summary()
        self.classifier.summary()


FRCNN()
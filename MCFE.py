import tensorflow as tf
import pandas as pd
import os
import json


from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout, Concatenate
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.objectives import categorical_crossentropy

from keras.models import Model
from keras.utils import generic_utils
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers

from RoiPolling import RoiPoolingConv

from VGG import nn_base

def MCFE(base_layer, num_fitur = 512):
    print("MCFE LAYER ==== Start")
    
    #dilated rate = [1,3,5,7]
    n_dilated = int(num_fitur // 4)
    
    #four dilated conv for mcfe
    d1 = Conv2D(n_dilated, (3,3), padding='same', dilation_rate=(1,1), activation="relu", name='atrous_conv1')(base_layer)
    d2 = Conv2D(n_dilated, (3,3), padding='same', dilation_rate=(3,3), activation="relu", name='atrous_conv2')(base_layer)
    d3 = Conv2D(n_dilated, (3,3), padding='same', dilation_rate=(5,5), activation="relu", name='atrous_conv3')(base_layer)
    d4 = Conv2D(n_dilated, (3,3), padding='same', dilation_rate=(7,7), activation="relu", name='atrous_conv4')(base_layer)
    
    mcfe = Concatenate(axis=3, name='concat_mcfe')([d1, d2, d3,d4])
    
    return mcfe

def DecoupledClassifier(base_layers, input_rois, num_rois, nb_classes = 4):
    input_shape = (num_rois,7,7,512)

    pooling_regions = 7
    
    MCFE_LAYER = MCFE(base_layers)
    
    # channles poll replace with MCFE
    # out_roi_pool.shape = (1, num_rois, channels, pool_size, pool_size)
    # num_rois (4) 7x7 roi pooling
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([MCFE_LAYER, input_rois])
    flatten = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
    
    # Branch dense layer for classifier
    dense_cls = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(flatten)
    dense_cls = TimeDistributed(Dropout(0.5))(dense_cls)
    dense_cls = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(dense_cls)
    dense_cls = TimeDistributed(Dropout(0.5))(dense_cls)
    
    # Branch dense layer for classifier
    dense_reg = TimeDistributed(Dense(4096, activation='relu', name='fc3'))(flatten)
    dense_reg = TimeDistributed(Dropout(0.5))(dense_reg)
    dense_reg = TimeDistributed(Dense(4096, activation='relu', name='fc4'))(dense_reg)
    dense_reg = TimeDistributed(Dropout(0.5))(dense_reg)

    # There are two output layer
    # out_class: softmax acivation function for classify the class name of the object
    # out_regr: linear activation function for bboxes coordinates regression
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(dense_cls)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(dense_reg)

    return [out_class, out_regr]


# def test():
#     input_shape_features = (None, None, 512)
#     feature_map_input = Input(shape=input_shape_features)
#     roi_input = Input(shape=(4, 4))
#     classifier = DecoupledClassifier(feature_map_input, roi_input , 4, nb_classes=3)
#     model_classifier = Model([feature_map_input, roi_input], classifier)
#     model_classifier.compile(optimizer='sgd', loss='mse')
#     model_classifier.summary()
#     with open('debug.txt','w') as fh:
#         # Pass the file handle in as a lambda function to make it callable
#         model_classifier.summary(print_fn=lambda x: fh.write(x + '\n'))
# test()
    
    
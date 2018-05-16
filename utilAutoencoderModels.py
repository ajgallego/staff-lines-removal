#!/usr/bin/python
# -*- coding: utf-8 -*- 
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import regularizers


# ----------------------------------------------------------------------------
def get_autoencoder(nb_layers, window_size, nb_filters, kernel_size):
    if nb_layers == 1:
        autoencoder, encoder, decoder = build_convolutional_autoencoder_l1(window_size, nb_filters, kernel_size)
    elif nb_layers == 2:
        autoencoder, encoder, decoder = build_convolutional_autoencoder_l2(window_size, nb_filters, kernel_size)
    elif nb_layers == 3:
        autoencoder, encoder, decoder = build_convolutional_autoencoder_l3(window_size, nb_filters, kernel_size)
    else:
        raise Exception('Unknown model')
    return autoencoder, encoder, decoder


# ----------------------------------------------------------------------------
def build_convolutional_autoencoder_l1(input_size, nb_filter=32, k_size=3):
    input_img = Input(shape=(1, input_size, input_size))

    conv1 = Convolution2D(nb_filter, k_size, k_size, activation='relu', border_mode='same', name='conv1')(input_img)    
    encoder = MaxPooling2D((2, 2), border_mode='same', name='encoder')(conv1)

    conv2 = Convolution2D(nb_filter, k_size, k_size, activation='relu', border_mode='same', name='conv3')(encoder)
    upsa1 = UpSampling2D((2, 2), name='upsa1')(conv2)
        
    decoder = Convolution2D(1, k_size, k_size, activation='sigmoid', border_mode='same')(upsa1)
    
    autoencoder = Model(input_img, decoder)
    
    return autoencoder, None, decoder


# ----------------------------------------------------------------------------
def build_convolutional_autoencoder_l2(input_size, nb_filter=32, k_size=3):
    input_img = Input(shape=(1, input_size, input_size))

    conv1 = Convolution2D(nb_filter, k_size, k_size, activation='relu', border_mode='same', name='conv1')(input_img)
    maxp1 = MaxPooling2D((2, 2), border_mode='same', name='maxp1')(conv1)
    
    conv2 = Convolution2D(nb_filter, k_size, k_size, activation='relu', border_mode='same', name='conv2')(maxp1)
    encoder = MaxPooling2D((2, 2), border_mode='same', name='encoder')(conv2)

    conv3 = Convolution2D(nb_filter, k_size, k_size, activation='relu', border_mode='same', name='conv3')(encoder)
    upsa1 = UpSampling2D((2, 2), name='upsa1')(conv3)
    
    conv4 = Convolution2D(nb_filter, k_size, k_size, activation='relu', border_mode='same', name='conv4')(upsa1)
    upsa2 = UpSampling2D((2, 2), name='upsa2')(conv4)
    
    decoder = Convolution2D(1, k_size, k_size, activation='sigmoid', border_mode='same')(upsa2)
    
    autoencoder = Model(input_img, decoder)
    
    return autoencoder, encoder, decoder


# ----------------------------------------------------------------------------
def build_convolutional_autoencoder_l3(input_size, nb_filter=32, k_size=3):
    input_img = Input(shape=(1, input_size, input_size))

    conv1 = Convolution2D(nb_filter, k_size, k_size, activation='relu', border_mode='same', name='conv1')(input_img)
    maxp1 = MaxPooling2D((2, 2), border_mode='same', name='maxp1')(conv1)
    
    conv2 = Convolution2D(nb_filter, k_size, k_size, activation='relu', border_mode='same', name='conv2')(maxp1)
    maxp2 = MaxPooling2D((2, 2), border_mode='same', name='maxp2')(conv2)
    
    conv3 = Convolution2D(nb_filter, k_size, k_size, activation='relu', border_mode='same', name='conv3')(maxp2)
    encoder = MaxPooling2D((2, 2), border_mode='same', name='encoder')(conv3)

    conv4 = Convolution2D(nb_filter, k_size, k_size, activation='relu', border_mode='same', name='conv4')(encoder)
    upsa1 = UpSampling2D((2, 2), name='upsa1')(conv4)
    
    conv4 = Convolution2D(nb_filter, k_size, k_size, activation='relu', border_mode='same', name='conv5')(upsa1)
    upsa2 = UpSampling2D((2, 2), name='upsa2')(conv4)
    
    conv5 = Convolution2D(nb_filter, k_size, k_size, activation='relu', border_mode='same', name='conv6')(upsa2)
    upsa3 = UpSampling2D((2, 2), name='upsa3')(conv5)
    
    decoder = Convolution2D(1, k_size, k_size, activation='sigmoid', border_mode='same')(upsa3)
    
    autoencoder = Model(input_img, decoder)
    
    return autoencoder, encoder, decoder


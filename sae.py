#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import time
import argparse
import util, utilDataGenerator, utilAutoencoderModels
from keras.callbacks import EarlyStopping
from keras import backend as K

util.init()

K.set_image_data_format('channels_first')


# ----------------------------------------------------------------------------
def prv_limit_datagenerator(data_generator, path, x_path, pfrom, pto):
    if pfrom != -1 or pto != -1:
        offset = 0 if pfrom == -1 else pfrom
        limit = len(data_generator) if pto == -1 else pto
        assert(limit <= len(data_generator))
        assert(offset >=0 and offset <= limit)

        input_files = []
        for i in range(limit - offset + 1):
            input_files.append( os.path.join(path, x_path, '%s_%04d.png' % (x_path, i+offset) ) )
        
        data_generator.truncate_to_set(input_files)


# ----------------------------------------------------------------------------
def prv_fit_lazy_loading(autoencoder, config, train_data_generator, test_data_generator, weights_filename, page_test=False):
    patience = 10
    early_stopping = EarlyStopping(monitor='loss', patience=patience)

    for se in range(config.nb_super_epoch):
        print(80 * "-")
        print("SUPER EPOCH: {}/{}".format(se+1, config.nb_super_epoch))
        train_data_generator.reset()
        train_data_generator.shuffle()

        for X_train, Y_train, _ in train_data_generator:
            if config.early_stopping_mode == 'p':
                early_stopping = EarlyStopping(monitor='loss', patience=patience)
            try:
                X_test, Y_test, _ = next(test_data_generator)
            except StopIteration:
                test_data_generator.reset()
                X_test, Y_test, _ = next(test_data_generator)

            autoencoder.fit(X_train, Y_train,
                                batch_size=config.batch_size,
                                nb_epoch=config.nb_epoch,
                                verbose=2,
                                validation_data=(X_test, Y_test),
                                callbacks=[early_stopping])

            if page_test == True:
                pos = test_data_generator.get_pos()
                util.run_test(autoencoder, test_data_generator, args.show)
                test_data_generator.set_pos(pos) 

    # Save final model
    autoencoder.save_weights(weights_filename, overwrite=True)


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Staff-line removal with Selectional Auto-Encoders')
parser.add_argument('-path',                        required=True,                          help='path to dataset')
parser.add_argument('--gray',                       action='store_true',                    help='Use gray pictures')

parser.add_argument('-modelpath',  default=None,    required=False,                         help='Path to the model used to initialize')

parser.add_argument('--ptest',                      action='store_true',                    help='Test after each page training')
parser.add_argument('--only_test',                  action='store_true',                    help='Only evaluate. Do not train')

parser.add_argument('-layers',     default=3,       dest='nb_layers',           type=int,   help='Nb layers [1, 2, 3]')
parser.add_argument('-window',     default=256,     dest='window_size',         type=int,   help='Window size')
parser.add_argument('-step',       default=-1,      dest='step_size',           type=int,   help='Step size. -1 to use window_size')
parser.add_argument('-filters',    default=96,      dest='nb_conv_filters',     type=int,   help='Nb. conv filters')
parser.add_argument('-ksize',      default=5,       dest='kernel_size',         type=int,   help='Kernel size (k x k)')
parser.add_argument('-th',         default=0.3,     dest='threshold',           type=float, help='Selectional threshold')

parser.add_argument('-super',      default=1,       dest='nb_super_epoch',      type=int,   help='Nb. super epochs')
parser.add_argument('-epoch',      default=200,     dest='nb_epoch',            type=int,   help='Nb. epochs')
parser.add_argument('-batch',      default=8,       dest='batch_size',          type=int,   help='Batch size')
parser.add_argument('-page',       default=25,      dest='page_train_size',     type=int,   help='Page size used for the training set')
parser.add_argument('-page_test',  default=-1,      dest='page_test_size',      type=int,   help='Page size used for the test set. -1 to use train size')
parser.add_argument('-esmode',     default='g',     dest='early_stopping_mode',             help='Early stopping mode. g=\'global\', p=\'per page\'')

parser.add_argument('-train_from', default=-1,                                  type=int,   help='Train from this number of file')
parser.add_argument('-train_to',   default=-1,                                  type=int,   help='Train to this number of file')

parser.add_argument('-test_from',  default=-1,                                  type=int,   help='Test from this number of file')
parser.add_argument('-test_to',    default=-1,                                  type=int,   help='Test to this number of file')

args = parser.parse_args()


if args.step_size == -1:
    args.step_size = args.window_size
if args.page_test_size == -1:
    args.page_test_size = args.page_train_size

x_path = 'GR' if args.gray else 'BW'

basepath = str(args.path)
train_path = os.path.join(basepath, 'TrainingData')
test_path = os.path.join(basepath, 'Test')
BASE_LOG_NAME = "{}_{}x{}_s{}_l{}_f{}_k{}_se{}_e{}_b{}_p{}_es{}".format(
                            x_path, args.window_size, args.window_size, args.step_size,
                            args.nb_layers, args.nb_conv_filters, args.kernel_size,
                            args.nb_super_epoch, args.nb_epoch, args.batch_size,
                            args.page_train_size, args.early_stopping_mode)
weights_filename = 'MODELS/model_weights_' + BASE_LOG_NAME + '.h5'

util.mkdirp('MODELS')


# Load data generators

train_data_generator = utilDataGenerator.LazyChunkGenerator(train_path, x_path, args.page_train_size, args.window_size, args.step_size)

test_step_size = args.window_size
test_data_generator = utilDataGenerator.LazyChunkGenerator(test_path, x_path, args.page_test_size, args.window_size, test_step_size)
test_data_generator.load_validation('BW')


# Train from / to

prv_limit_datagenerator(train_data_generator, train_path, x_path, args.train_from, args.train_to)

prv_limit_datagenerator(test_data_generator, test_path, x_path, args.test_from, args.test_to)


# Print configuration

print('# Processing path:', basepath)
print('# Img type:', x_path)
print('# Total train files:', len(train_data_generator))
print('# Total test files:', len(test_data_generator))

print('# Nb_layers:', args.nb_layers)
print('# Nb_conv_filters:', args.nb_conv_filters)
print('# Kernel_size:', args.kernel_size)
print('# Window size:', args.window_size)
print('# Step_size:', args.step_size)
print('# Nb_super_epoch:', args.nb_super_epoch)
print('# Nb_epoch:', args.nb_epoch)
print('# Batch_size:', args.batch_size)
print('# Threshold:', args.threshold)

print('# Page_train_size:', args.page_train_size)
print('# Page_test_size:', args.page_test_size)
print('# Early_stopping_mode:', args.early_stopping_mode)
print('# Log files:', BASE_LOG_NAME)


# Load autoencoder and weights

autoencoder, encoder, decoder = utilAutoencoderModels.get_autoencoder(args.nb_layers, args.window_size, args.nb_conv_filters, args.kernel_size)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

print(autoencoder.summary())

if args.modelpath != None:
    autoencoder.load_weights( args.modelpath )
    weights_filename = weights_filename + '_ftune.h5'


# Fit

if args.only_test == False:
    prv_fit_lazy_loading(autoencoder, args, train_data_generator, test_data_generator, weights_filename, args.ptest)


# Test

print(80 * "-")
print('Testing...')

if args.threshold == -1: # Evaluate threshold from 0.0 to 1.0
    for th in range(11):  # from 0 to 10
        th_value = float(th) / 10.0
        print('\nThreshold: {:.1f}'.format( th_value ) )
        util.run_test(autoencoder, test_data_generator, False, th_value)
else:
    util.run_test(autoencoder, test_data_generator, False, args.threshold)




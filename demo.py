#!/usr/bin/python
# -*- coding: utf-8 -*- 
from __future__ import print_function
import time
import argparse
import cv2
import numpy as np
import util
import utilDataGenerator, utilAutoencoderModels

from keras import backend as K

util.init()

K.set_image_data_format('channels_first')


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Staff-line removal with Selectional Auto-Encoders')
parser.add_argument('-imgpath',    required=True,   help='path to the image to process') 
parser.add_argument('-modelpath',  required=True,   help='Path to the model to load') 
parser.add_argument('-layers',     default=3,       dest='nb_layers',           type=int,   help='Nb layers [1, 2, 3]')
parser.add_argument('-window',     default=256,     dest='window_size',         type=int,   help='Window size')
parser.add_argument('-step',       default=-1,      dest='step_size',           type=int,   help='Step size. -1 to use window_size')
parser.add_argument('-filters',    default=96,      dest='nb_conv_filters',     type=int,   help='Nb. conv filters')
parser.add_argument('-ksize',      default=5,       dest='kernel_size',         type=int,   help='Kernel size (k x k)')
parser.add_argument('-th',         default=0.3,     dest='threshold',           type=float, help='Selectional threshold')
parser.add_argument('--demo',                       dest='demo',       action='store_true', help='Activate demo mode')
parser.add_argument('-save',     default=None,      dest='outFilename',                     help='Save the output image')
args = parser.parse_args()

if args.step_size == -1:
    args.step_size = args.window_size

if args.demo == False and args.outFilename == None:
    util.print_error("ERROR: no output mode selected\nPlease choose between --demo or -save options")
    parser.print_help()
    quit()

autoencoder, encoder, decoder = utilAutoencoderModels.get_autoencoder(args.nb_layers, args.window_size, args.nb_conv_filters, args.kernel_size)

autoencoder.load_weights(args.modelpath)


img = cv2.imread(args.imgpath, False)
img = np.asarray(img)
img = img.astype('float32') / 255.
    
finalImg = img.copy()

start_time = time.time()

for (x, y, window) in utilDataGenerator.sliding_window(img, stepSize=args.step_size, windowSize=(args.window_size, args.window_size)):
    if window.shape[0] != args.window_size or window.shape[1] != args.window_size:
        continue

    roi = img[y:(y + args.window_size), x:(x + args.window_size)].copy()
    roi = roi.reshape(1, 1, args.window_size, args.window_size)

    prediction = autoencoder.predict(roi)
    prediction = (prediction > args.threshold) * roi

    finalImg[y:(y + args.window_size), x:(x + args.window_size)] = prediction[0][0]

    if args.demo == True:
        demo_time = time.time()
        clone = finalImg.copy()
        cv2.rectangle(clone, (x, y), (x + args.window_size, y + args.window_size), (255, 255, 255), 2)
        cv2.namedWindow("Demo", 0)
        cv2.imshow("Demo", clone)
        cv2.waitKey(1)
        time.sleep(0.1)
        start_time += time.time() - demo_time

print( 'Time: {:.3f} seconds'.format( time.time() - start_time ) )

if args.demo == True:
    cv2.namedWindow("Demo", 0)
    cv2.imshow("Demo", finalImg)
    cv2.waitKey(0)

if args.outFilename != None :
    finalImg *= 255
    cv2.imwrite(args.outFilename, finalImg)



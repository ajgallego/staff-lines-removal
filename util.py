#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import os
import re
import sys
import random
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
def init():
    random.seed(1337)
    np.set_printoptions(threshold=np.nan)   # Print full np matrix
    np.random.seed(1337)                    # for reproducibility
    sys.setrecursionlimit(40000)

# ----------------------------------------------------------------------------
def print_error(str):
    print('\033[91m' + str + '\033[0m')

# ----------------------------------------------------------------------------
def LOG(fname, str):
    with open(fname, "a") as f:
        f.write(str+"\n")

# ----------------------------------------------------------------------------
def mkdirp(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)

# ----------------------------------------------------------------------------
# Return the list of files in folder
# ext param is optional. For example: 'jpg' or 'jpg|jpeg|bmp|png'
def list_files(directory, ext=None):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and ( ext==None or re.match('([\w_-]+\.(?:' + ext + '))', f) )]

# ----------------------------------------------------------------------------
def show_images(X_test, Y_test, prediction):
    n = 10
    titles = ['Original', 'GT', 'Prediction']
    plt.figure(figsize=(20, 6))
    for i in range(n):
        for idx, matrix in enumerate([X_test, Y_test, prediction]):
            ax = plt.subplot(3, n, i + 1 + (n*idx))
            plt.imshow(matrix[i][0])
            plt.gray()
            ax.set_title(titles[idx])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()

# ----------------------------------------------------------------------------
def run_test(autoencoder, test_data_generator, show_result=False, threshold=.2):
    test_data_generator.reset()
    results = []
    
    for X_test, Y_test, validation in test_data_generator:
        prediction = autoencoder.predict(X_test)
        prediction = (prediction > threshold) * validation
        result = calculate_metrics(prediction, Y_test)
        results.append(result)
        if show_result == True:
            show_images(X_test, Y_test, prediction)
    
    final = defaultdict(float) # Dictionary that does not fail if an unknown key is accessed 
    nb_total = 0
    for r in results:
        nb_items = r['nb_items']
        nb_total += nb_items    
        del r['nb_items']
        for k, v in r.items(): 
            final[k] += r[k] * nb_items
   
    for k, v in final.items():
        final[k] /= nb_total

    print(80*'-')
    print('Results...')        
    print('TP\tTN\tFP\tFN\tError\tAccuracy\tPrecision\tRecall\tFm\tSpecif.')
    print('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
            final['tp'],final['tn'],final['fp'],final['fn'],final['error'],
            final['accuracy'],final['precision'],final['recall'],final['fm'],final['specificity']))

# ----------------------------------------------------------------------------
def calculate_metrics(prediction, gt):
    gt = gt > 0.5
    not_prediction = np.logical_not(prediction)
    not_gt = np.logical_not(gt)

    tp = np.logical_and(prediction, gt)
    tn = np.logical_and(not_prediction, not_gt)
    fp = np.logical_and(prediction, not_gt)
    fn = np.logical_and(not_prediction, gt)

    tp = (tp.astype('float32')).sum()
    tn = (tn.astype('float32')).sum()
    fp = (fp.astype('float32')).sum()
    fn = (fn.astype('float32')).sum()

    gt = gt.astype('float32')    
    prediction = prediction.astype('float32')
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fm = 2 * (precision * recall) / (precision + recall)
    specificity = tn / (tn + fp)

    # http://lionel.kr.hsnr.de/%7Edalitz/data/publications/tpami-staffremoval.pdf
    # E = (#misclassified sp + #misclassfied non sp) / (#all sp + #all non sp), where sp = staff pixels
    difference = np.absolute(prediction - gt)
    totalSize = gt.shape[0] * gt.shape[1] * gt.shape[2] * gt.shape[3]
    error = difference.sum() / totalSize  
    
    return {'nb_items':gt.shape[0], 'tp':tp, 'tn':tn, 'fp':fp, 'fn':fn, 
            'error':error, 'accuracy':accuracy, 'precision':precision, 
            'recall':recall, 'fm':fm, 'specificity':specificity}


#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import math
import random
import cv2
import numpy as np
import util

GT_SUFIX = 'GT'

# ----------------------------------------------------------------------------
def normalize_data(data):
    if len(data) > 0:
        input_size = len(data[1])
        data = data.astype('float32') / 255.
        data = data.reshape(data.shape[0], 1, input_size, input_size)
    return data

# ----------------------------------------------------------------------------
class LazyFileLoader:
    def __init__(self, path, x_sufix, page_size):
        self.path = path
        self.x_sufix = x_sufix
        self.v_sufix = None
        self.array_x_files = util.list_files(os.path.join(path, x_sufix), ext='png')
        self.pos = 0
        self.page_size = page_size

    def __len__(self):
        return len(self.array_x_files)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def load_validation(self, v_sufix):
        self.v_sufix = v_sufix

    def truncate_to_size(self, truncate_to):
        self.array_x_files = self.array_x_files[0:truncate_to]

    def truncate_to_set(self, new_array_x_files):
        self.array_x_files = new_array_x_files

    def reset(self):
        self.pos = 0

    def get_pos(self):
        return self.pos

    def set_pos(self, pos):
        self.pos = pos

    def shuffle(self):
        random.shuffle(self.array_x_files)

    def __load_files(self, array_X_files):
        x_data = []
        y_data = []
        v_data = []
        for fname_x in array_X_files:
            fname_y = fname_x.replace(self.x_sufix, GT_SUFIX)
            x_data.append( cv2.imread(fname_x, False) )
            y_data.append( cv2.imread(fname_y, False) )
            if self.v_sufix != None and self.v_sufix != self.x_sufix:
                fname_v = fname_x.replace(self.x_sufix, self.v_sufix)
                v_data.append( cv2.imread(fname_v, False) )
        return np.asarray(x_data), np.asarray(y_data), np.asarray(v_data)

    def next(self):
        psize = self.page_size
        if self.pos + psize >= len(self.array_x_files): 
            if self.pos >= len(self.array_x_files):
                raise StopIteration
            else:
                psize = len(self.array_x_files) - self.pos

        print('> Loading page from', self.pos, 'to', self.pos + psize, 'of', self.path, '...')
        X_data, Y_data, V_data = self.__load_files(self.array_x_files[self.pos:self.pos + psize])
        self.pos += self.page_size

        X_data = normalize_data(X_data)
        Y_data = normalize_data(Y_data)
        if self.v_sufix == self.x_sufix:
            V_data = X_data
        elif self.v_sufix != None:
            V_data = normalize_data(V_data)

        return X_data, Y_data, V_data

# ----------------------------------------------------------------------------
# slide a window across the image
def sliding_window(img, stepSize, windowSize):
    n_steps_y = int( math.ceil( img.shape[0] / float(stepSize) ) )
    n_steps_x = int( math.ceil( img.shape[1] / float(stepSize) ) )
    
    for y in xrange(n_steps_y):
        for x in xrange(n_steps_x):
            posX = x * stepSize
            posY = y * stepSize
            posToX = posX + windowSize[0]
            posToY = posY + windowSize[1]

            if posToX > img.shape[1]: 
                posToX = img.shape[1] - 1
                posX = posToX - windowSize[0]
                
            if posToY > img.shape[0]: 
                posToY = img.shape[0] - 1
                posY = posToY - windowSize[1]

            yield (posX, posY, img[posY:posToY, posX:posToX]) # yield the current window

# ----------------------------------------------------------------------------
class LazyChunkGenerator(LazyFileLoader):
    def __init__(self, path, x_sufix, page_size, window_size, step_size):
        LazyFileLoader.__init__(self, path, x_sufix, page_size)
        self.window_size = window_size
        self.step_size = step_size

    def __generate_chunks(self, array_X_files):
        x_data = []
        y_data = []
        v_data = []
        for fname_x in array_X_files:
            fname_y = fname_x.replace(self.x_sufix, GT_SUFIX)
            img_x = cv2.imread(fname_x, False)
            img_y = cv2.imread(fname_y, False)
            if self.v_sufix != None and self.v_sufix != self.x_sufix:
                fname_v = fname_x.replace(self.x_sufix, self.v_sufix)
                img_v = cv2.imread(fname_v, False)

            for (x, y, window) in sliding_window(img_x, stepSize=self.step_size, windowSize=(self.window_size, self.window_size)):
                if window.shape[0] != self.window_size or window.shape[1] != self.window_size:  # if the window does not meet our desired window size, ignore it
                    continue
                x_data.append( window.copy() )
                y_data.append( img_y[y:y + self.window_size, x:x + self.window_size].copy() )
                if self.v_sufix != None and self.v_sufix != self.x_sufix:
                    v_data.append( img_v[y:y + self.window_size, x:x + self.window_size].copy() )

            """for y in xrange(0, img_x.shape[0], self.step_size):
                for x in xrange(0, img_x.shape[1], self.step_size):
                    if y + self.window_size > img_x.shape[0] or x + self.window_size > img_x.shape[1]:
                        continue
                    x_data.append( img_x[y:y + self.window_size, x:x + self.window_size].copy() )
                    y_data.append( img_y[y:y + self.window_size, x:x + self.window_size].copy() )
                    if self.v_sufix != None and self.v_sufix != self.x_sufix:
                        v_data.append( img_v[y:y + self.window_size, x:x + self.window_size].copy() )"""

        return np.asarray(x_data), np.asarray(y_data), np.asarray(v_data)

    def next(self):
        psize = self.page_size
        if self.pos + psize >= len(self.array_x_files): 
            if self.pos >= len(self.array_x_files):
                raise StopIteration
            else:
                psize = len(self.array_x_files) - self.pos

        print('> Loading page from', self.pos, 'to', self.pos + psize, 'of', self.path, '...')
        X_data, Y_data, V_data = self.__generate_chunks(self.array_x_files[self.pos:self.pos + psize])
        self.pos += self.page_size

        X_data = normalize_data(X_data)
        Y_data = normalize_data(Y_data)
        if self.v_sufix == self.x_sufix:
            V_data = X_data
        elif self.v_sufix != None:
            V_data = normalize_data(V_data)

        return X_data, Y_data, V_data

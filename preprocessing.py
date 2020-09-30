# Python 3.7.6
# -*- coding: utf-8 -*-
# Author: Ines Pisetta

import os
import random
import cv2
import fnmatch
import numpy as np
from tqdm import tqdm

char_list = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
len_char_list = len(char_list)

path = str(os.getcwd())

while True:
    windows = input('Are you using Windows? (y/n) ')
    if (windows != 'y' and windows != 'n'):
        windows = input('Are you using Windows? (y/n) ')
    else:
        break

if windows == 'y':
    path = path + r'\mjsynth.tar\mnt\ramdisk\max\90kDICT32px'
else:
    path = path + '/mnt/ramdisk/max/90kDICT32px'

width = 32
height = 128
dim = (width, height)

# convert each character of the word into a label (int)
def char_to_int(word):
    int_list = []
    for i, c in enumerate(word):
        int_list.append(char_list.index(c))
    return np.array(int_list)

def preprocessing(datafile, max_data):
    
    max_text_length = 0
    len_data = 0
    
    data_img = []
    data_txt = []
    
    # prepare storage
    if not os.path.exists('train'):
        os.mkdir('train')
    
    if not os.path.exists('val'):
        os.mkdir('val')
    
    if not os.path.exists('test'):
        os.mkdir('test')
    
    data_samples = []
    with open(path+datafile, 'r', encoding='utf-8') as tr:
        for line in tr.readlines():
            data_samples.append(line[1:].split(' ')[0])
    
    random_selection = random.sample(data_samples, max_data)
    print(len(random_selection))
    
    for filename in tqdm(random_selection):
        
        # get text seen on the image (which is the filename)
        text = filename.split('_')[-2]
        
        # keep track of the maximum text length
        if len(text) > max_text_length:
            max_text_length = len(text)            
        
        # convert image to gray scale
        img = cv2.cvtColor(cv2.imread(path+filename), cv2.COLOR_BGR2GRAY)
        
        # convert image to shape (32, 128, 1)
        w, h = img.shape
        
        if (w != width and h != height):
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        
        img = np.expand_dims(img , axis=2)
        
        # normalize image
        img = img/255
        
        data_img.append(img)
        data_txt.append(char_to_int(text))
        
        len_data += 1
        
        # break loops when enough data is collected
        if len_data == max_data:
            break
        
    print('Max text length: ', max_text_length)
    
    sav_str = datafile.split('_')[1].split('.')[0]
    
    # saving
    np.save(sav_str+'/'+sav_str+'_data.npy', np.array(data_img))
    np.save(sav_str+'/'+sav_str+'_labels.npy', np.array(data_txt))
    
    print(len(data_img))
    print(len(data_txt))
    
if __name__ == '__main__':
    preprocessing('/annotation_test.txt', 20000)
    preprocessing('/annotation_val.txt', 20000)
    preprocessing('/annotation_train.txt', 160000)
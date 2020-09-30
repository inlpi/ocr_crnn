# Python 3.7.6
# -*- coding: utf-8 -*-
# Author: Ines Pisetta

import os
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

char_list = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
len_char_list = len(char_list)

def pad_labels():
    
    path = os.getcwd()
    
    labels = []
    
    max_text_length = 0
    
    np_data = np.load(path+'/val/val_labels.npy', allow_pickle=True)
    np_data1 = np.load(path+'/test/test_labels.npy', allow_pickle=True)
    np_data2 = np.load(path+'/train/train_labels.npy', allow_pickle=True)
    
    datasets = ['train', 'val', 'test']
    sizes = [160000, 20000, 20000]
    
    for dataset in datasets:
        
        dset = np.load(path+'/'+dataset+'/'+dataset+'_labels.npy', allow_pickle=True).tolist()
        new_dset = []
        for arr in dset:
            #new_arr = arr.tolist()
            new_arr = torch.from_numpy(arr).long().cuda()
            if len(new_arr) > max_text_length:
                max_text_length = len(new_arr)
            new_dset.append(new_arr)
        labels.extend(new_dset)
    
    new_labels = pad_sequence(labels, padding_value=len_char_list).transpose(0,1).tolist()
    
    tmp = 0
    for dataset, size in zip(datasets, sizes):
        new_dataset = new_labels[tmp:tmp+size]
        res = torch.stack([torch.LongTensor(x) for x in new_dataset])
        torch.save(res, path+'/'+dataset+'/'+dataset+'_labels.pt')
        tmp += size
    
if __name__ == '__main__':
    pad_labels()
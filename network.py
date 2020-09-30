# Python 3.7.6
# -*- coding: utf-8 -*-
# Author: Ines Pisetta

import os
import torch
from torch import nn
import numpy as np
#from fast_ctc_decode import beam_search, viterbi_search
#import tensorflow as tf

char_list = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
# len_char_list = 62
len_char_list = len(char_list)

# classes are characters including a blank
classes = char_list+' '

path = os.getcwd()

batch_size = 256
epochs = 10

# for calculating the CTC loss
input_lengths = torch.full(size=(batch_size,), fill_value=31, dtype=torch.long)
target_lengths = torch.full(size=(batch_size,), fill_value=23, dtype=torch.long)


def decode_gold(labels):
    
    # restores the original words for the labels in a batch
    
    labels = labels.tolist()
    
    batch_gold = []
    
    for groundTruth in labels:
        word = ''.join([char_list[x] for x in groundTruth if x!=len_char_list])
        #print(word)
        batch_gold.append(word)
    
    return batch_gold


def decode_pred(predictions):
    
    """
    is supposed to apply beam search (or a similar algorithm) to decode the log probabilities of the softmax in the output (=prediction) into sequences of characters (words)
    done for each element in the batch
    """
    
    """
    # first approach
    predictions = list(predictions.detach().cpu().transpose(0,1))
    print(len(predictions))
    print(type(predictions))
    print(type(predictions[0]))
    print(predictions[0].shape)
    
    batch_pred = []
    
    for pred in predictions:
        seq, path = viterbi_search(pred.numpy(), classes)
        print(seq)
        seq, path = beam_search(pred.numpy(), classes, beam_size=5, beam_cut_threshold=0.00001)
        print(seq)
        batch_pred.append(seq)
    
    return batch_pred
    
    
    # second approach
    inp_tf = tf.convert_to_tensor(input_lengths.numpy())
    pred_np = predictions.detach().cpu().transpose(0,1).numpy()
    pred_tf = tf.convert_to_tensor(pred_np, dtype=tf.float32)
    out = tf.keras.backend.ctc_decode(pred_tf, inp_tf, greedy=True)
    
    print(out[0])
    print(out[0][0])
    """


class Dataset(torch.utils.data.Dataset):
    
    # class to prepare the datasets as input for the dataloaders

    def __init__(self, dset_str):
        
        
        data_path = path + '/' + dset_str + '/' + dset_str + '_data.npy'
        labels_path = path + '/' + dset_str + '/' + dset_str + '_labels.pt'
        
        #print(data_path)
        #print(labels_path)
        
        np_data = np.load(data_path, allow_pickle=True)
        t_labels = torch.load(labels_path)
        
        tmp_data = []
        tmp_labels = []
        
        for d,l in zip(np_data, t_labels):
            new_tensor = torch.FloatTensor(d).transpose(0,2)
            if new_tensor.shape != (1, 32, 128):
                continue
            else:
                tmp_data.append(new_tensor)
                tmp_labels.append(l)
        
        self.data = torch.stack(tmp_data)
        self.labels = torch.stack(tmp_labels)
        
        #print(self.data.shape)
        #print(self.labels.shape)
        
    def __len__(self):
        assert len(self.data) == len(self.labels)
        return len(self.data)
    
    def __getitem__(self, index):
        
        X = self.data[index]
        y = self.labels[index]

        return X, y


class Network(nn.Module):
    
    """
    CRNN architecture with a deep CNN (7 layers) combined with ReLU, Pooling and BatchNormalization layers
    RNN is a 2 layer LSTM
    output of the network are log probablities (LogSoftmax)
    """

    def __init__(self):
    
        super(Network, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=(2,1))
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()
        self.batch5 = nn.BatchNorm2d(512)
        
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU()
        self.batch6 = nn.BatchNorm2d(512)
        self.pool6 = nn.MaxPool2d(kernel_size=(2,1))
        
        self.conv7 = nn.Conv2d(512, 512, kernel_size=2)
        self.relu7 = nn.ReLU()
        
        self.lstm = nn.LSTM(512, 128, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        
        self.out = nn.Linear(256, len_char_list+1)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, x):
        
        x = x.cuda()
        #print('Input: ', x.shape)
        
        x = self.relu1(self.conv1(x))
        #print('Conv1: ', x.shape)
        x = self.pool1(x)
        #print('Pool1: ', x.shape)
        
        x = self.relu2(self.conv2(x))
        #print('Conv2: ', x.shape)
        x = self.pool2(x)
        #print('Pool2: ', x.shape)
        
        x = self.relu3(self.conv3(x))
        #print('Conv3: ', x.shape)
        
        x = self.relu4(self.conv4(x))
        #print('Conv4: ', x.shape)
        x = self.pool4(x)
        #print('Pool4: ', x.shape)
        
        x = self.relu5(self.conv5(x))
        #print('Conv5: ', x.shape)
        x = self.batch5(x)
        #print('Batch5: ', x.shape)
        
        x = self.relu6(self.conv6(x))
        #print('Conv6: ', x.shape)
        x = self.batch6(x)
        #print('Batch6: ', x.shape)
        x = self.pool6(x)
        #print('Pool6: ', x.shape)
        
        x = self.relu7(self.conv7(x))
        #print('Conv7: ', x.shape)
        
        x = torch.squeeze(x)
        #print('squeezed: ', x.shape)
        
        x = x.transpose(-2, -1)
        #print('squeezed: ', x.shape)
        
        x, h = self.lstm(x)
        #print('Lstm: ', x.shape)
        
        x = self.out(x)
        #print('Out: ', x.shape)
        
        x = self.softmax(x)
        #print('Softmax: ', x.shape)
        
        x = x.transpose(0,1)
        #print('Final: ', x.shape)
        
        return x


def train():
    
    train_on_gpu = torch.cuda.is_available()
    
    model = Network().cuda()
    
    optimizer = torch.optim.Adam(model.parameters())
    
    """
    index of blank character is the length of the char_list (and thus not in the char_list, because indexing starts at 0)
    in the network architecture, room for a 63th class is added in the output layer
    """
    criterion = nn.CTCLoss(blank=len_char_list, zero_infinity=True)
    
    trainloader = torch.utils.data.DataLoader(Dataset('train'), batch_size=batch_size, shuffle=False, drop_last=True)
    
    testloader = torch.utils.data.DataLoader(Dataset('val'), batch_size=batch_size, shuffle=False, drop_last=True)
    
    train_losses, test_losses, test_accuracy = [], [], []
    
    for epoch in range(epochs):
        
        epoch_loss = 0
        model.train()
        
        for local_batch, local_labels in trainloader:
            
            optimizer.zero_grad()
            
            prediction = model(local_batch.cuda())
            
            """
            CTC loss takes as input:
            log_probs [tensor of size = (input_length, batch_size, number of classes incl. blank)]
            targets [tensor of size = (batch_size, max_target_length)]
            input_lengths [tensor of size = (batch_size,) with the lengths of the inputs]
            target_lengths [tensor of size = (batch_size,) with the lengths of the targets]
            """
            loss = criterion(prediction, local_labels.cuda(), input_lengths, target_lengths)
            
            # multiply with 100 for better readability
            #print('Batchloss: ', loss.detach().item()*100)
            epoch_loss += loss.detach().item()*100
            
            loss.backward()
            optimizer.step()
        
        epoch_loss = epoch_loss/len(trainloader)
        train_losses.append(epoch_loss)
        print('Train Epoch: ', epoch+1, ' Loss on Train Set: ', epoch_loss)
        
        epoch_loss = 0
        accuracy = 0
        model.eval()
        
        for local_batch, local_labels in testloader:
            
            prediction = model(local_batch.cuda())
            
            #print(prediction.shape)
            #print(local_labels.shape)
            
            loss = criterion(prediction, local_labels.cuda(), input_lengths, target_lengths)
            
            # multiply with 100 for better readability
            epoch_loss += loss.detach().item()*100
            
            """
            # deocde_pred not yet working; decode_gold works properly
            gold = decode_gold(local_labels)
            predicted = decode_pred(prediction)
            
            tmp_accuracy = 0
            for seq_p, seq_g, in zip(predicted, gold):
                if seq_p == seq_g:
                    accuracy += 1
            accuracy += tmp_accuracy/batch_size
            """
        
        epoch_loss = epoch_loss/len(testloader)
        test_losses.append(epoch_loss)
        print('Val   Epoch: ', epoch+1, ' Loss on Val   Set: ', epoch_loss)
        
        """
        accuracy = accuracy/len(testloader)
        test_accuracy.append(accuracy)
        print('Val   Epoch: ', epoch+1, ' Accuracy Val  Set: ', accuracy, '\n')
        """
    torch.save(model.state_dict(), 'ocr_model_v2.pt')
    

def test(model):
    
    model_path = path + '/' + model
    
    model = Network().cuda()
    model.load_state_dict(torch.load(trained_model_path))
    model.eval()
    
    testloader = torch.utils.data.DataLoader(Dataset(dataset, 'test'), batch_size=5, shuffle=False)
    
    accuracy = 0
    
    for local_batch, local_labels in testloader:
        
        prediction = model(local_batch.cuda())
        
        gold = decode_gold(local_labels)
        predicted = decode_pred(prediction)
        
        """
        tmp_accuracy = 0
        for seq_p, seq_g, in zip(predicted, gold):
            if seq_p == seq_g:
                accuracy += 1
        accuracy += tmp_accuracy/batch_size
        """
    
    accuracy = accuracy/len(testloader)
    print('Accuracy Test Set: ', accuracy, '\n')

    
if __name__ == '__main__':
    train()
    #test('ocr_model_v1.pt')
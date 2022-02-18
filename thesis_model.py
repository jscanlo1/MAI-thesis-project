
import os
import string
import torch
import dataset
import argparse

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import itertools

import pandas as pd
import matplotlib.pyplot as plt

from transformers import BertTokenizer
from transformers import BertForSequenceClassification
import torch.nn as nn
import torch.optim as optim

#Import some libraries for calculating metrics
from sklearn.metrics import f1_score,precision_score,accuracy_score


#from nltk.corpus import stopwords
#from sklearn.preprocessing import LabelEncoder

from models.FakeNewsModel import FakeNewsModel
from models.EmotionDetectionModel import EmotionDetectionModel

bert_lr = 1e-5
weight_decay = 1e-5
lr = 5e-5
alpha = 0.95
max_grad_norm = 1.0


class Trainer(object):
    def __init__(self, model):
        self.model = model

        self.label_criterion = nn.CrossEntropyLoss()

        
        # self.multi_loss = MultiTaskLoss(2).cuda()

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0}
        ]

        '''
        self.optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=2e-5,
                     warmup=.1)
        '''

        self.optimizer = optim.Adam(optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, alpha)

    def train(self, data_loader):

        self.model.train()

        size = len(data_loader.dataset)

        loss_array = []

        for batch, (train_features, train_mask , train_token_type_ids, truth_label) in enumerate(data_loader):
            train_features = train_features.to(device)
            train_mask = train_mask.to(device)
            train_token_type_ids = train_token_type_ids.to(device)
            truth_label = truth_label.to(device)

            
            # Compute prediction and loss
            #This model uses a pretrained classification so some changes mayu be necessary
            truth_output = self.model(train_features, token_type_ids=None, attention_mask=train_mask, labels=truth_label)
            
            #print("Prediction: ", truth_output)
            #print("Actual: ", truth_label)
            loss = truth_output.loss
            #print("Loss Item: ",loss.item())
            
            #loss = self.label_criterion(truth_output, truth_label)

            # Backpropagation
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_array.append(loss.item())

            if batch % 20 == 0:
                loss, current = loss.item(), batch * len(train_features)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        loss = np.mean(loss_array)
        return loss
        

        

    def eval(self, data_loader):
        self.model.eval()
        loss_array = []
        pred_flat_array = []
        labels_flat_array = []

        size = len(data_loader.dataset)
        num_batches = len(data_loader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for train_features, train_mask , train_token_type_ids, truth_label in data_loader:
                train_features = train_features.to(device)
                train_mask = train_mask.to(device)
                train_token_type_ids = train_token_type_ids.to(device)
                truth_label = truth_label.to(device)


                truth_output = self.model(train_features, token_type_ids=None, attention_mask=train_mask, labels=truth_label)

                
                test_loss += truth_output.loss.item()

                logits = truth_output.logits.detach().cpu().numpy()
                #logits = truth_output.logits.detach().numpy()

                pred_flat = np.argmax(logits, axis=1).flatten()
                labels_flat = truth_label.to('cpu').numpy()
                #labels_flat = truth_label.numpy().flatten()

                correct += np.sum(pred_flat == labels_flat)

                pred_flat_array.append(pred_flat)
                labels_flat_array.append(labels_flat)

                

                #loss_array.append(loss.item())
        labels_flat_array = np.concatenate(labels_flat_array)
        pred_flat_array = np.concatenate(pred_flat_array)

        print("Labels: ", labels_flat_array[0])
        print("Preds: ", pred_flat_array[0])

        f1 = f1_score(labels_flat_array,pred_flat_array, average='weighted')
        acc = accuracy_score(labels_flat_array,pred_flat_array)
        prec = precision_score(labels_flat_array,pred_flat_array, average='weighted')


        #loss = np.mean(loss_array)
        print('Correct: ', correct)
        test_loss /= num_batches
        correct /= size

        print('Size: ', size)
        #print('Correct: ', correct)

        test_accuracy = (100*correct)
        #print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        return test_loss, acc, prec, f1


    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))









if __name__ == '__main__':

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=string, default='AAAI')
    #parser.add_argument('--dataset', type=string, default='AAAI')
    args = parser.parse_args()
    '''


    dataset_type = 'AAAI'

    num_labels = 2

    print(torch.version.cuda)
    print(torch.cuda.current_device())

    torch.cuda.device(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')




    #Read in data and load it
    #(train_set, dev_set, test_set), vocab = dataset.load_data(args.input_max_length)
    (train_set, val_set, test_set), vocab = dataset.load_data(512, dataset_type)

    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=32, shuffle=True)




    #Test that data is read in correctly

    
    '''
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_features, train_mask , train_token_type_ids,train_labels = next(itertools.islice(train_dataloader, 0, None))
    print(f"Feature batch shape: {train_features.size()}")
    print(f'Feature mask shape: {train_mask.size()}' )
    print(f'Feature token type ids shape: {train_token_type_ids.size()}' )
    print(f"Labels batch shape: {train_labels.size()}")
    text = train_features[4]
    mask = train_mask[4]
    token_type_ids = train_token_type_ids[4]
    label = train_labels[4]
    print(f"Text Tokens: {text}")
    
    print("Words: " , tokenizer.convert_ids_to_tokens(text))
    print('Mask: ', mask)
    print('Token Type IDs: ', token_type_ids)
    print(f"Label: {label}")
    '''


    #exit()

    #Create Full fake news model


    #Set up emotion model
    '''
    emotion_model_path = 'saved_models/emotion_model.pt'
    EmotionModel = EmotionDetectionModel()
    EmotionModel.load_state_dict(torch.load(emotion_model_path))
    EmotionModel.eval()

    model = FakeNewsModel(EmotionModel)

    '''

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)

    print(model)




    #Training
    trainer = Trainer(model)



    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = trainer.train(train_dataloader)
        print("Epoch: {}     Train Loss: {:.8f} ".format(t+1, train_loss))
        dev_loss, dev_acc, dev_prec, dev_F1 = trainer.eval(val_dataloader)
        print("Epoch: {}     Dev Loss: {:.8f}     Dev Acc: {:.4f}     Dev Prec {:.4f}     Dev F1 {:.4f}".format(t+1, dev_loss, dev_acc, dev_prec, dev_F1))
        
        print("---------------------------------")

    #test_loss, test_f1 = trainer.eval(test_loader)
    test_loss, test_acc, test_prec, test_F1 = trainer.eval(test_dataloader)
    print("Test Loss: {:.4f}    Test Acc: {:.4f}    Dev Prec {:.4f}    Dev F1 {:.4f}".format(test_loss, test_acc, test_prec, test_F1))


    #Save models
    save_path = 'saved_models/prelim_model.pt'

    trainer.save(save_path)



    #Load Model
    '''
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(save_path))
    model.eval()
    '''

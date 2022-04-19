from __future__ import print_function, division, unicode_literals
import itertools
import os
import torch
import numpy as np
import dataset_2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from torch.utils.data import RandomSampler

from models.FakeNewsModel import FakeNewsModel
from models.EmotionDetectionModel import EmotionDetectionModel
from models.ClassifierModel import ClassifierModel
from models.BertClassifierModel import BertClassifierModel
import json
import csv
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score,precision_score,accuracy_score

import torch.nn as nn
import torch.optim as optim



bert_lr = 1e-5
weight_decay = 1e-5
#lr = 5e-5
lr = 0.001
#lr = 0.0001
alpha = 0.95
max_grad_norm = 1.0

class Trainer(object):
    def __init__(self, model_bert,model_deepMoji,num_batches):
        self.model_bert = model_bert
        self.model_deepMoji = model_deepMoji

        self.loss_fn = nn.CrossEntropyLoss()

        # Set up params for thesis model
        # Must include provisions for frozen emotion detection model

        #self.model.EmotionModel.parameters().requires_grad = False
        #self.model.EmotionModel.bias.requires_grad = False
        '''

        for param in self.model.EmotionModel.parameters():
            param.requires_grad = False

        bert_params = set(self.model.bert.parameters())
        emotion_params = set(self.model.EmotionModel.parameters())
        other_params = list(set(self.model.parameters()) - bert_params - emotion_params)
        '''

        params = list(set(self.model_bert.parameters()))

        no_decay = ['bias', 'LayerNorm.weight']

        #Include Paramters for Loss [possibly e.g. multiLoss]

        optimizer_grouped_parameters = [
            
            {'params': params,
            'lr': lr,
            'weight_decay': weight_decay}
        ]

        self.optimizer = optim.Adam(optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, alpha)
        #self.scheduler = get_linear_schedule_with_warmup(optimizer_grouped_parameters,num_warmup_steps=3,num_training_steps=5*num_batches)

    def train(self, data_loader,epoch):

        


        self.model_bert.train()
        self.model_deepMoji.train()

        size = len(data_loader.dataset)

        loss_array = []

        for batch, (BERT_train_features, emoji_Train_Features, truth_label) in enumerate(data_loader):
            BERT_train_features = BERT_train_features.to(device).float()
            emoji_Train_Features = emoji_Train_Features.to(device).float()
            truth_label = truth_label.to(device)


            #This uses custom models
            truth_output = self.model_bert(BERT_train_features,emoji_Train_Features,epoch)
            loss = self.loss_fn(truth_output ,truth_label.flatten())
            

            # Backpropagation
            self.model_bert.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model_bert.parameters(), max_grad_norm)
            self.optimizer.step()
            

            loss_array.append(loss.item())

            if batch % 20 == 0:
                loss, current = loss.item(), batch * len(BERT_train_features)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        self.scheduler.step()
        loss = np.mean(loss_array)
        return loss   

    def eval(self, data_loader, sample_dataloader,epoch):
        self.model_bert.eval()
        self.model_deepMoji.eval()
        loss_array = []
        bert_pred_flat_array = []
        model_pred_flat_array = []
        labels_flat_array = []

        fake_news_features = []

        size = len(data_loader.dataset)
        num_batches = len(data_loader)
        bert_test_loss, deepMoji_test_loss = 0, 0

        with torch.no_grad():
            for BERT_train_features, emoji_Train_Features, truth_label in data_loader:
                BERT_train_features = BERT_train_features.to(device).float()
                emoji_Train_Features = emoji_Train_Features.to(device).float()
                truth_label = truth_label.to(device)

                
                #Custom Models
                bert_truth_output = self.model_bert(BERT_train_features, emoji_Train_Features,epoch)
                model_truth_output = self.model_deepMoji(BERT_train_features, emoji_Train_Features,epoch)
                
                bert_test_loss += self.loss_fn(bert_truth_output ,truth_label.flatten())
                deepMoji_test_loss += self.loss_fn(model_truth_output,truth_label.flatten())
                bert_logits = bert_truth_output.detach().cpu().numpy()
                model_logits = model_truth_output.detach().cpu().numpy()



                bert_pred_flat = np.argmax(bert_logits, axis=1).flatten()
                model_pred_flat = np.argmax(model_logits,axis=1).flatten()

                labels_flat = truth_label.to('cpu').cpu().numpy()
                #labels_flat = truth_label.numpy().flatten()


                bert_pred_flat_array.append(bert_pred_flat)
                model_pred_flat_array.append(model_pred_flat)
                labels_flat_array.append(labels_flat)

                

                #loss_array.append(loss.item())
        labels_flat_array_con = np.concatenate(labels_flat_array)
        bert_pred_flat_array_con = np.concatenate(bert_pred_flat_array)
        model_pred_flat_array_con = np.concatenate(model_pred_flat_array)

        #print("Labels: ", labels_flat_array[0])
        #print("Preds: ", pred_flat_array[0])

        f1_bert = f1_score(labels_flat_array_con,bert_pred_flat_array_con, average='weighted')
        acc_bert = accuracy_score(labels_flat_array_con,bert_pred_flat_array_con)
        prec_bert = precision_score(labels_flat_array_con,bert_pred_flat_array_con, average='weighted')

        f1_model = f1_score(labels_flat_array_con,model_pred_flat_array_con, average='weighted')
        acc_model = accuracy_score(labels_flat_array_con,model_pred_flat_array_con)
        prec_model = precision_score(labels_flat_array_con,model_pred_flat_array_con, average='weighted')


        #loss = np.mean(loss_array)
        #print('Correct: ', correct)
        bert_test_loss /= num_batches

        #print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        acc_diff =  acc_model - acc_bert

        B = len(sample_dataloader)

        sample_acc_array = []
        mean_sample_acc_diff = 0
        with torch.no_grad():
            for BERT_train_features, emoji_Train_Features, truth_label in sample_dataloader:
                BERT_train_features = BERT_train_features.to(device).float()
                emoji_Train_Features = emoji_Train_Features.to(device).float()
                truth_label = truth_label.to(device)

                
                #Custom Models
                bert_truth_output = self.model_bert(BERT_train_features, emoji_Train_Features,epoch)
                model_truth_output = self.model_deepMoji(BERT_train_features, emoji_Train_Features,epoch)

                bert_logits = bert_truth_output.detach().cpu().numpy()
                model_logits = model_truth_output.detach().cpu().numpy()

                print(len(bert_logits))



                bert_pred_flat = np.argmax(bert_logits, axis=1).flatten()
                model_pred_flat = np.argmax(model_logits,axis=1).flatten()

                labels_flat = truth_label.to('cpu').cpu().numpy()

                sample_bert_acc = accuracy_score(labels_flat,bert_pred_flat)
                sample_model_acc = accuracy_score(labels_flat,model_pred_flat)



                sample_acc_diff = sample_model_acc - sample_bert_acc
                sample_acc_array.append(sample_acc_diff)
                mean_sample_acc_diff += sample_acc_diff
            
        mean_sample_acc_diff /= B

        c = 0

        print(f"sample MEan: {mean_sample_acc_diff}")
        print(f"DIFF: {acc_diff}")

        for x in sample_acc_array:
            if abs(x - mean_sample_acc_diff) > abs(acc_diff):
                c = c + 1
        print(f"C: {c}")
        print(f"B: {B}")
        print(f"c/B: {c/B}")
        reject_null_hypo = False
        if c/B <=0.05:
            reject_null_hypo = True



        return reject_null_hypo

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


dataset_type = 'LIAR'




torch.cuda.empty_cache()



torch.cuda.device(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


#Read in data and load it
(train_set, val_set, test_set), vocab = dataset_2.load_data(512, dataset_type)


sampler = RandomSampler(test_set, replacement=True, num_samples=100000)


test_dataloader_sampler = DataLoader(test_set, sampler = sampler, batch_size=32)


#train_dataloader = DataLoader(train_set, batch_size=1, shuffle=False  )
#val_dataloader = DataLoader(val_set, batch_size=1, shuffle=False)
test_dataloader = DataLoader(test_set, batch_size=32, shuffle=False)

num_labels = vocab.num_labels()
num_batches = len(test_dataloader)


#Test that data is read in correctly

'''
   
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


train_features, emoji_feature, train_mask , train_token_type_ids,train_labels = next(itertools.islice(train_dataloader, 0, None))

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

#exit()
'''




#Model

save_path_bert = 'saved_models\LIAR_BERT__bootstrap.pt'
save_path_bert_with_deepMoji = 'saved_models\LIAR_BERT_with_deepMoji_bootstrap.pt'

model_bert = BertClassifierModel(num_labels)
model_bert.load_state_dict(torch.load(save_path_bert, map_location=torch.device('cpu')))
#model.eval()

model_deepMoji = ClassifierModel(num_labels)
model_deepMoji.load_state_dict(torch.load(save_path_bert_with_deepMoji, map_location=torch.device('cpu')))
#model.eval()


trainer = Trainer(model_bert, model_deepMoji,num_batches)

reject_null_hyp = trainer.eval(test_dataloader,test_dataloader_sampler,51)

print(f"Statistically Significant: {reject_null_hyp}")



exit()
test_loss, test_acc, test_prec, test_F1, test_fake_news_features = trainer.eval(test_dataloader)
print("Test Loss: {:.4f}    Test Acc: {:.4f}    Dev Prec {:.4f}    Dev F1 {:.4f}".format(test_loss, test_acc, test_prec, test_F1))

#Get Feature Vectors for every BERT ITEM
test_loss, test_acc, test_prec, test_F1, val_fake_news_features = trainer.eval(val_dataloader)
print("VAL Loss: {:.4f}    Test Acc: {:.4f}    Dev Prec {:.4f}    Dev F1 {:.4f}".format(test_loss, test_acc, test_prec, test_F1))

test_loss, test_acc, test_prec, test_F1, train_fake_news_features = trainer.eval(train_dataloader)
print("Train Loss: {:.4f}    Test Acc: {:.4f}    Dev Prec {:.4f}    Dev F1 {:.4f}".format(test_loss, test_acc, test_prec, test_F1))


#model_output = model(train_features[4], token_type_ids=None, attention_mask=train_mask[4])

#index_max = np.argmax(model_output)

#print(index_max)
'''
if(index_max == 1):
    print("Real")
else:
    print("Fake")

'''

torch.save(train_fake_news_features,"fake_news_features/LIAR/FN_LIAR_train.pt")
torch.save(val_fake_news_features,"fake_news_features/LIAR/FN_LIAR_val.pt")
torch.save(test_fake_news_features,"fake_news_features/LIAR/FN_LIAR_test.pt")

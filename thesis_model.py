
import os
import random
import torch
import dataset

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import itertools

import torch.nn as nn
import torch.optim as optim

#Import some libraries for calculating metrics
from sklearn.metrics import f1_score,precision_score,accuracy_score
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from models.FakeNewsModel import FakeNewsModel

#from torchMoji.torchmoji.model_def import torchmoji_emojis
#from torchMoji.torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH

bert_lr = 1e-5
weight_decay = 1e-5
#lr = 5e-5
lr = 0.1
alpha = 0.95
max_grad_norm = 1.0


class Trainer(object):
    def __init__(self, model,num_batches):
        self.model = model

        self.loss_fn = nn.CrossEntropyLoss()

        # Set up params for thesis model
        # Must include provisions for frozen emotion detection model

        '''

        for param in self.model.EmotionModel.parameters():
            param.requires_grad = False

        bert_params = set(self.model.bert.parameters())
        emotion_params = set(self.model.EmotionModel.parameters())
        other_params = list(set(self.model.parameters()) - bert_params - emotion_params)
        '''

        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, alpha)
        #self.scheduler = get_linear_schedule_with_warmup(optimizer_grouped_parameters,num_warmup_steps=3,num_training_steps=5*num_batches)

    def train(self, data_loader):

        self.model.train()

        size = len(data_loader.dataset)

        loss_array = []

        for batch, ( emoji_Train_Features ,truth_label) in enumerate(data_loader):

           #print(emoji_Train_Features)
            #print(truth_label)
            emoji_Train_Features = emoji_Train_Features.to(device).float()

            truth_label = truth_label.to(device)


            #This uses custom models
            truth_output = self.model(emoji_Train_Features)
            loss = self.loss_fn(truth_output ,truth_label.flatten())
            

            # Backpropagation
            self.model.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.optimizer.step()
            

            loss_array.append(loss.item())

            if batch % 20 == 0:
                loss, current = loss.item(), batch * len(emoji_Train_Features)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        self.scheduler.step()
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
            for emoji_Train_Features, truth_label in data_loader:
                emoji_Train_Features = emoji_Train_Features.to(device).float()
                truth_label = truth_label.to(device)

                
                #Custom Models
                truth_output = self.model(emoji_Train_Features)
                test_loss += self.loss_fn(truth_output ,truth_label.flatten())
                logits = truth_output.detach().cpu().numpy()
                



                pred_flat = np.argmax(logits, axis=1).flatten()
                labels_flat = truth_label.to('cpu').cpu().numpy()
                #labels_flat = truth_label.numpy().flatten()

                correct += np.sum(pred_flat == labels_flat)

                pred_flat_array.append(pred_flat)
                labels_flat_array.append(labels_flat)

                

                #loss_array.append(loss.item())
        labels_flat_array = np.concatenate(labels_flat_array)
        pred_flat_array = np.concatenate(pred_flat_array)

        #print("Labels: ", labels_flat_array[0])
        #print("Preds: ", pred_flat_array[0])

        f1 = f1_score(labels_flat_array,pred_flat_array, average='weighted')
        acc = accuracy_score(labels_flat_array,pred_flat_array)
        prec = precision_score(labels_flat_array,pred_flat_array, average='weighted')


        #loss = np.mean(loss_array)
        #print('Correct: ', correct)
        test_loss /= num_batches
        correct /= size

        #print('Size: ', size)
        #print('Correct: ', correct)

        test_accuracy = (100*correct)
        #print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        return test_loss, acc, prec, f1


    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))



if __name__ == '__main__':

    dataset_type = 'LIAR'

    writer = SummaryWriter()
    torch.cuda.empty_cache()

    
    seed = 123
    #seed = 111
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    


    torch.cuda.device(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    

    #Read in data and load it
    (train_set, val_set, test_set), vocab = dataset.load_data(512, dataset_type)
    print("THIS FAR")
    #print(train_set.deepMoji_input[0])
    #print(train_set.truth_labels[0])

    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=32, shuffle=True)

    num_labels = vocab.num_labels()
    num_batches = len(train_dataloader)

    '''
    train_features, train_labels = next(itertools.islice(train_dataloader, 0, None))
    print(f"Feature batch shape: {train_features.size()}")

    print(f"Labels batch shape: {train_labels.size()}")
    emo_probs = train_features[0]
    label = train_labels[0]
    print(f"Text Tokens: {emo_probs}")
    '''
    

    
    #INCLUDE SOME FLOW CONTROL HERE TO STREAMLINE
    #Create Full fake news model


    '''
    # Set up emotion model
    with open("tokenizer.pickle", "rb") as handle:
            t = pickle.load(handle)
    matrix_len = len(t.word_index) + 1
    embedding_matrix = torch.load("glove/embedding_matrix.pt")
    emotion_model_path = 'saved_models/sent2emo.pt'
    sent2emoModel = sent2emoModel(embedding_matrix=embedding_matrix,max_features = matrix_len ,num_labels=7).to(device)
    sent2emoModel.load_state_dict(torch.load(emotion_model_path))
    sent2emoModel.eval()
    '''

    model = FakeNewsModel(num_labels).to(device)
    print(model)

    torch.cuda.memory_summary(device=None, abbreviated=False)

    #Training
    trainer = Trainer(model,num_batches)

    epochs = 10
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = trainer.train(train_dataloader)
        print("Epoch: {}     Train Loss: {:.8f} ".format(t+1, train_loss))
        dev_loss, dev_acc, dev_prec, dev_F1 = trainer.eval(val_dataloader)
        print("Epoch: {}     Dev Loss: {:.8f}     Dev Acc: {:.4f}     Dev Prec {:.4f}     Dev F1 {:.4f}".format(t+1, dev_loss, dev_acc, dev_prec, dev_F1))
        test_loss, test_acc, test_prec, test_F1 = trainer.eval(test_dataloader)

        writer.add_scalars('Training Vs validation Vs test Loss',{'Training':train_loss, 'Validation': dev_loss, 'Test': test_loss }, t+1)
        writer.add_scalars('Val Vs Test acc',{'Validation': dev_acc, 'Test': test_acc }, t+1)
        writer.add_scalars('Val Vs Test F1',{'Validation': dev_F1, 'Test': test_F1 }, t+1)
        
        print("---------------------------------")
    
    print('Finished Training')

    writer.flush()
    print(model)

    #test_loss, test_f1 = trainer.eval(test_loader)
    test_loss, test_acc, test_prec, test_F1 = trainer.eval(test_dataloader)
    print("Test Loss: {:.4f}    Test Acc: {:.4f}    Dev Prec {:.4f}    Dev F1 {:.4f}".format(test_loss, test_acc, test_prec, test_F1))

    print(model.hidden[2].weight)

    np.savetxt('Final_Layer_Weights', model.hidden[2].weight.detach().cpu().numpy())

    #Save models
    #save_path = 'saved_models/LIAR_BERT_with_deepMoji.pt'
    #trainer.save(save_path)

    #Load Model
    '''
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(save_path))
    model.eval()
    '''

import os
import re
import torch
import json
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import Dataset
import pandas as pd
#import matplotlib.pyplot as plt
from collections import defaultdict
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from torchMoji.torchmoji.sentence_tokenizer import SentenceTokenizer
from torchMoji.torchmoji.model_def import torchmoji_emojis
from torchMoji.torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH


train_path = 'data/constraint_dataset/English_Train.xlsx'
val_path = 'data/constraint_dataset/English_Val.xlsx'
test_path = 'data/constraint_dataset/English_Test_With_Labels.xlsx'
'''
train_path = 'data/liar_dataset/train.tsv'
val_path = 'data/liar_dataset/valid.tsv'
test_path = 'data/liar_dataset/test.tsv'
'''
dataset_type = 'AAAI'


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]


if dataset_type == 'AAAI':
    train_data = pd.read_excel(train_path)
    val_data = pd.read_excel(val_path)
    test_data = pd.read_excel(test_path)

    train_text_items = train_data["tweet"]
    train_text_labels = train_data["label"]
    val_text_items = val_data["tweet"]
    val_text_labels = val_data["label"]
    test_text_items = test_data["tweet"]
    test_text_labels = test_data["label"]

elif dataset_type == 'LIAR':
   train_data = pd.read_csv(train_path, sep='\t',header=None)
   val_data = pd.read_csv(val_path, sep='\t',header=None)
   test_data = pd.read_csv(test_path, sep='\t',header=None)

   train_text_items = train_data.iloc[:,2]
   train_text_labels = train_data.iloc[:,1]
   val_text_items = val_data.iloc[:,2]
   val_text_labels = val_data.iloc[:,1]
   test_text_items = test_data.iloc[:,2]
   test_text_labels = test_data.iloc[:,1]
   
train_final = []
val_final = []
test_final = []


maxlen = 30

print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)

print('Loading model from {}.'.format(PRETRAINED_PATH))
model = torchmoji_emojis(PRETRAINED_PATH)
st = SentenceTokenizer(vocabulary, maxlen)

train_deepMoji = []
for x in (train_text_items):
    train_tokenized, _, _ = st.tokenize_sentences([x])
    train_deepMoji_ = model(train_tokenized)
    print(train_deepMoji_)
    train_deepMoji.append(train_deepMoji_[0])

#train_deepMoji = np.stack(train_deepMoji,axis=0)

print(train_deepMoji_) 

val_deepMoji = []
for x in val_text_items:
    val_tokenized, _, _ = st.tokenize_sentences([x])
    val_deepMoji_ = model(val_tokenized)
    val_deepMoji.append(val_deepMoji_[0])

#val_deepMoji = np.stack(val_deepMoji,axis=0)



test_deepMoji = []
for x in test_text_items:
    test_tokenized, _, _ = st.tokenize_sentences([x])
    test_deepMoji_ = model(test_tokenized)
    test_deepMoji.append(test_deepMoji_[0])

#test_deepMoji = np.stack(test_deepMoji,axis=0)


print(test_deepMoji[0])

torch.save(train_deepMoji,"deepMoji_inputs/AAAI/AAAI_train.pt")
torch.save(val_deepMoji,"deepMoji_inputs/AAAI/AAAI_val.pt")
torch.save(test_deepMoji,"deepMoji_inputs/AAAI/AAAI_test.pt")

'''
torch.save(max_train_deepMoji,"deepMoji_inputs/LIAR/LIAR_train.pt")
torch.save(max_val_deepMoji,"deepMoji_inputs/LIAR/LIAR_val.pt")
torch.save(max_test_deepMoji,"deepMoji_inputs/LIAR/LIAR_test.pt")

'''





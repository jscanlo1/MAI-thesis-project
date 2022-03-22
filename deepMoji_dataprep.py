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

dataset_type = 'AAAI'

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
   train_data = pd.read_excel(train_path)
   val_data = pd.read_excel(val_path)
   test_data = pd.read_excel(test_path)

   train_text_items = train_data.iloc[:,2]
   train_text_labels = train_data.iloc[:,1]
   val_text_items = val_data.iloc[:,2]
   val_text_labels = val_data.iloc[:,1]
   test_text_items = test_data.iloc[:,2]
   test_text_labels = test_data.iloc[:,1]
   
train_final = []
val_final = []
test_final = []


maxlen = 128

print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)

print('Loading model from {}.'.format(PRETRAINED_PATH))
model = torchmoji_emojis(PRETRAINED_PATH)
st = SentenceTokenizer(vocabulary, maxlen)
train_tokenized, _, _ = st.tokenize_sentences(train_text_items)
train_deepMoji = model(train_tokenized)
val_tokenized, _, _ = st.tokenize_sentences(val_text_items)
val_deepMoji = model(val_tokenized)
test_tokenized, _, _ = st.tokenize_sentences(test_text_items)
test_deepMoji = model(test_tokenized)

print(test_deepMoji.shape)
print(test_deepMoji)

torch.save(train_deepMoji,"deepMoji_inputs/AAAI/AAAI_train.pt")
torch.save(val_deepMoji,"deepMoji_inputs/AAAI/AAAI_val.pt")
torch.save(test_deepMoji,"deepMoji_inputs/AAAI/AAAI_test.pt")


test_stuff = torch.load("deepMoji_inputs/AAAI/AAAI_test.pt")
print(test_stuff)


#new_list = val_deepMoji + test_deepMoji

#print(new_list.shape)
#print(new_list)

















from __future__ import print_function, division, unicode_literals
import os
import torch
import numpy as np

from transformers import BertTokenizer

from models.FakeNewsModel import FakeNewsModel
import json
import csv
from keras.preprocessing.sequence import pad_sequences
from torchMoji.torchmoji.sentence_tokenizer import SentenceTokenizer
from torchMoji.torchmoji.model_def import torchmoji_emojis
from torchMoji.torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH




test_text = ['COVID causes cancer']

print(f"Test Text: {test_text}")

#Handle EMojis

maxlen = 30
print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)

print('Loading model from {}.'.format(PRETRAINED_PATH))
model = torchmoji_emojis(PRETRAINED_PATH)
st = SentenceTokenizer(vocabulary, maxlen)

emoji_tokenized, _, _ = st.tokenize_sentences(test_text)
emoji_prob = model(emoji_tokenized)

#Handle Bert tokens
bert_inputs = []
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
BERT_text = tokenizer.encode(test_text)
bert_inputs.append(BERT_text)
bert_inputs = pad_sequences(bert_inputs, maxlen=128, dtype="long", truncating="post", padding="post")

attention_masks = []
for seq in bert_inputs:
            seq_mask = [float(i>0) for i in seq]
            attention_masks.append(seq_mask)




#Model

save_path = 'saved_models\AAAI_BERT_with_deepMoji.pt'

model = FakeNewsModel(2)
model.load_state_dict(torch.load(save_path))

bert_input = bert_inputs[0]
attention_mask = attention_masks[0]

print(bert_input)
print(attention_mask)

model_output = model(bert_inputs[0],emoji_prob, token_type_ids=None, attention_mask=None)

index_max = np.argmax(model_output)

print(index_max)

if(index_max == 1):
    print("Real")
else:
    print("Fake")
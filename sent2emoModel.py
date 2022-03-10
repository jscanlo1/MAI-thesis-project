from unicodedata import bidirectional
import torch
import torch.nn as nn
from transformers import BertModel
import numpy as np

def create_emb_layer(weights_matrix, trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if not trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


class sent2emoModel(nn.Module):
    def __init__(self,embedding_matrix,max_features, num_labels, embed_size = 50):
        super(sent2emoModel, self).__init__()
        #self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        #self.embedding, num_embeddings, embedding_dim = create_emb_layer(embedding_matrix, trainable=False)

        
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        

        #Hidden Layer set to 128
        #Possibly Change
        self.lstm = nn.LSTM(embed_size,128,batch_first=True,bidirectional=True)
        
        #self.soft

        self.linear = nn.Linear(128*4 , 8)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(8, num_labels)

        

    def forward(self, text_input):

        #bert_output = self.bert(input_ids = text_input, attention_mask  = attention_mask)
        embedding_output = self.embedding(text_input)
        #lstm_output, (h,c) = self.lstm(embedding_output)
        h_lstm, _ = self.lstm(embedding_output)
        avg_pool = torch.mean(h_lstm, 1)
        max_pool, _ = torch.max(h_lstm, 1)
        conc = torch.cat(( avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        label_output = self.out(conc)

        return label_output
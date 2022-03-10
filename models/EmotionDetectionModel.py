from unicodedata import bidirectional
import torch
import torch.nn as nn
from transformers import BertModel

class EmotionDetectionModel(nn.Module):
    def __init__(self,num_labels):
        super(EmotionDetectionModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        self.lstm = nn.LSTM(768,256,batch_first=True,bidirectional=True)
        
        #self.soft

        self.label_output_layer = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(256*2, num_labels)
        )
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, text_input,token_type_ids,attention_mask):

        #print(f'Text Input: {text_input}')

        bert_output = self.bert(input_ids = text_input, attention_mask  = attention_mask)

        lstm_output, (h,c) = self.lstm(bert_output.last_hidden_state)
        hidden = torch.cat((lstm_output[:,-1, :256],lstm_output[:,-1, :256]), dim = -1)
        label_output = self.label_output_layer(hidden.view(-1,256*2))
        
        #print(f'Bert Output: {bert_output.pooler_output }    Size: {bert_output.pooler_output.size()}')        
        #bert_outputs = torch.cat(bert_outputs, dim=1)
        

        #output = torch.softmax(label_output)
        
        return label_output
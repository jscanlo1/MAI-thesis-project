import torch
import torch.nn as nn
from transformers import BertModel

class EmotionDetectionModel(nn.Module):
    def __init__(self,num_labels):
        super(EmotionDetectionModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.label_output_layer = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(768, num_labels)
        )
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, text_input,token_type_ids,attention_mask):

        #print(f'Text Input: {text_input}')

        bert_output = self.bert(input_ids = text_input, attention_mask  = attention_mask)

        #print(f'Bert Output: {bert_output.pooler_output }    Size: {bert_output.pooler_output.size()}')        
        #bert_outputs = torch.cat(bert_outputs, dim=1)
        label_output = self.label_output_layer(bert_output.pooler_output)

        #output = torch.softmax(label_output)
        
        return label_output
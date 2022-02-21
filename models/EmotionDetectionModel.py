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

    def forward(self, x):

        

        bert_output, _ = self.bert(x)
        #bert_outputs = torch.cat(bert_outputs, dim=1)
        label_output = self.label_output_layer(bert_output)
        
        return label_output
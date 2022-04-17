from unicodedata import bidirectional
import torch
import torch.nn as nn
from transformers import BertModel

class ClassifierModel(nn.Module):
    def __init__(self,num_labels):
        super(ClassifierModel, self).__init__()
        '''
        self.label_output_layer = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768 + 64, 500),
            nn.ReLU(),
            nn.Linear(500,num_labels)

        )
        '''
        
        
        self.label_output_layer = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(64, num_labels)

        )
        
        
        
        '''
        self.label_output_layer = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, num_labels)
        )
        '''
        


    def forward(self, text_input,emo_input):

        #full_feature = torch.cat((text_input, emo_input), dim=1)
        #label_output = self.label_output_layer(full_feature)

        #label_output = self.label_output_layer(text_input)
        label_output = self.label_output_layer(emo_input)

        
        return label_output
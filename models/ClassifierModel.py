from unicodedata import bidirectional
import torch
import torch.nn as nn
from transformers import BertModel

class ClassifierModel(nn.Module):
    def __init__(self,num_labels):
        super(ClassifierModel, self).__init__()
        
        '''
        self.label_output_layer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(768 + 64, 400),
            nn.ReLU(),
            nn.Linear(400,num_labels)

        )
        '''
        
        '''
        self.label_output_layer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(768 + 64, num_labels)
        )
        '''
        
        
        '''
        self.emo_output_layer = nn.Sequential(
            nn.Dropout(0.1),
            #nn.Linear(64, 64),
            #nn.ReLU(),
            nn.Linear(64,num_labels)

        )
        '''
        
        
        
        
        
        
        self.bert_output_layer = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, num_labels)
        )
        
        '''

        self.final_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(4,2)
        )
        '''
        
        


    def forward(self, text_input,emo_input,epoch):

        #full_feature = torch.cat((text_input, emo_input), dim=1)
        #label_output = self.label_output_layer(full_feature)

        label_output = self.bert_output_layer(text_input)
        #label_output = self.emo_output_layer(emo_input)
        '''
        if epoch < 50:

            label_output = self.emo_output_layer(emo_input)
        else:
            emo_output = self.emo_output_layer(emo_input)
            bert_output = self.bert_output_layer(text_input)

            label_output = self.final_layer(torch.cat((bert_output, emo_output), dim=1))
        '''


        
        return label_output
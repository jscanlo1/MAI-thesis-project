from unicodedata import bidirectional
import torch
import torch.nn as nn
from transformers import BertModel

class EmotionDetectionModel(nn.Module):
    def __init__(self,num_labels):
        super(EmotionDetectionModel, self).__init__()
        #64 emojis
        self.relu = nn.ReLU()

        self.hidden = nn.Sequential(
 
            nn.Linear(64,64),

            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,num_labels)
        )

    def forward(self, emoji_input):

        
        label_output = self.hidden(emoji_input)

        #output = torch.softmax(label_output)
        
        return label_output
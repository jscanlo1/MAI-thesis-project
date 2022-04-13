import torch
#import EmotionDetectionModel
import torch.nn as nn
from transformers import BertModel




class FakeNewsModel(nn.Module):
    def __init__(self,num_labels):
        super(FakeNewsModel, self).__init__()

        #64 emojis
        self.relu = nn.ReLU()

        self.hidden = nn.Sequential(
 
            nn.Linear(64,64),

            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,num_labels)
        )


    def forward(self, emoji_Input):

        
        emoji_output = self.hidden(emoji_Input)
        #print(emoji_output)

        return emoji_output
        
import torch
#import EmotionDetectionModel
import torch.nn as nn
from transformers import BertModel




class FakeNewsModel(nn.Module):
    def __init__(self,EmotionModel):
        super(FakeNewsModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.EmotionModel = EmotionModel

        self.label_output_layer = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(768, 2)
        )

    def forward(self, x,mask):

        

        bert_output, _ = self.bert(x,mask)
        emotion_output = self.EmotionModel(x)

        output = torch.cat((bert_output, emotion_output), dim=1)

        #bert_outputs = torch.cat(bert_outputs, dim=1)
        label_output = self.label_output_layer(output)
        
        return label_output
        
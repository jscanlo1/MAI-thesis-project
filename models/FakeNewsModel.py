import torch
#import EmotionDetectionModel
import torch.nn as nn
from transformers import BertModel




class FakeNewsModel(nn.Module):
    def __init__(self,num_labels, EmotionModel):
        super(FakeNewsModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.EmotionModel = EmotionModel

        self.bert_output_layer = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(768, num_labels)
        )

        self.label_output_layer = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(num_labels+4,num_labels)
        )



    def forward(self, text_input,token_type_ids,attention_mask):

        

        bert_output = self.bert(input_ids = text_input, attention_mask  = attention_mask)
        emotion_output = self.EmotionModel(text_input, token_type_ids=None, attention_mask=attention_mask)

        bert_emotion_output = self.bert_output_layer(bert_output.pooler_output)

        output = torch.cat((bert_emotion_output, emotion_output), dim=1)

        #bert_outputs = torch.cat(bert_outputs, dim=1)
        
        label_output = self.label_output_layer(output)
        
        return label_output
        
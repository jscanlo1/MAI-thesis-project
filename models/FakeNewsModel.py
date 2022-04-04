import torch
#import EmotionDetectionModel
import torch.nn as nn
from transformers import BertModel




class FakeNewsModel(nn.Module):
    def __init__(self,num_labels):
        super(FakeNewsModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        #64 emojis
        self.emoji_output_layer = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(64,num_labels),
            nn.ReLU()
        )

        self.bert_output_layer = nn.Sequential(
            #Possibly Exclude first two lines
            nn.Dropout(0.1),
            nn.Linear(768,num_labels),
            nn.ReLU()
            
            #nn.Linear(num_labels,num_labels)
        )
        self.final_output_layer = nn.Sequential(
            nn.Linear(64 + num_labels,num_labels)
        )
        


    def forward(self, text_input,emoji_Input,token_type_ids,attention_mask):
        bert_output = self.bert(input_ids = text_input, attention_mask  = attention_mask)
        #emoji_output = self.emo_layer(emoji_Input)
        bert_output_ = self.bert_output_layer(bert_output.pooler_output)


        output = torch.cat((bert_output_, emoji_Input), dim=1)

        label_output = self.final_output_layer(output)
        
        return label_output
        
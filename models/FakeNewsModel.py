import torch
#import EmotionDetectionModel
import torch.nn as nn
from transformers import BertModel




class FakeNewsModel(nn.Module):
    def __init__(self,num_labels):
        super(FakeNewsModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        #self.deepMoji_model = deepMoji_model


        #64 emojis
        self.relu = nn.ReLU()

        self.label_output_layer = nn.Sequential(
            #Possibly Exclude first two lines
            nn.Dropout(0.2),
            nn.Linear(768,num_labels),
            #nn.ReLU(),
            
            #nn.Linear(num_labels,num_labels)
        )



    def forward(self, text_input,emoji_Input,token_type_ids,attention_mask):
        #print(text_input.shape)
        #print(emoji_Input.shape)
        #print(emoji_Input)
        #print(emoji_Input)
        bert_output = self.bert(input_ids = text_input, attention_mask  = attention_mask)
        #print(bert_output)
        #emotion_output = self.deepMoji_model(emoji_Input)
        

        #bert_emotion_output = self.bert_output_layer(bert_output.pooler_output)
        #print(bert_emotion_output.shape)
        #output = torch.cat((bert_output.pooler_output, emoji_Input), dim=1)

        #bert_outputs = torch.cat(bert_outputs, dim=1)
        #print(output.shape)
        label_output = self.label_output_layer(bert_output.pooler_output)
        
        return label_output
        
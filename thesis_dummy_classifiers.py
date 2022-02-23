import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC

from sklearn.pipeline import Pipeline

from transformers import BertTokenizer
from sklearn.metrics import f1_score, precision_score, accuracy_score

import dataset

# This programme aims to develop some dummy classifiers for the relevant datasets.
# These are implemented using the sklearn library 



dataset_type = 'LIAR'


(train_set, val_set, test_set), vocab = dataset.load_data(512, dataset_type)

train_data = pd.DataFrame(train_set.text)
val_data = pd.DataFrame(val_set.text)
test_data = pd.DataFrame(test_set.text)

train_label = pd.DataFrame(train_set.truth_labels)
val_label = pd.DataFrame(val_set.truth_labels)
test_label = pd.DataFrame(test_set.truth_labels)


print(train_data.head())
print(train_label.head())
    
pipeline = Pipeline([
        ('bow', CountVectorizer()),  
        ('tfidf', TfidfTransformer()),  
        ('c', LinearSVC())
    ])
fit = pipeline.fit(train_data, train_label)

label_preds = pipeline.predict(val_data)
#exit()
#model = DummyClassifier()

#model = LinearSVC(C=0.1).fit(train_data, train_label)

#label_preds = model.predict(val_data)

dev_F1 = f1_score(val_label,label_preds, average='weighted')
dev_acc = accuracy_score(val_label,label_preds)
dev_prec = precision_score(val_label,label_preds, average='weighted')

print("Dev Acc: {:.4f}     Dev Prec {:.4f}     Dev F1 {:.4f}".format( dev_acc, dev_prec, dev_F1))


#label_preds = model.predict(test_data)
label_preds = pipeline.predict(test_data)

test_F1 = f1_score(test_label,label_preds, average='weighted')
test_acc = accuracy_score(test_label,label_preds)
test_prec = precision_score(test_label,label_preds, average='weighted')

print("Test Acc: {:.4f}     Test Prec {:.4f}     Test F1 {:.4f}".format( test_acc, test_prec, test_F1))

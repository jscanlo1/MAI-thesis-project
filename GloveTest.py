import pickle
import bcolz
import numpy as np


vectors = bcolz.open(f'glove/6B.50.dat')[:]
words = pickle.load(open(f'glove/6B.50_words.pkl', 'rb'))
word2idx = pickle.load(open(f'glove/6B.50_idx.pkl', 'rb'))

glove = {w: vectors[word2idx[w]] for w in words}


print(glove['the'])
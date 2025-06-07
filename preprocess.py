import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter

dataset_path = 'flickr8k/captions.txt'

data = pd.read_csv(dataset_path)
captions = data['caption'].to_list()

print(captions[:5])

def make_lowercase(text):
    return text.lower()

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

captions = [ make_lowercase(caption) for caption in captions ]
captions = [ remove_punctuation(caption) for caption in captions ]

print(captions[:5])

'''
# Tokenize with spaCy, make sure to add the start and end tokens
def tokenize_spacy(captions):
    tokenized = []
    
    for doc in nlp.pipe(captions, batch_size=1000, n_process=-1):
        tokenized.append( ['<start>'] + [ token.text for token in doc ] + ['<end>'] )
    
    return tokenized
    #return ['<start>'] + [ token.text for token in nlp(text) ] + ['<end>']
'''

def tokenize_basic(text):
    tokens = text.strip().split() # clean whitespaces
    return ['<start>'] + tokens + ['<end>']


tokenized = [ tokenize_basic(caption) for caption in captions ]
print(tokenized[:5])

def create_vocab(tokenized_captions, specials):
    counter = Counter()

    # Note: this doesn't work with a set!
    for tokens in tokenized_captions:
        counter.update(tokens)

    seen_words = [ word for word in counter.keys() if word not in specials ]

    vocab_to_idx = { word: (idx + len(specials)) for idx, word in enumerate(seen_words) }
    vocab_to_idx = dict(vocab_to_idx, **{ special_token: idx for idx, special_token in enumerate(specials) })


    idx_to_vocab = { idx: word for word, idx in vocab_to_idx.items() }

    return vocab_to_idx, idx_to_vocab

specials = ['<pad>', '<start>', '<end>', '<unk>']

vocab_to_idx, idx_to_vocab = create_vocab(tokenized, specials)
vocab_len = len(vocab_to_idx)
print(vocab_len)

def tokens_to_idx(tokenized_caption, vocab_to_idx):
    return [ vocab_to_idx[token] for token in tokenized_caption ]

captions_idx = [ tokens_to_idx(caption, vocab_to_idx) for caption in tokenized ]

def pad_captions(captions_idx, pad_token_index=0):
    max_len = max([ len(caption) for caption in captions_idx ])

    for caption in captions_idx:
        while len(caption) < max_len:
            caption.append(pad_token_index)

    return captions_idx

captions_idx = pad_captions(captions_idx, 0)
captions_idx = np.array(captions_idx)

pd.DataFrame( zip(idx_to_vocab.keys(), idx_to_vocab.values()) , columns=['ID', 'token']).to_csv('captions_idx_to_vocab.csv', index=False, header=True)

np.savetxt('captions_tokenized_data2.txt', captions_idx, fmt='%d')
#np.savetxt('captions_idx_to_vocab.txt', np.array(idx_to_vocab.items()), fmt='%s')

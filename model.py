#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
import re
import pandas as pd
import os
import nltk

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

from tqdm import tqdm
from nltk.corpus import stopwords

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec


# preprocess data


nltk.download('stopwords')
# mit Stemmer: 
from nltk.stem.snowball import SnowballStemmer 

def replace_umlauts(text:str) -> str:
    """replace special German umlauts (vowel mutations) from text. 
    ä or ae -> a...
    ü or ue -> ue 
    """
    vowel_char_map = {ord('ä'):'a', ord('ü'):'u', ord('ö'):'o', ord('ß'):'ss'}
    text = text.translate(vowel_char_map)
    text = text.replace('ae','a').replace('ue', 'u').replace('oe','o')
    return text

def parse_text(text):
    
    RE_WSPACE = re.compile(r"\s+", re.IGNORECASE) #remove whitespace
    RE_TAGS = re.compile(r"<[^>]+>") #remove HTML tags
    RE_ASCII = re.compile(r"[^A-Za-zÀ-ž ]", re.IGNORECASE) # remove ASCII
    RE_SINGLECHAR = re.compile(r"\b[A-Za-zÀ-ž]\b", re.IGNORECASE) # remove single characters

    text = re.sub(RE_TAGS, " ", text)
    text = re.sub(RE_ASCII, " ", text)
    text = re.sub(RE_SINGLECHAR, " ", text)
    text = re.sub(RE_WSPACE, " ", text)
    
    text = text.lower().split()
    swords = set(stopwords.words("german"))
    
    stemmer = SnowballStemmer("german")
    text = [stemmer.stem(w) for w in text if w not in swords]

    text = [replace_umlauts(w) for w in text if w not in swords]

    text = " ".join(text)
    
    return text


rootdir = 'data'
data = []
for subdir, dirs, files in os.walk(rootdir):    
    for file in files:
        f = os.path.join(subdir, file)
        with open(f, encoding='utf-8') as f:
            if not file.startswith('.'):
                    x = parse_text(f.read())
                    data.append((subdir.split('\\')[-1],x))


df = pd.DataFrame(data,columns=['label', 'text'])

from sklearn.utils import shuffle

df = shuffle(df)


df.label = df.label.replace(to_replace=['ni', 'i'], value=[0, 1])

def make_numeric(df):
    df["label"] = pd.factorize(df["label"])[0]
    return df
df = make_numeric(df)
df.head()

df_copy = df.copy()
new_df_list = []
# Iterate over each row in the original dataframe

for i in df_copy.index:
    # Get the text, label from the row
    text = df_copy.loc[i]['text']
    label = df_copy.loc[i]['label']
    # Split the text into tokens
    tokens = text.split(' ')
    if len(tokens) >= 200:
        num_chunks = len(tokens) // 200
        # Iterate over each chunk
        for i in range(num_chunks):
            # Get the start and end index of each chunk
            start_index = i * 200
            end_index = (i + 1) * 200
            # Get the chunk of text
            chunk = tokens[start_index:end_index]
            # Create a new row in the list with the label and the chunk of text
            new_row = [label, ' '.join(chunk)]
            # Append the new row to the list
            new_df_list.append(new_row)
            # If the length of tokens is less than 200, then just add the row to the list
    else:
        new_df_list.append([label, text])
        
new_df = pd.DataFrame(new_df_list, columns=['label', 'text'])


new_df = shuffle(new_df)
new_df = new_df.dropna()

labels = new_df['label'].value_counts()
print(labels)
#calculate the number of rows with label 1 that need to be removed
if labels[1] > labels[0]:
    majority_label = 1
    num_rows_to_remove = labels[1] - labels[0]
else:
    majority_label = 0
    num_rows_to_remove = labels[0] - labels[1]

#randomly select the rows with label 1 to remove
import random
rows_to_remove = random.sample(list(new_df.loc[new_df['label'] == majority_label].index), num_rows_to_remove)
new_df = new_df[new_df['text'].str.split().str.len() >= 71]
#remove the rows and update the dataframe
balanced_df = new_df.copy()
balanced_df.drop(rows_to_remove, inplace=True)

from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(balanced_df,test_size = 0.2,random_state=7)

# With Val Data
train_data, val_data = train_test_split(train_data,test_size = 0.2,random_state=7)

print(len(train_data))
print(len(test_data))
print(len(val_data))


# Create Word2Vec
sentences = pd.concat([train_data['text'], test_data['text']],axis=0)
tqdm.pandas()
train_sentences = list(sentences.progress_apply(str.split).values)


embed_vec = Word2Vec(sentences=train_sentences, 
                 sg=0, 
                 vector_size=100,  
                 workers=4,
                    epochs=4)


filePath = 'custom_embed_100d.txt'

if os.path.exists(filePath):
    os.remove(filePath)

embed_vec.wv.save_word2vec_format('custom_embed_100d.txt')


tokenizer = Tokenizer() # initialize the Tokenizer object
tokenizer.fit_on_texts(train_data.text) #each word is assigned a unique number & every word is now represented by a number
vocab_size = len(tokenizer.word_index) + 1 # create vocab index

train_sequences = tokenizer.texts_to_sequences(train_data.text) # convert each sentence into a sequence of numbers 
MAX_LENGTH = max(len(sentence) for sentence in train_sequences) 

x_train = pad_sequences(tokenizer.texts_to_sequences(train_data.text),
                        maxlen = MAX_LENGTH)
x_test = pad_sequences(tokenizer.texts_to_sequences(test_data.text),
                       maxlen = MAX_LENGTH)
x_val = pad_sequences(tokenizer.texts_to_sequences(val_data.text),
                       maxlen = MAX_LENGTH)



from sklearn.preprocessing import LabelEncoder
labels = train_data.label.unique().tolist()

encoder = LabelEncoder()
encoder.fit(train_data.label.to_list())

y_train = encoder.transform(train_data.label.to_list())
y_test = encoder.transform(test_data.label.to_list())
y_val = encoder.transform(val_data.label.to_list())

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_val = y_val.reshape(-1,1)


print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print("y_val shape:", y_val.shape)


EMBEDDING_DIM = 100
LR = 1e-3
embeddings_dict = {}
with open("custom_embed_100d.txt", 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split() #split lines at white space
        word = values[0] # word equals first element (0th)
        vector = np.asarray(values[1:], "float32") # rest of line convert to numpy arr = vector of word position
        embeddings_dict[word] = vector # update dict with word + vector
        
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
for word,i in tokenizer.word_index.items():
    embedding_vect = embeddings_dict.get(word)
    if embedding_vect is not None:
        embedding_matrix[i] = embedding_vect


embedding_layer = tf.keras.layers.Embedding(vocab_size,
                                          EMBEDDING_DIM,
                                          weights=[embedding_matrix],
                                          input_length=MAX_LENGTH,
                                          trainable=False)




from keras.layers import Conv1D, Bidirectional, LSTM, Dense, Input, Dropout
from keras.layers import SpatialDropout1D

sequence_input = Input(shape=(MAX_LENGTH,), dtype='int32')
embedding_sequences = embedding_layer(sequence_input)
x = SpatialDropout1D(0.2)(embedding_sequences)
x = Conv1D(64, 5, activation='relu')(x)
x = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(sequence_input, outputs)
model.summary()


from tensorflow.keras.optimizers import Adam

model.compile(optimizer=Adam(learning_rate=LR), loss='binary_crossentropy',  metrics=['accuracy'])



earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)


history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
          batch_size = 100,
            epochs = 10,
            verbose = 1 , callbacks=[earlystop])


results = model.evaluate(x_test, y_test, batch_size=100)
#history.history
print("test loss, test acc:", results)


from sklearn.metrics import classification_report
def decode_sentiment(score):
    return 1 if score>0.5 else 0


scores = model.predict(x_test, verbose=1, batch_size=100)
y_pred = np.array([decode_sentiment(score) for score in scores])


print(classification_report(y_test, y_pred))
print("precision: "+ str(precision_score(y_test, y_pred)))
print("recall: "+ str(recall_score(y_test, y_pred)))
print("f1: "+ str(f1_score(y_test, y_pred)))


#!/usr/bin/env python
# coding: utf-8

import re
import pandas as pd
import os

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score

from nltk.corpus import stopwords

# preprocess data

import nltk
nltk.download('stopwords')
# mit Stemmer: 
from nltk.stem.snowball import SnowballStemmer 

def replace_umlauts(text:str) -> str:
    """replace special German umlauts (vowel mutations) from text. 
    ä -> ae...
    ü -> ue 
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
df.head()


df.label = df.label.replace(to_replace=['ni', 'i'], value=[0, 1])

def make_numeric(df):
    df["label"] = pd.factorize(df["label"])[0]
    return df
df = make_numeric(df)
df.head()

# # Split Text into smaller chunks of max. 200 token length

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

labels = new_df['label'].value_counts()

# calculate majoritylabel/the number of rows that need to be removed
if labels[1] > labels[0]:
    majoritylabel = 1
    num_rows_to_remove = labels[1] - labels[0]
elif labels[0] > labels[1]:
    majoritylabel = 0
    num_rows_to_remove = labels[0] - labels[1]

# randomly select the rows with label 1 to remove
import random
rows_to_remove = random.sample(list(new_df.loc[new_df['label'] == majoritylabel].index), num_rows_to_remove)


# remove the rows and update the dataframe
balanced_df = new_df.copy()
balanced_df.drop(rows_to_remove, inplace=True)

balanced_df = shuffle(balanced_df)


from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(balanced_df,test_size = 0.2,random_state=7)
# With Val Data
train_data, val_data = train_test_split(train_data,test_size = 0.2,random_state=7)


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
x_train = cv.fit_transform(train_data.text.values)
x_test = cv.transform(test_data.text.values)
x_val = cv.transform(val_data.text.values)


from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB().fit(x_train, train_data.label.values)
# making predictions on the validation set
val_pred = clf.predict(x_val)

# making predictions on the test set
print("Accuracy: "+str(clf.score(x_val, val_data.label.values)))
y_pred = clf.predict(x_test)
# evaluate the performance 
print("precision: "+ str(precision_score(test_data.label.values, y_pred)))
print("recall: "+ str(recall_score(test_data.label.values, y_pred)))
print("f1: "+ str(f1_score(test_data.label.values, y_pred)))
print("accuracy: "+ str(accuracy_score(test_data.label.values, y_pred, normalize=False)))
print(classification_report(test_data.label.values, y_pred))

#!/usr/bin/env python
# coding: utf-8

# # Lab 11: Dialogue Act Tagging
# 
# Dialogue act (DA) tagging is an important step in the process of developing dialog systems. DA tagging is a problem usually solved by supervised machine learning approaches that all require large amounts of hand labeled data. A wide range of techniques have been investigated for DA tagging. In this lab, we explore two approaches to DA classification. We are using the Switchboard Dialog Act Corpus for training.
# Corpus can be downloaded from http://compprag.christopherpotts.net/swda.html.
# 

# The downloaded dataset should be kept in a data folder in the same directory as this file. 

# In[1]:


import pandas as pd
import glob
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np

import sklearn.metrics
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm


# In[2]:


# f = glob.glob("swda/sw*/sw*.csv")
# frames = []
# for i in range(0, len(f)):
#     frames.append(pd.read_csv(f[i]))

# result = pd.concat(frames, ignore_index=True)
# result.to_csv('swda_all_data.csv',index=False)


# In[4]:


result = pd.read_csv('swda_all_data.csv')


# In[5]:


print("Number of converations in the dataset:",len(result))


# The dataset has many different features, we are only using act_tag and text for this training.
# 

# In[6]:


reduced_df = result[['act_tag','text']]


# Reduce down the number of tags to 43 - converting the combined tags to their generic classes:

# In[7]:


# Imported from "https://github.com/cgpotts/swda"
# Convert the combination tags to the generic 43 tags

import re
def damsl_act_tag(input):
        """
        Seeks to duplicate the tag simplification described at the
        Coders' Manual: http://www.stanford.edu/~jurafsky/ws97/manual.august1.html
        """
        d_tags = []
        tags = re.split(r"\s*[,;]\s*", input)
        for tag in tags:
            if tag in ('qy^d', 'qw^d', 'b^m'): pass
            elif tag == 'nn^e': tag = 'ng'
            elif tag == 'ny^e': tag = 'na'
            else: 
                tag = re.sub(r'(.)\^.*', r'\1', tag)
                tag = re.sub(r'[\(\)@*]', '', tag)            
                if tag in ('qr', 'qy'):                         tag = 'qy'
                elif tag in ('fe', 'ba'):                       tag = 'ba'
                elif tag in ('oo', 'co', 'cc'):                 tag = 'oo_co_cc'
                elif tag in ('fx', 'sv'):                       tag = 'sv'
                elif tag in ('aap', 'am'):                      tag = 'aap_am'
                elif tag in ('arp', 'nd'):                      tag = 'arp_nd'
                elif tag in ('fo', 'o', 'fw', '"', 'by', 'bc'): tag = 'fo_o_fw_"_by_bc'            
            d_tags.append(tag)
        # Dan J says (p.c.) that it makes sense to take the first;
        # there are only a handful of examples with 2 tags here.
        return d_tags[0]


# In[8]:


reduced_df["act_tag"] = reduced_df["act_tag"].apply(lambda x: damsl_act_tag(x))


# There are 43 tags in this dataset. Some of the tags are Yes-No-Question('qy'), Statement-non-opinion('sd') and Statement-opinion('sv'). Tags information can be found here http://compprag.christopherpotts.net/swda.html#tags. 
# 

# To get unique tags:

# In[9]:


unique_tags = set()
for tag in reduced_df['act_tag']:
    unique_tags.add(tag)


# In[10]:


one_hot_encoding_dic = pd.get_dummies(list(unique_tags))


# In[11]:


tags_encoding = []
for i in range(0, len(reduced_df)):
    tags_encoding.append(one_hot_encoding_dic[reduced_df['act_tag'].iloc[i]])


# The tags are one hot encoded.

# To create sentence embeddings:

# In[12]:


sentences = []
for i in range(0, len(reduced_df)):
    sentences.append(reduced_df['text'].iloc[i].split(" "))


# In[13]:


wordvectors = {}
index = 1
for s in sentences:
    for w in s:
        if w not in wordvectors:
            wordvectors[w] = index
            index += 1


# In[14]:


# Max length of 137
MAX_LENGTH = len(max(sentences, key=len))


# In[15]:


sentence_embeddings = []
for s in sentences:
    sentence_emb = []
    for w in s:
        sentence_emb.append(wordvectors[w])
    sentence_embeddings.append(sentence_emb)


# Then we split the dataset into test and train.

# In[16]:


from sklearn.model_selection import train_test_split
import numpy as np
X_train, X_test, y_train, y_test = train_test_split(sentence_embeddings, np.array(tags_encoding))


# And pad the sentences with zero to make all sentences of equal length.
# 

# In[17]:


MAX_LENGTH = 137


# In[18]:


from keras.preprocessing.sequence import pad_sequences
 
train_sentences_X = pad_sequences(X_train, maxlen=MAX_LENGTH, padding='post')
test_sentences_X = pad_sequences(X_test, maxlen=MAX_LENGTH, padding='post')


# Split Train into Train and Validation - about 10% into validation - In order to validate the model as it is training

# In[19]:




train_input = train_sentences_X[:140000]
val_input = train_sentences_X[140000:]

train_labels = y_train[:140000]
val_labels = y_train[140000:]


# # Model 1 - 
# 
# The first approach we'll try is to treat DA tagging as a standard multi-class text classification task, in the way you've done before with sentiment analysis and other tasks. Each utterance will be treated independently as a text to be classified with its DA tag label. This model has an architecture of:
# 
# - Embedding  
# - BLSTM  
# - Fully Connected Layer
# - Softmax Activation

#  The model architecture is as follows: Embedding Layer (to generate word embeddings) Next layer Bidirectional LSTM. Feed forward layer with number of neurons = number of tags. Softmax activation to get the probabilities.
# 

# In[20]:


VOCAB_SIZE = len(wordvectors) # 43,731
MAX_LENGTH = len(max(sentences, key=len))
EMBED_SIZE = 100 # arbitary
HIDDEN_SIZE = len(unique_tags) 


# In[21]:


from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout, InputLayer, Bidirectional, TimeDistributed, Activation, Embedding
from keras.optimizers import Adam

#Building the network

# Include 2 BLSTM layers, in order to capture both the forward and backward hidden states

# Embedding layer
# Bidirectional 1
# Bidirectional 2
# Dense layer
# Activation
##
model = Sequential()
model.add(Embedding(VOCAB_SIZE,EMBED_SIZE,input_length=MAX_LENGTH))
model.add(Bidirectional(LSTM(HIDDEN_SIZE,return_sequences=True)))
model.add(Bidirectional(LSTM(HIDDEN_SIZE)))
model.add(Dense(HIDDEN_SIZE,activation='relu'))
model.add(Activation('sigmoid'))
##
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()


# In[ ]:


from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout, InputLayer, Bidirectional, TimeDistributed, Activation, Embedding
from keras.optimizers import Adam

#Building the network

# Include 2 BLSTM layers, in order to capture both the forward and backward hidden states

# Embedding layer
# Bidirectional 1
# Bidirectional 2
# Dense layer
# Activation
##

##
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()


# In[23]:


# Train the model - using validation 
history = model.fit(train_input,
                    train_labels,
                    epochs=3,
                    batch_size=32,
                    validation_data=(val_input, val_labels),
                    verbose=1)


# In[ ]:


score = model.evaluate(test_sentences_X, y_test, batch_size=100)


# In[ ]:


print("Overall Accuracy:", score[1]*100)


# ## Evaluation
# 
# 
# The overall accuracy is 67%, an effective accuracy for this task.

# In addition to overall accuracy, you need to look at the accuracy of some minority classes. Signal-non-understanding ('br') is a good indicator of "other-repair" or cases in which the other conversational participant attempts to repair the speaker's error. Summarize/reformulate ('bf') has been used in dialogue summarization. Report the accuracy for these classes and some frequent errors you notice the system makes in predicting them. What do you think the reasons are？

# ## Minority Classes

# In[ ]:


# Generate predictions for the test data


# In[ ]:


# Build the confusion matrix off these predictions


# In[ ]:


# Calculate Accuracies for "br" and "bf"


# 
# Due to the reduced lack of training data for the minority classes, these minority classifiers will not be very confident in classification, as they have not been fully optimised. The frequent classifiers will be more optimised and will generate more confident scores for all examples, effectively crowding out the less confident minority classifiers. 
# 
# 
# 


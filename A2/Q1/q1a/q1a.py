#!/usr/bin/env python
# coding: utf-8

# # Q1. Text Classification

# (a) MN Naive Bayes with Smoothing

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


from wordcloud import WordCloud
from collections import defaultdict
import re


# (a) Naive Bayes and Wordcloud

# In[4]:


train_data = pd.read_csv('../ML-A2/Corona_train.csv')
cv_data = pd.read_csv('../ML-A2/Corona_validation.csv')


# In[5]:


print(train_data)


# In[6]:


train_data.columns


# In[7]:


def ethnicCleansing(tweet):
    tweet = tweet.lower()

    link_pattern = re.compile(r"https?://\S+")
    tweet = link_pattern.sub("", tweet)

    punctn_pattern = re.compile(r"[^A-Za-z ]")
    tweet = punctn_pattern.sub("", tweet)

    tw_words = tweet.split()
    return tw_words


# In[8]:


def sentimentClassLabel(sent):
    if sent == "Positive":
        return 1
    elif sent == "Neutral":
        return 0
    elif sent == "Negative":
        return -1
    else:
        return None

train_data['SentimentNB'] = train_data['Sentiment'].map(lambda sent: sentimentClassLabel(sent))


# In[9]:


train_data['CoronaTweetNB'] = train_data['CoronaTweet'].map(lambda tweet: ethnicCleansing(tweet))


# In[10]:


# now on test data
cv_data['SentimentNB'] = cv_data['Sentiment'].map(lambda sent: sentimentClassLabel(sent))
cv_data['CoronaTweetNB'] = cv_data['CoronaTweet'].map(lambda tweet: ethnicCleansing(tweet))


# In[11]:


# train_data['SentimentNB'].value_counts()
# train_data['Sentiment'].value_counts()
train_data


# i. Naive Bayes

# In[12]:


m_train = len(train_data)
m_cv = len(cv_data)


# In[13]:


priors = train_data['SentimentNB'].value_counts().to_dict()

# priors = {sent: c/m for sent,c in priors.items()}
priors = {sent: np.log(c/m_train) for sent,c in priors.items()}   #take log??
priors


# In[14]:


# initialize 0 counts for 3 classes by default
word_condn_counts = defaultdict(lambda: {1:0, -1:0, 0:0})
word_net_counts = defaultdict(lambda: 0)
vocab = set()


# In[15]:


for row in train_data[['SentimentNB','CoronaTweetNB']].itertuples(index=False):
    i = row[0]
    tw_words = row[1]
    # print(i)
    # print(tw_words)
    for word in tw_words:
        word_condn_counts[word][i] += 1
        word_net_counts[word] += 1
        vocab.add(word)


# In[16]:


print(word_condn_counts["kinds"])
print(word_net_counts)


# In[17]:


condn_prob = defaultdict(dict)

alpha = 1
v_size = len(vocab)

for i in range(-1,2):
    total_smoothed = sum((word_condn_counts[w][i] + alpha) for w in vocab)
    for word in vocab:
        count_smoothed = word_condn_counts[word][i] + alpha
        condn_prob[word][i] = np.log(count_smoothed/total_smoothed)     #save log(val)

# takes tooo long
#condn_prob


# In[18]:


def predict(tw_words):
    pred_class = None
    max_ll = float('-inf')

    for i in range(-1,2):
        ll = priors[i]
        for word in tw_words:
            if word in vocab:
                ll += condn_prob[word][i]
        if ll > max_ll:
            pred_class = i
            max_ll = ll

    return pred_class


# In[19]:


train_predict = train_data['CoronaTweetNB'].map(lambda tw_words: predict(tw_words))
nb_train = np.sum(train_predict == train_data['SentimentNB'])

print("Accuracy on training data = ", end='')
print(nb_train/m_train)


# In[20]:


cv_predict = cv_data['CoronaTweetNB'].map(lambda tw_words: predict(tw_words))
nb_cv = np.sum(cv_predict == cv_data['SentimentNB'])

print("Accuracy on validation data = ", end='')
print(nb_cv/m_cv)


# In[31]:


# a = {x: 2*x for x in range(5)}
# a


# ii. Wordclouds for each class

# In[70]:


pos_words_str = {w: word_condn_counts[w][1] for w in vocab}
word_cloud_pos = WordCloud(max_font_size=96, background_color = 'white', width = 800, height = 480, colormap="Reds").generate_from_frequencies(pos_words_str)

# def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
#     return "hsl(0, 0%%, %d%%)" % np.random.randint(60, 100)

# word_cloud_pos.recolor(color_func=grey_color_func, random_state=1000)

plt.title("Positive words", pad=20)
plt.imshow(word_cloud_pos, interpolation="bilinear")
plt.axis("off")

plt.savefig("./q1a/q1a_positive_words.png")


# haha toiletpaper

# In[71]:


neg_words_str = {w: word_condn_counts[w][-1] for w in vocab}
word_cloud_neg = WordCloud(max_font_size=96, background_color = 'white', width = 800, height = 480, colormap='Greys').generate_from_frequencies(neg_words_str)


plt.title("Negative words", pad=20)
plt.imshow(word_cloud_neg, interpolation="bilinear")
plt.axis("off")

plt.savefig("./q1a/q1a_negative_words.png")


# In[72]:


ntr_words_str = {w: word_condn_counts[w][0] for w in vocab}
word_cloud_ntr = WordCloud(max_font_size=96, background_color = 'white', width = 800, height = 480, colormap='Blues').generate_from_frequencies(ntr_words_str)

plt.title("Neutral words", pad=20)
plt.imshow(word_cloud_ntr, interpolation="bilinear")
plt.axis("off")

plt.savefig("./q1a/q1a_neutral_words.png")
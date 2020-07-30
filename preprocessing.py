#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 21:51:39 2020

@author: Zijing Wu (Miles)
"""

import pandas as pd
import numpy as np

import re
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk import wordnet

# =============================================================================
# Gather and preprocess the dataset
# =============================================================================

# NUM_TWEET = 800000

# #read training data from sentiment140 dataset http://help.sentiment140.com/for-students
col_names = ['Labels', 'ID', 'Time', 'Query', 'User_ID','Tweets']
tweet_df = pd.read_csv('data/trainingandtestdata/training.1.6m.csv', encoding="latin1", names=col_names, header=None, usecols=['Labels', 'Tweets'])

# # For processing data of the geo-tagged COVID-19 tweets

# NUM_TWEET = 10000
# start = 1
# num_day = 29

# tweet = [pd.read_csv('data/hydrated/ready_june{}_june{}.csv'.format(start, start + 1)) for start in np.arange(start, start + num_day, 1)]
# id_score = [pd.read_csv("data/original/june{}_june{}.csv".format(start, start + 1), header=None).rename(columns={0:'id', 1:'score'}) for start in np.arange(start, start + num_day, 1)]
# tweet = pd.concat(tweet, ignore_index=True)
# id_score = pd.concat(id_score, ignore_index=True)
# tweet_df = pd.merge(tweet, id_score, how='inner', on='id').loc[:, ['coordinates', 'id', 'text', 'score']].dropna()

# print("twe et dataframe has a shape of {}".format(tweet.shape))
# print()
# print("id_score dataframe has a shape of {}".format(id_score.shape))
# print()
# print("tweet_pd dataframe has a shape of {}".format(tweet_df.shape))

NUM_TWEET = 10000

def balanced_sample(tweet_positive, tweet_negative):
    #Create balanced tweetset
    sample_size = min(tweet_positive.shape[0], tweet_negative.shape[0])
    raw_tweet = np.concatenate((tweet_positive['Tweets'].values[:sample_size], 
                                tweet_negative['Tweets'].values[:sample_size]), axis=0) 
    labels = [1]*sample_size + [0]*sample_size
    print("sample size is {}".format(sample_size * 2))
    print()
    print("raw_tweet has a shape of {}".format(len(raw_tweet)))
    print()
    return raw_tweet, labels

def sample(tweet_positive, tweet_negative):
    #Not intentionally create a balanced tweetset
    raw_tweet = np.concatenate((tweet_positive['Tweets'].values[0:], 
                                tweet_negative['Tweets'].values[0:]), axis=0)
    labels = [1]*tweet_positive['Tweets'].shape[0] + [0]*tweet_negative['Tweets'].shape[0]
    
    print("sample size is {}\n".format(tweet_positive['Tweets'].shape[0] + tweet_negative['Tweets'].shape[0]))
    print("raw_tweet has a shape of {}\n".format(len(raw_tweet)))
    return raw_tweet, labels



cList = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "I'd": "I would",
  "I'd've": "I would have",
  "I'll": "I will",
  "I'll've": "I will have",
  "I'm": "I am",
  "I've": "I have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have",
  "hadn't've": "had not have",
  "hadn't": "had not"
}

c_re = re.compile('(%s)' % '|'.join(cList.keys()))    

def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text.lower())

def word_process(word):
    lemma =  wordnet.WordNetLemmatizer()
    word = word.replace("'", "")
    word = word.replace("&amp", "")
    word = word.strip(punctuation)
    return lemma.lemmatize(word)


def _process(tweet):
    stop_words = set( list(punctuation) + ['...', '..'] + ['AT_USER','URL', 'at_user', 'url'])
    tweet = tweet.lower() # convert text to lower-case
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet) # remove URLs
    tweet = re.sub('@[^\s]+', 'AT_USER', tweet) # remove usernames
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag
    tweet = re.sub('\w*\d\w*', '', tweet) # remove words containing digits
    tweet = expandContractions(tweet) # expand contractions
    tweet = word_tokenize(tweet) #create a list of words
    tweet = [word_process(word.strip()) for word in tweet if word not in stop_words] #remove stopwords
    return ' '.join(tweet) #return the tweet string


def get_encode_error(data):
    count = 0
    uncoded = []
    for t in data:
        if 'ï¿½' in t:
            uncoded.append(t)
            count += 1
    print('There are {} sentences contain encoding error.'.format(count))
    return count, uncoded

#Save data
def save(data, name, r, file_path = "data/preprocessed/"):
    """
    save preprocessed data for later use
    data: list, data to save
    name: string, type of data
    r: int, day range the data covered; or string ''
    """
    from pickle import dump
    with open('_'.join([file_path, name, str(r)])+ '.pkl','wb') as file:
        dump(data, file)
    print('_'.join([file_path, name, str(r)])+ '.pkl' + " saved")

def load_data(name, r, file_path = 'data/preprocessed/'):
    """
    load saved preprocessed data
    name: string, type (tweet or labels)
    r: int, day range the data covered; or string ''
    """
    from pickle import load
    with open('_'.join([file_path, name, str(r)])+ '.pkl','rb') as file:
        data = load(file)
    return data



def get_test_data():
    test_df = pd.read_csv('data/trainingandtestdata/testdata.manual.csv', encoding="latin1", names=col_names, header=None, usecols=['Labels', 'Tweets'])
    test_df = test_df[~test_df.Tweets.str.contains('ï¿½')]
    test_df = test_df[test_df.Labels != 2]
    tweet_positive = test_df[test_df['Labels'] == 4]
    tweet_negative = test_df[test_df['Labels'] == 0]
    test_tweet, labels = sample(tweet_positive, tweet_negative)
    test_data = [_process(t) for t in test_tweet]
    get_encode_error(data)
    return test_data, labels

tweet_df = tweet_df[~tweet_df.Tweets.str.contains('ï¿½')]


tweet_positive = tweet_df[tweet_df['Labels'] == 4].iloc[:NUM_TWEET]
tweet_negative = tweet_df[tweet_df['Labels'] == 0].iloc[:NUM_TWEET]

print("tweet_df dataframe has a shape of {}\n".format(tweet_df.shape))
print("tweet_positive dataframe has a shape of {}\n".format(tweet_positive.shape))
print("tweet_negative dataframe has a shape of {}\n".format(tweet_negative.shape))


raw_tweet, labels = sample(tweet_positive, tweet_negative)



data = [_process(t) for t in raw_tweet]
get_encode_error(data)

save(data, 'tweet', NUM_TWEET)
save(labels, 'labels', NUM_TWEET)

# # For processing test data
# # save test data
# test, labels = get_test_data()
# save(test, 'test_data', '')
# save(labels, 'test_labels', '')

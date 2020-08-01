import pandas as pd
import re, string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score

# url = './data/final/final_june1_june2.csv'
NUM_TWEET = 50000

# def load_final_file(url):
#     df = pd.read_csv(url, index_col='id')
#     # print(df.columns.values)
#     return df


# def clean_tweet(tweet):
#     tweet = tweet.lower() # convert text to lower-case
#     tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet) # remove URLs
#     tweet = re.sub('@[^\s]+', 'AT_USER', tweet) # remove usernames
#     tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag
#     tweet = re.sub('\[.*?\]', '', tweet)  # remove puntuation
#     tweet = re.sub('[%s]' % re.escape(string.punctuation), '', tweet)
#     tweet = re.sub('\w*\d\w*', '', tweet) # removes words containing digits
#     tweet = re.sub('[‘’“”…]', '', tweet)
#     tweet = re.sub('\n', '', tweet)

#     emoji_pattern = re.compile("["
#                       u"\U0001F600-\U0001F64F"  # emoticons
#                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
#                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
#                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
#                       u"\U00002500-\U00002BEF"  # chinese char
#                       u"\U00002702-\U000027B0"
#                       u"\U00002702-\U000027B0"
#                       u"\U000024C2-\U0001F251"
#                       u"\U0001f926-\U0001f937"
#                       u"\U00010000-\U0010ffff"
#                       u"\u2640-\u2642"
#                       u"\u2600-\u2B55"
#                       u"\u200d"
#                       u"\u23cf"
#                       u"\u23e9"
#                       u"\u231a"
#                       u"\ufe0f"  # dingbats
#                       u"\u3030"
#                       "]+", re.UNICODE)
#     tweet = emoji_pattern.sub(r'', tweet)

#     tweet = re.sub(r'[^\x00-\x7f]', r'', tweet)
#     # tweet = word_tokenize(tweet)
#     return tweet #return the tweet string


def to_term_matrix(df):
    cv = CountVectorizer(stop_words='english')
    data_cv = cv.fit_transform(df.text)
    data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
    data_dtm.index = df.index
    return data_dtm

def word_cloud_generator(data):
    top_dict = {}
    for c in data.columns:
        top = data[c].sort_values(ascending=False).head(30)
        top_dict[c] = list(zip(top.index, top.values))

    print(top_dict)


def load_data(name, r, file_path = 'data/preprocessed/'):
    """
    load saved preprocessed data
    name: string, type (tweet or labels)
    r: int, day range the data covered
    """
    from pickle import load
    with open('_'.join([file_path, name, str(r)])+ '.pkl','rb') as file:
        data = load(file)
    return data


data = load_data('tweet', NUM_TWEET)
labels = load_data('labels', NUM_TWEET)
print('data loaded.')


# split into train set and test set
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(data, labels, test_size=0.3, random_state=42)
print()
print("data splitted...")
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(data)
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)
# print(Tfidf_vect.vocabulary_)

# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
print()
print("fitting SVM...")
print()

import time
start_time = time.time()
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', verbose=True)
SVM.fit(Train_X_Tfidf,Train_Y)
print("--- SVM spends {}s seconds on training. ---".format(time.time() - start_time))
print()
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Test_Y)*100)

#-----------
from sklearn.metrics import classification_report

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
print("Sanity check. The report for classifying the training set")
print(classification_report(Train_Y, SVM.predict(Train_X_Tfidf), digits=4))
print()
print("The report for classifying the test set")

print(classification_report(Test_Y, SVM.predict(Test_X_Tfidf), digits=4))
print()    

#prediction of some random sentences
# print("classification probability of \"I am happy\" is {}".format(SVM.predict(Tfidf_vect.transform(["I am happy"]))))
# print()
# print("classification probability of \"I am so happy\" is {}".format(SVM.predict(Tfidf_vect.transform(["I am so happy"]))))
# print()
# print("classification probability of \"I have never been so happy before.\" is {}".format(SVM.predict(Tfidf_vect.transform(["I have never been so happy before."]))))
# print()
# print("classification probability of \"I am not happy\" is {}".format(SVM.predict(Tfidf_vect.transform(["I am not happy"]))))
# print()
# print("classification probability of a neutral string: {}".format(SVM.predict(Tfidf_vect.transform(["Call decision_function on the estimator with the best found parameters."]))))
# print()
# print("classification probability of a long neutral string: {}".format(SVM.predict(Tfidf_vect.transform(["Exhaustive search over specified parameter values for an estimator. Important members are fit, \
#                    predict. GridSearchCV implements a “fit” and a “score” method. It also implements “predict”, “predict_proba”, \
#                    “decision_function”, “transform” and “inverse_transform” if they are implemented in the estimator used. \
#                    The parameters of the estimator used to apply these methods are optimized by cross-validated grid-search \
#                    over a parameter grid."]))))
# print()
# print("classification probability of a tweet: {}".format(SVM.predict(Tfidf_vect.transform(["Time for a Royal Celebration! #Royalbaby pic.twitter.com/lITsX3lHfQ"]))))
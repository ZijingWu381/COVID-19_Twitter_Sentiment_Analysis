
"""
Created on Sat Jun 27 09:34:49 2020

@author: Zijing Wu (Miles)
"""

import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (classification_report,
							 confusion_matrix)
# =============================================================================
# Load data
# =============================================================================


NUM_TWEET = 100000

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
x_test = load_data('test_data', '')
y_test = load_data('test_labels', '')

print('data loaded.')

# =============================================================================
# Find the baseline results
# =============================================================================
def baseline(X, y):
    '''Find the baseline results of the data
    X: the observations
    y: the results
    '''
    strategies = {'stratified': 0, 'most_frequent': 0, 'prior': 0, 'uniform': 0}
    for s in strategies.keys():
        dummy_clf = DummyClassifier(strategy=s) 
        dummy_clf.fit(X, y)
        strategies[s] = dummy_clf.score(X, y)
    best = max(strategies, key=strategies.get) # the performance of the best dummy classifier
    return strategies, best, strategies[best]



# =============================================================================
# Training MultinomialNB by finding the best parameters
# =============================================================================
# The following is adapted from Sergey Smetanin's code
# https://github.com/sismetanin/sentiment-analysis-of-tweets-in-russian/blob/master/Sentiment%20Analysis%20of%20Tweets%20in%20Russian%20using%20Multinomial%20Naive%20Bayes.ipynb

text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB())])

# # Tuning Pipeline
# tuned_parameters = {
#     'vect__ngram_range': [(1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 3)],
#     'tfidf__use_idf': [False, True],
#     'tfidf__norm': ('l1', 'l2'),
#     'clf__alpha': [1, 1e-1, 1e-2],
# }



tuned_parameters = {
    'vect__ngram_range': [(1, 3)],
    'tfidf__use_idf': [False],
    'tfidf__norm': ('l2'), 
    'clf__alpha': [1],
}

x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.05, random_state=42)

def train():
    #The dataset was plited into train and test subsets.
    
    print("Our training set size is {}".format(len(x_train)))
    print()
    print("Our testing set size is {}".format(len(x_val)))
    
    
    
    #Tune parameters
    score = 'accuracy'
    print("# Tuning hyper-parameters for %s" % score)
    print()
    np.errstate(divide='ignore')
    clf = GridSearchCV(text_clf, tuned_parameters, cv=10, scoring=score);
    clf.fit(x_train, y_train)
    
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    for mean, std, params in zip(clf.cv_results_['mean_test_score'], 
                                  clf.cv_results_['std_test_score'], 
                                  clf.cv_results_['params']):
        if params == clf.best_params_:
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    return clf


clf = train()

def test(clf, x_train, y_train, x_val, y_val, x_test, y_test):

    # try some predictions here [negative, positive] Here are some random things I predict with the clf.
    
    print("Detailed classification report:")
    print()
    print("Sanity check. The report for classifying the training set")
    print(classification_report(y_train, clf.predict(x_train), digits=4))
    print()
    print("The report for classifying the validation set")
    print(classification_report(y_val, clf.predict(x_val), digits=4))
    print("The report for classifying the test set")
    print(classification_report(y_test, clf.predict(x_test), digits=4))

from sklearn.metrics import plot_confusion_matrix


class_names = ['negative', 'positive']

def plot_cm(clf, labels, x, data='test'):
    disp = plot_confusion_matrix(clf, x, labels,
                                  display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
    plt.title("Confusion matrix ({})".format(data))
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    

    plt.show()

plot_cm(clf, y_test, x_test, data='test')
plot_cm(clf, y_val, x_val, data='validation')


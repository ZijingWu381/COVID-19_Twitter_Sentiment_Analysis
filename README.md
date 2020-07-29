# COVID-19_Twitter_Sentiment_Analysis
#### Akash Shah, Chiche Tsai, Jiaxi Xu, Zijing Wu


![Twitter Sentiment Analysis](https://github.com/miles-zijingwu/COVID-19_Twitter_Sentiment_Analysis/blob/master/Image/heading_image.jpg)


## Introduction and Motivation

Since its discovery, the COVID-19 pandemic has quickly become a worldwide crisis that is threatening public health. By the time of writing this report, there have been over 4.3 millions infections and 149 thousands deaths cumulatively in the United States and over 16.2 millions cases and 648 thousands deaths worldwide [1]. Polarized emotion arises as people facing this unprecedented event and could lead to social unrest. When trying to mitigate such public health risk and reopen business, policy makers need to know the real-time and accurate assessment of the public sentiment so that they can make the most effective regulations accordingly. 

### Why Twitter?
However, the traditional approaches like census or sampling are either slow or not comprehensive. Instead, we aim to provide a faster and more comprehensive assessment of public sentiment. This could be done by performing sentiment analysis on the recent tweet data using machine learning approaches. Nowadays, people frequently share personal experience which may contain a certain level of  sentiment on social media. Especially for Twitter, the US accounts for 31 millions active daily users becoming a decent resource  to collect the public’s sentiment [2]. Given the large number of active users, mining tweets has the potential to give fast and real estimation of people’s sentiment towards the pandemic. 

### Project Overview
In our work, we acquired two datasets on twitter sentiment analysis subsequently. The first is the a COVID-19 specific sentiment analysis dataset [] found online when we first started the project. It contains over 150 thousands geo-tagged tweets mined with 54 COVID-19 related keywords. However, the sentiment score was assessed using an existing API and not accurate as discovered by our inspection. Thus, we switched our dataset option, but meanwhile, we exploited the geo-tagging feature of this dataset to vividly demonstrate our results by geographically plotting the sentiment. We eventually trained our models on Sentiment140 [], a balanced tweet sentiment analysis dataset contains 1.6 millions tweets classified using emoticons, a method could classify the sentiment at a high accuracy []. We assessed and compared our performance on a manually graded testset provided by the dataset owner. 

The second step is to extract textual content from Twitter via hydrator or Twitter API. In the third step, we perform preprocessing on the raw tweets. This includes removing punctuations, hashtags, and url, performing word normalization, creating vector space format and feature selection etc.. 

The final step is to build up and train machine learning classifiers to classify the tweet sentiment as either ‘negative’ or ‘positive’. We employed four approaches, which includes three traditional machine learning methods, Logistic Regression (LR), Multinomial Naive Bayes (NB), and Support Vector Machine (SVM), and deep learning approach to tackle the problem. We set LR as the baseline model because of its simple yet effective nature in classification problems. While NB and SVM empirically show fast and relatively effective results in text classification, deep learning approaches have shown the state of the art results in various machine learning tasks including sentiment analysis. In our work, we built a Bidirectional LSTM Recurrent Neural Network (BiLSTM) model with GloVe embedding which yields the highest 83.84% accuracy in our balanced testset. While our LR, NB and SVM classifiers achieve 73.54%, 77.16% and 76.40% accuracy respectively.

The model would not be able to precisely classify sentiment on an individual level, but it can estimate the people’s sentiment towards the pandemic on a regional or national scale, and it has the potential to become a real-time estimator. At the end of the report, we demonstrated this potential with a geographic bubble chart. The color of the bubbles represents the sentiment score of being positive or negative. 

## Background

Our work lies in the larger background of sentiment analysis in NLP. Based on comparative studies on machine learning techniques in general sentiment analysis, deep learning approaches have dominated the field by achieving the state of the art in the majority of the datasets. However, deep learning methods don’t significantly outperform traditional machine learning approaches in tweet sentiment classification. The reasons behind might be due to the informal language norm, misspelling, or interruptions by hashtags and URL etc.. For example , the words ‘cooool,’ ‘borreeeed,’ and  ‘energyyyyy’, though carry important sentimental value by themselves, don’t yield statistical significance in the model and are hard to be normalized. 

Because of this and the exploration nature of this project, it is worth experimenting with multiple different machine learning techniques to approach the task. So we tried both traditional machine learning classification approaches and deep learning methods in this project.

## Related Works

It has been shown that compared to Ebola and Zika virus, information flow of twitter on coronavirus outbreak is relevant and mostly accurate and contains minor misinformation [6]. So it could potentially generate helpful public benefit in the current outbreak. 

However, while much work has been done towards various aspects of COVID-19, little work has directly addressed the problem of estimating public sentiment towards this major issue [6], which might be due to a lack of labeled dataset. Existing work [7] [8] has tried to evaluate the sentiment score by existing API’s, but this might not be effective. By our inspection, the evaluated scores are not accurate, which might be due to either the capability of the API’s algorithm or that the sentiment analysis methods of the API’s are not trained specifically to address tweets’ special language expression.

Therefore, to provide the solution, we had our data trained and tested on Sentiment140, a sentiment analysis dataset containing 1.6 millions tweets. And then we hydrated and assessed the tweets mined with 54 COVID-19 related keywords with our models to present the result. 


## Dataset

## Methods & Designs

Given the course based nature of the project,  it is worth trying out multiple different machine learning techniques for our task. We used the classic logistic regression as the baseline algorithm in this project and then implemented Multinomial NB, SVM, and BiLSTM classifiers. NB and SVM are two machine learning approaches that traditionally perform well in sentiment classification tasks. Compared to neural networks, both algorithms require comparatively short training time and yield relatively high classification accuracy with limited data. It is commonly held that Support Vector Machine has an overall high performance especially with large feature sets. While when dealing with relatively small feature sets, Naive Bayes performs well [9] [10].

### Preprocessing
### Evaluation Metric

Since our test set is almost balanced, we relied on “accuracy” as the evaluation metric to compare our models’ performance. While other classification metrics including F1 scores, recall, precision are also presented in the results.

### Baseline: Logistic Regression
#### Model Design
#### Tuning Process
#### Result

### Multinomial Naive Bayes
#### Model Design

Despite its naive probability independence assumption and inaccurate numerical prediction, Naive Bayes is an effective text classification algorithm especially for short texts. [8] We chose Multinomial Naive Bayes algorithm because the textual features are finite discrete variables. Before we fit the classifier on the training data, we used a vectorizer to vectorize the text with different combinations of n-grams. Then, we used a tf-idf (term frequency–inverse document frequency) transformer to extract textual features. Using a tf-idf transformer gives more weights to terms that have statistical significance in sentiment and less weights to common terms in the corpus which have insignificant impact on the text sentiment, for example, the stopwords.
To tune the hyperparameters of the vectorizer, transformer, classifiers, we used the PipeLine method in the scikit-learn library to pipe the chosen values for the hyperparameters, as described in the next section.


#### Tuning Process

We first trained the model on a 3000 subset of the training set, using PipeLine to collectively gather the algorithm performance. We narrowed down the hyperparameter selections by comparing the mean and standard deviation of prediction accuracy using different hyperparameter combinations. We eventually chose the following hyperparameter values.


```
text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB())])

tuned_parameters = {
    'vect__ngram_range': [(1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 3)],
    'tfidf__use_idf': [False, True],
    'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': [1, 1e-1, 1e-2],
}

score = 'f1_macro'

clf = GridSearchCV(text_clf, tuned_parameters, cv=10, scoring=score) 
```

```
tuned_parameters = {
    'vect__ngram_range': [(1, 3)],
    'tfidf__use_idf': [False],
    'tfidf__norm': ('l2'), 
    'clf__alpha': [1],
}
```

#### Result

We scaled the training set to 200,000 tweets with a training-validation split of 0.05. We then tested it on the test set. We achieved 77.16% accuracy for our Multinomial Naive Bayes model. The classification report for classifying the training, validation, and test set are illustrated in fig[].

<p align="center">
  <img src="https://github.com/miles-zijingwu/COVID-19_Twitter_Sentiment_Analysis/blob/master/Image/Naive%20Bayes/NB_class_report.png" width="500">
</p>
<p align="center">This is a centered caption for the image<p align="center">

### Support Vector Machine (SVM)
#### Model Design
#### Tuning Process
#### Result

### Bidirectional LSTM Recurrent Neural Network
#### Model Design
#### Tuning Process
#### Result

## Result Comparison
```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/miles-zijingwu/COVID19_Twitter_Sentiment_Analysis/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.

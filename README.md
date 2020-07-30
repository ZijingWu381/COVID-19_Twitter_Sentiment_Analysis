# COVID-19_Twitter_Sentiment_Analysis
#### Akash Shah, Chiche Tsai, Jiaxi Xu, Zijing Wu


![Twitter Sentiment Analysis](https://github.com/miles-zijingwu/COVID-19_Twitter_Sentiment_Analysis/blob/master/Image/heading_image.jpg)


## Introduction and Motivation

Since its discovery, the COVID-19 pandemic has quickly become a worldwide crisis that is threatening public health. By the time of writing this report, there have been over 4.3 millions infections and 149 thousands deaths cumulatively in the United States and over 16.2 millions cases and 648 thousands deaths worldwide ('Google News', 2020). Polarized emotion arises as people facing this unprecedented event and could lead to social unrest. When trying to mitigate such public health risk and reopen business, policy makers need to know the real-time and accurate assessment of the public sentiment so that they can make the most effective regulations accordingly. 

### Why Twitter?

However, the traditional approaches like census or sampling are either slow or not comprehensive. Instead, we aim to provide a faster and more comprehensive assessment of public sentiment. This could be done by performing sentiment analysis on the recent tweet data using machine learning approaches. Nowadays, people frequently share personal experience which may contain a certain level of  sentiment on social media. Especially for Twitter, the US accounts for 31 millions active daily users becoming a decent resource  to collect the public’s sentiment ('OMNICORE', 2020). Given the large number of active users, mining tweets has the potential to give fast and real estimation of people’s sentiment towards the pandemic. 

### Project Overview

In our work, we acquired two datasets on twitter sentiment analysis subsequently. The first is the a COVID-19 specific sentiment analysis dataset (Lamsal, 2020) found online when we first started the project. It contains over 150 thousands geo-tagged tweets mined with 54 COVID-19 related keywords. However, the sentiment score was assessed using an existing API and not accurate as discovered by our inspection. Thus, we switched our dataset option, but meanwhile, we exploited the geo-tagging feature of this dataset to vividly demonstrate our results by geographically plotting the sentiment. We eventually trained our models on Sentiment140 (Go et al., 2009), a balanced tweet sentiment analysis dataset contains 1.6 millions tweets classified using emoticons, a method could classify the sentiment at a high accuracy (Read et al., 2005). We assessed and compared our performance on a manually graded testset provided by the dataset owner. 

The second step is to extract textual content from Twitter via hydrator or Twitter API. In the third step, we perform preprocessing on the raw tweets. This includes removing punctuations, hashtags, and url, performing word normalization, creating vector space format and feature selection etc.. 

The final step is to build up and train machine learning classifiers to classify the tweet sentiment as either ‘negative’ or ‘positive’. We employed four approaches, which includes three traditional machine learning methods, Logistic Regression (LR), Multinomial Naive Bayes (NB), and Support Vector Machine (SVM), and deep learning approach to tackle the problem. We set LR as the baseline model because of its simple yet effective nature in classification problems. While NB and SVM empirically show fast and relatively effective results in text classification, deep learning approaches have shown the state of the art results in various machine learning tasks including sentiment analysis. In our work, we built a Bidirectional LSTM Recurrent Neural Network (BiLSTM) model with GloVe embedding which yields the highest 83.84% accuracy in our balanced testset. While our LR, NB and SVM classifiers achieve 73.54%, 77.16% and 76.40% accuracy respectively.

The model would not be able to precisely classify sentiment on an individual level, but it can estimate the people’s sentiment towards the pandemic on a regional or national scale, and it has the potential to become a real-time estimator. At the end of the report, we demonstrated this potential with a geographic bubble chart. The color of the bubbles represents the sentiment score of being positive or negative. 

## Background

Our work lies in the larger background of sentiment analysis in NLP. Based on comparative studies on machine learning techniques in general sentiment analysis, deep learning approaches have dominated the field by achieving the state of the art in the majority of the datasets. However, deep learning methods don’t significantly outperform traditional machine learning approaches in tweet sentiment classification. The reasons behind might be due to the informal language norm, misspelling, or interruptions by hashtags and URL etc.. For example , the words ‘cooool,’ ‘borreeeed,’ and  ‘energyyyyy’, though carry important sentimental value by themselves, don’t yield statistical significance in the model and are hard to be normalized. 

Because of this and the exploration nature of this project, it is worth experimenting with multiple different machine learning techniques to approach the task. So we tried both traditional machine learning classification approaches and deep learning methods in this project.

## Related Works

It has been shown that compared to Ebola and Zika virus, information flow of twitter on coronavirus outbreak is relevant and mostly accurate and contains minor misinformation (Kaila et al., 2020). So it could potentially generate helpful public benefit in the current outbreak. 

However, while much work has been done towards various aspects of COVID-19, little work has directly addressed the problem of estimating public sentiment towards this major issue, which might be due to a lack of labeled dataset. Existing work (Samuel et al., 2020), (Karisani, N., & Karisani, P., 2020) has tried to evaluate the sentiment score by existing API’s, but this might not be effective. By our inspection, the evaluated scores are not accurate, which might be due to either the capability of the API’s algorithm or that the sentiment analysis methods of the API’s are not trained specifically to address tweets’ special language expression. 

Therefore, to provide the solution, we had our data trained and tested on Sentiment140, a sentiment analysis dataset containing 1.6 millions tweets. And then we hydrated and assessed the tweets mined with 54 COVID-19 related keywords with our models to present the result. 


## Dataset

### Training and Validation Set

The dataset we used to train our models is Sentiment140 (Go et al., 2009). It is a balanced dataset containing around 1.6 million tweets, with 0.8 million tweets each for tweets with positive and negative sentiment. Sentiment polarity is labeled by 0 and 4, corresponding to negative and positive sentiment. The data is mined and classified using emoticons. The emoticons are removed in the tweets afterwards. A tweet was labeled as positive if it contained emoticons with positive sentiment, and vice versa. This method has been shown effective for identifying the sentiment (Read et al., 2005). The data would be preprocessed. An appropriate amount of tweets would be randomly selected when training each classifier and splitted into training and validation set by a customized ratio.

### Testing Set

The test set contains 177 negative and 182 positive tweets after dropping tweets with neutral sentiment. The sentiment scores are manually graded to ensure the correctness. The performance of each machine learning model is assessed by its accuracy on the testing set to gain an unbiased comparison. In the end, we applied the models on the geo-tagged COVID-19 tweet dataset to visualize the sentiment scores on a map. 

## Methods & Designs

Given the course based nature of the project, it is worth trying out multiple different machine learning techniques for our task. We used the classic logistic regression as the baseline algorithm in this project and then implemented Multinomial NB, SVM, and BiLSTM classifiers. NB and SVM are two machine learning approaches that traditionally perform well in sentiment classification tasks. Compared to neural networks, both algorithms require comparatively short training time and yield relatively high classification accuracy with limited data. It is commonly held that Support Vector Machine has an overall high performance especially with large feature sets. While when dealing with relatively small feature sets, Naive Bayes performs well (Bhavitha et al., 2017), (Mittal & Patida, 2019). Finally, we leverage the large amount of data to build and train a BiLSTM classifier. 

The following will elaborate our text preprocessing process, evaluation metric. For each model, we will explain the model design and tuning process, following which we will present the result on the test set for comparison. 

### Preprocessing

For the training data, we first read in the csv files and keep only 'Tweets' and 'Labels' columns. Originally in the dataset, positive and negative tweets were labeled by '4' and '0' respectively. We replaced '4' with '1.' For each individual tweets, we apply the following preprocessting steps:

1. Normalizing text by lemmatization and converting words to lower case
2. Remove common tweet elements don't contribute to the sentiment, urls, usernames, and hashtags
3. Remove words containing digits
4. Expand all contractions like, for example, 'don't' to 'do not'
5. Remove punctuations
6. Delete tweets has encoding errors, for there were the non-English tweets that were not correctly encoded by the encoder
7. Tokenize and turn each tweet string into a list of tokens.

For the test data, besides applying the same preprocessing process, we remove tweets labeled with neutral sentiment since our classification is binary.

### Evaluation Metric

Since our test set is almost balanced, we relied on “accuracy” as the evaluation metric to compare our models’ performance. While other classification metrics including F1 scores, recall, precision are also presented in the results.

### Baseline: Logistic Regression
#### Model Design

Logistic Regression was implemented as the baseline model due to its relatively simple implementation as well as its general intuitiveness. Logistic Regression will perform well in many tasks and is a great place to start when building models. In our literature research, we found that logistic regression was fairly accurate in analyzing and predicting shorter tweets and decided to implement it and then work to improve prediction accuracy with various other models. Before this model, along with the future models, could be trained, the text data had to be vectorized in order to assign a number to different n-grams, as well as to extract text features with higher significance toward sentiment. This model was then trained and validated on a 200,000 tweet dataset, using 75% to train and 25% to validate. 
<p align = "center">
  <img src = "https://github.com/miles-zijingwu/COVID-19_Twitter_Sentiment_Analysis/blob/master/LogRegression.png" width="500">
</p>
<p align="center">Visual demonstration of the Logistic Regression model</p>
<p align = "center">Source: https://machinelearning-blog.com/2018/04/23/logistic-regression-101/</p>
  

#### Tuning Process

Next, the model hyperparameters were tuned and chosen based on the mean and standard deviation of each combination of hyperparameters. The hyperparameters most important for Logistic Regression are ‘solver’, ‘penalty’, and ‘C’, and we chose the values based on the following results.

```
logModel = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]
grid = dict(solver = solvers, penalty = penalty, C = c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator = logModel, param_grid = grid, n_jobs=-1, cv = cv, scoring='accuracy', error_score=0)
grid_result = grid_search.fit(x_train_features, y_train)

>>> Best: 0.819647 using {'C': 10, 'penalty': 'l2', 'solver': 'newton-cg'}
```
#### Result

The model results were indeed quite successful for a baseline model. The above result shows the result of the validation set; however, on the test set the model had an accuracy score of 73.54%. This was very pleasing, but there was certainly room for improvement with further work in implementing some more sophisticated models.
<p align="center">
  <img src = "https://github.com/miles-zijingwu/COVID-19_Twitter_Sentiment_Analysis/blob/master/results.png" width="500">
</p>
<p align="center">The testset classification report for the LR model <p align="center">

### Multinomial Naive Bayes
#### Model Design

Despite its naive probability independence assumption and inaccurate numerical prediction, Naive Bayes is an effective text classification algorithm especially for short texts. [8] We chose Multinomial Naive Bayes algorithm because the textual features are finite discrete variables. Before we fit the classifier on the training data, we used a vectorizer to vectorize the text with different combinations of n-grams. Then, we used a tf-idf (term frequency–inverse document frequency) transformer to extract textual features. Using a tf-idf transformer gives more weights to terms that have statistical significance in sentiment and less weights to common terms in the corpus which have insignificant impact on the text sentiment, for example, the stopwords.
To tune the hyperparameters of the vectorizer, transformer, classifiers, we used the PipeLine method in the scikit-learn library to pipe the chosen values for the hyperparameters, as described in the next section.


#### Tuning Process

We first trained the model on a 3000 subset of the training set, using PipeLine to collectively gather the algorithm performance. We narrowed down the hyperparameter selections by comparing the mean and standard deviation of prediction accuracy using different hyperparameter combinations.


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
We eventually chose the following hyperparameter values.

```
tuned_parameters = {
    'vect__ngram_range': [(1, 3)],
    'tfidf__use_idf': [False],
    'tfidf__norm': ('l2'), 
    'clf__alpha': [1],
}
```

#### Result

We scaled the training set to 200,000 tweets with a training-validation split of 0.05. We then tested it on the test set. We achieved 77.16% accuracy for our Multinomial Naive Bayes model. The classification report for classifying the training, validation, and test set are illustrated in the folowing.

<p align="center">
  <img src="https://github.com/miles-zijingwu/COVID-19_Twitter_Sentiment_Analysis/blob/master/Image/Naive%20Bayes/NB_class_testset_report.png" width="500">
</p>
<p align="center">The testset classification report for the NB model <p align="center">

### Support Vector Machine (SVM)
#### Model Design

In the training phase, the dataset is applied by additional preprocessing. The tweets are tokenized  into words as features by TfidfVectorizer. In order to prevent from having too many features which could be much greater than the number of samples, the model randomly extracts 5000 features to train the model. The model uses 70% of 100,000 data to train and 30% to validate. 

```
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(data, labels, test_size = 0.3, random_state = 42)
Tfidf_vect = TfidfVectorizer(max_features = 5000)
```

#### Tuning Process

The model parameters are tuned properly. The most important parameter for the SVM model would be the kernel function which generates the best hyperplane that maximizes the margins. The kernel function chosen in this model is linear kernel The model has been tried on other common kernel functions such as polynomial, sigmoid, and Gaussian radial basis function; however, their accuracy is far below the one trained on the linear kernel. Furthermore, the regularization parameter which serves as the cost of misclassification is set to 1. That is, the model more cannot accept misclassification when the regularization parameter is higher. Here, we use exponentially growing sequences to identify the best value for the regularization parameter in this model from 2-5to 25.

```
SVM = svm.SVC(C = 1, kernel = 'linear', degree = 1, gamma = 'auto', verbose = 'False')
```

#### Result

In the validation and testing result, the model gets 77.32% of accuracy for positive sentiment, 80.61% for negative sentiment, and 78.94% for the overall. Since we use a balanced dataset to train the model, the accuracy between positive and negative sentiment does not have an obvious difference. The accuracy of negative sentiment is slightly higher than the positive sentiment. Speaking of the overall performance, the SVM model performs quite well regarding the common machine learning classification models. 

<p align = "center">
  <img src = "https://github.com/miles-zijingwu/COVID-19_Twitter_Sentiment_Analysis/blob/master/Image/SVM/SVM_results_v2.png" width = "500">
</p>
<p align="center">The testset classification report for the SVM model <p align="center">

### Bidirectional LSTM Recurrent Neural Network

#### Model Design

Although NB and SVM have shown their consistent reliable performance in a variety of text classification tasks, in recent years, deep learning methods have outperformed them as more data are available and more neural network architectures and word embedding methods are designed. And given the large size of the 1.6 million training data, we decided to build a neural network model to classify the sentiment. 

For word embedding, we used a pre-trained GloVe word vector obtained by crawling 840B words online (Pennington et al., 2014). The words vector has 2.2 million vocabulary and 300 dimensional semantic properties. With word embedding, we are able to train a model to predict sentiment of words even though they are not contained in the training set vocabulary. For example, we could still use our model to accurately predict the sentiment of “It is fantastic,” suppose the term ‘fantastic’ is not in our vocabulary,  because a synonym of it, take ‘awesome’ for example, is in our vocabulary. That the two terms have similar word representation vectors in our pre-train GloVe model enables our model to transfer its training. 

We employed 2 Bidirectional LSTM layers (BiLSTM) in our model architecture. BiLSTM is a sequential neural network layer that could capture the long-term dependency among the words in a sentence. A single module of a BiLSTM layer is illustrated below.

<p align="center">
  <img src="https://github.com/miles-zijingwu/COVID-19_Twitter_Sentiment_Analysis/blob/master/Image/BiLSTM/LSTM_architecture_explained.png" width="600">
</p>
<p align="center">Source: https://colah.github.io/posts/2015-08-Understanding-LSTMs/" <p align="center">
  
In practice, although we found that CNN (Convolutional Neural Network) requires significantly less training time and produce higher results compared the other three machine learning algorithms we implemented, Recurrent Neural Network offers higher performance in all the classification evaluation metrics. The figure shows the overall architecture of our model. The model used Adam as the optimizer and binary cross entropyloss function metric.

<p align="center">
  <img src="https://github.com/miles-zijingwu/COVID-19_Twitter_Sentiment_Analysis/blob/master/Image/BiLSTM/NN_architecture.png" width="350" >
</p>
<p align="center">The architecture of the model. The input and output dimensions are indicated.">
  
#### Tuning Process

While the implementation of the model was done on personal computer, the whole training is carried out on Google colab using its cloud computing power. We used the platform's TPU to accelerate the experiment iteration.

We first trained our model on a 20,000 subset of the training data with a train validation splitting ratio of 0.2. The small number of tweets let us make quick updates of the hyperparameters with short iterations. We apply orthogonal principle to our tuning procedure. Eventually, the maximum size of vocabulary was set as 20,000. The batch size was set to be 256. The maximum length of a sentence padding is 40. We applied a customized exponential learning rate decay which is constant in the initial 2 epochs and exponentially decays afterwards. The start learning rate is set to be 0.0018. We then trained our model on the full training set to fine tuned the start learning rate and make appropriate adjustments to the epoch numbers. A 0.0125 train-validation split is applied to make the model validate its training on about 20,000 tweets after each epoch.


#### Result

The model was trained for 8 epochs, which took 6 hours. The validation loss reached the minimum at around 0.40 in the 5th epoch, and the validation accuracy started to fluctuate around 82% after the 3rd epoch. 

<p align="center">
  <img src="https://github.com/miles-zijingwu/COVID-19_Twitter_Sentiment_Analysis/blob/master/Image/BiLSTM/NN_loss_fx_acc.png" width="500">
</p>
<p align="center">The trend of the loss and accuracy of the model during training <p align="center">

As showned in the figure below, the model correctly predicted 83.84% of the sentiment polarity of the tweets in the text set. Even though the size of the test set is relatively small, the validation set accuracy on about 20,000 tweets is as high as 82.06%. The model's performance on both validation and the manually graded test set shows its robustness on predicting tweets sentiment.

<p align="center">
  <img src="https://github.com/miles-zijingwu/COVID-19_Twitter_Sentiment_Analysis/blob/master/Image/BiLSTM/NN_class_testset_report.png" width="500">
</p>
<p align="center">The testset classification report for the BiLSTM model <p align="center">
  

## Result Comparison

Here is a comparison of the results.

<p align = "center">
  <img src = "https://github.com/miles-zijingwu/COVID-19_Twitter_Sentiment_Analysis/blob/master/Image/model_result_comparison.png" width="500">
</p>

## Visualisation: Geographic Distribution of Sentiment 

We used js.D3 for data visualisation ('What is D3.js', 2018). It is an open-source JavaScript library developed by Mike Bostock to create custom interactive data visualizations in the web browser using SVG, HTML and CSS (Holtz, 2019). One great feature is that it provides an abundant toolbox rather than fixed applications so that we can customize the graphs based on our needs. Geographical Bubble chart is the basic format in our case, with some deviations. First, the size of the bubble is constant, while the locations become the emphasis. Second, the color legend indicates that green color corresponds to positive sentiment, such as happiness and kindness, while the red color represents negative sentiment, such as sadness and hate. 

The calculated accuracy for BiSTML and Naive Bayes are 83.84%, 77.16% respectively. Comparing the results of them on the map, BiSTML more faithfully demonstrates that on June 1st, the public opinion towards COVID19 was still largely passive. On the contrary, the result from NB still claims that the sentiment was relatively optimistic.  

       
<p align="center">
  <img src="https://github.com/miles-zijingwu/COVID-19_Twitter_Sentiment_Analysis/blob/master/Image/resuts2/map_bilstm.png" width="500">
</p>

<p align="center">
  <img src="https://github.com/miles-zijingwu/COVID-19_Twitter_Sentiment_Analysis/blob/master/Image/resuts2/map_nb.png" width="500">
</p>

<p align="center">
  <img src="https://github.com/miles-zijingwu/COVID-19_Twitter_Sentiment_Analysis/blob/master/Image/resuts2/legend.png" width="250">
</p>

## Summary and Conclusion

The purpose of this project was to develop a tool to analyze public sentiment on Twitter about the COVID-19 pandemic. It is an important topic in today’s world as many people express themselves on social media and being able to appropriately analyze and predict public reactions to events is an extremely powerful tool. In this project, four models were trained for their accuracy in classifying positive and negative tweets on a general set of tweets, then applied to a set of tweets about COVID-19. Each model has its own pros and cons, but the overall outcome aligned with the expected outcome that the recurrent neural net would perform the best. 
<table style = "width: 75%">
  <tr>
    <th>Model</th>
    <th style = "width:150px">Pros</th>
    <th style = "width:150px">Cons</th>
  </tr>
  <tr>
    <td>LR</td>
    <td>
      <ul>
        <li>Relatively quick and easy implementation make it a good baseline model</li>
        <li>Efficient to train and tune hyperparameters</li>
      </ul>
    </td>
    <td>
      <ul>
        <li>Cannot solve nonlinear problems-easy to misclassify a tweet based on one feature</li>
        <li>Quite vulnerable to overfitting</li>
      </ul>
    </td>
  </tr>
  
  <tr>
    <td>NB</td>
    <td>
      <ul>
        <li>Efficient to tune with its few hyperparameters</li>
        <li>Perform well especially on short sentences</li>
      </ul>
    </td>
    <td>
      <ul>
        <li>Assume independence among words in the sentences</li>
        <li>Hard to generalize the learning to new form of data because the model heavily relies on the probability relation between the training data and predicted sentiment</li>
      </ul>
    </td>
  </tr>
  
  <tr>
    <td>SVM</td>
    <td>
      <ul>
        <li>Accurate in high dimensional space</li>
        <li>Memory efficiency</li>
      </ul>
    </td>
    <td>
      <ul>
        <li>Training time is very long if the dataset is large</li>
        <li>Prone to overfit the training date if number of feature is greater than the number of samples</li>
        <li>Do not proved probability estimate</li>
      </ul>
    </td>
  </tr>
  
  <tr>
    <td>BiLSTM</td>
    <td>
      <ul>
        <li>Achieve higher accuracy compared to other models by a relatively large margin</li>
        <li>Could capture sentiment of words not in the training set vocabulary because of the word embedding implementation</li>
        <li>Has the potential to achieve high performance when more data is available</li>
      </ul>
    </td>
    <td>
      <ul>
        <li>Many hyperparameters to tune, need to apply orthogonality principle rigorously to prevent misunderstanding the causality between the tuning and result</li>
        <li>Need large amount of time and computing resources to train</li>
      </ul>
    </td>
  </tr>
</table>

The neural network’s accuracy of 83.84% was very high and quite pleasing for the topic of estimating sentiment. This is a challenging process as sarcasm and other grammatical structures are generally unknown to a model, so this accuracy rate is certainly useful for a future use by public policy makers in enacting restrictions and guidelines for their respective cities. As the data in the previous section show, sentiment across Europe was moving to be relatively more positive, whereas America was becoming more negative. This analysis, in combination with recommendations from the CDC has the potential to be a real time estimator of sentiment to aid in the creation of mask-wearing and social distancing mandates and regulations. Moving forward, more globalized data and more data points would certainly be needed, but this neural network model is a great start to finding an effective solution for policy makers around the country and the world.


## Future Work

Stop word removal can be applied in the data preprocessing phase. The dataset will become more dense and consistent and the model will be trained more robustly. Furthermore, figurative language such as irony, sarcasm and metaphors is a hard issue for our model. If any tweets tries to use figurative language, it is very high possible that our models will misinterpret the sentiment.  Also we think emoticons could be important for classifying sentiment. It is removed in the training set for our project. However, it is one of the central elements of text sentiment and thus could be considered in future work.

## Reference

Bhavitha, B. K., Rodrigues, A. P., & Chiplunkar, N. N. (2017, March). Comparative study of machine learning techniques in sentimental analysis. In 2017 International Conference on Inventive Communication and Computational Technologies (ICICCT) (pp. 216-221). IEEE.

Go, A., Bhayani, R., & Huang, L. (2009). Twitter sentiment classification using distant supervision. CS224N project report, Stanford, 1(12), 2009.

Google News. (2020). Coronavirus (covid-19). Retrieved June 16, 2020, from https://news.google.com/covid19/map?hl=en-US&mid=%2Fm%2F09c7w0&gl=US&ceid=US%3Aen 

Karisani, N., & Karisani, P. (2020). Mining Coronavirus (COVID-19) Posts in Social Media. arXiv preprint arXiv:2004.06778.

Mittal, A., & Patidar, S. (2019, July). Sentiment Analysis on Twitter Data: A Survey. In Proceedings of the 2019 7th International Conference on Computer and Communications Management (pp. 91-95).

OMNICORE. (2020). Twitter by the numbers: Stats, demographics fun facts. Retrieved February 10, 2020, from https://www.omnicoreagency.com/twitter-statistics/

Pennington, J., Socher, R., & Manning, C. D. (2014, October). Glove: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP) (pp. 1532-1543).

Kaila, R. P., Prasad, A. V. K., & House, F. G. (2020). Informational flow on twitter – corona virus outbreak – topic modelling approach. nternational Journal of Advanced Research in Engi- neering and Technology (IJARET), 11(3), 128–134.

Lamsal, R. (2020). Coronavirus (covid-19) geo-tagged tweets dataset. IEEE Dataport. https://doi. org/10.21227/fpsb-jz61

Read, J. (2005, June). Using emoticons to reduce dependency in machine learning techniques for sentiment classification. In Proceedings of the ACL student research workshop (pp. 43-48).

Samuel, J., Ali, G. G., Rahman, M., Esawi, E., & Samuel, Y. (2020). Covid-19 public sentiment insights and machine learning for tweets classification. Information, 11(6), 314.

What is D3.js? (2018) Retrieved June 27, 2020, from https://www.tutorialsteacher.com/d3js/what-is-d3js.

Holtz, Y. (2019). The D3 Graph Gallery – Simple charts made in d3.js. The D3 Graph Gallery – Simple charts made with d3.js. Retrieved June 20, 2020,from https://www.d3-graph-gallery.com/.



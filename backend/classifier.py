"""
Classifier to determine if tweets are from Trump or Staff.
Part of the Trump vs Staff web app

Alexander Li (afl59), Rohan Patel (rp442)
1/10/19
"""

import numpy as np
import csv

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
#import matplotlib.pyplot as plt

 #training data
#testData = csv.DictReader(open('test.csv')) #test submission data

"""Train classifier functions"""



#Return random list of tweets and associated labels
def gatherData():
    trainData = csv.DictReader(open('train.csv'))
    tweets = []
    labels = []
    for x in trainData:
        #tweet = x['text']
        tweet = featureEngineer(x)
        tweets.append(tweet)
        label = x['label']
        labels.append(int(x['label']))
    tweets = np.array(tweets)
    labels = np.array(labels)
    #randomize order of xTr and yTr while keeping relations
    n = len(tweets)
    indices = np.random.choice(n, n, replace=False)
    tweets = tweets[indices]
    labels = labels[indices]

    return tweets, labels

#given a datapoint, extract interesting features
def featureEngineer(data):
    tweet = data['text']
    # #time
    # timedate = data['created']
    # space = timedate.find(' ')
    # justtime = timedate[space+1:]
    # hour = int(justtime[:justtime.find(':')])
    # if hour > 18 or hour < 3:
    #     time = 'nightnight'
    # elif hour >= 3 and hour <= 12:
    #     time = 'mornmorn'
    # else:
    #     time = 'noonnoon'
    # tweet = tweet + ' ' + time
    # #link
    # if 'https' in tweet:
    #     link = 'linklink'
    #     tweet = tweet + ' ' + link
    # #hash
    # if '#' in tweet:
    #     hash = 'hashash'
    #     tweet = tweet = ' ' + hash
    return tweet

def extractFeatures(tweets, labels):
    n = len(tweets)
    #split valid and train
    xTrain = tweets #xTrain = tweets[:int(n*.8)]
    yTrain = labels #yTrain = labels[:int(n*.8)]
    xValid = tweets[int(n*.8):int(n*.9)]
    yValid = labels[int(n*.8):int(n*.9)]
    xTest = tweets[int(n*.9):]
    yTest = labels[int(n*.9):]

    #vec = CountVectorizer(stop_words='english', binary = True)
    vec = TfidfVectorizer(ngram_range = (1,2), strip_accents = 'unicode',
    min_df = 2, decode_error = 'replace', analyzer = 'word')
    xTrain = vec.fit_transform(xTrain).toarray()
    xValid = vec.transform(xValid).toarray()
    #select best features
    selector = SelectKBest(f_classif, k=min(20000, xTrain.shape[1]))
    selector.fit(xTrain, yTrain)
    xTrain = selector.transform(xTrain).astype('float32')
    xValid = selector.transform(xValid).astype('float32')
    return xTrain, yTrain, xValid, yValid, vec, selector, xTest, yTest

#Trains and validates given xTr and yTr on many different models
def trainAndValidate(xTrain, yTrain, xValid, yValid):
    # models.append(RandomForestClassifier(n_estimators=100, max_depth=None,random_state=0, oob_score=True))
    # models.append(MultinomialNB())
    # #models.append(MLPClassifier(solver='sgd', activation = 'relu', alpha=1e-5, hidden_layer_sizes=(64,), random_state=1, max_iter = 5000))

    model = AdaBoostClassifier(n_estimators = 100, learning_rate = .1)

    model.fit(xTrain,yTrain)
    pred = model.predict(xTrain)
    trainingError = sum(yTrain!=pred)/len(yTrain)
    predV = model.predict(xValid)
    validationError = sum(np.array(yValid)!=np.array(predV))/len(yValid)

    return model, trainingError, validationError


#for Submission
def prepareSubmit(pred):
    with open('submit.csv', mode='w') as submit:
        writer = csv.writer(submit, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['ID', 'Label'])
        for i in range(len(pred)):
             id = i
             label = pred[i]
             writer.writerow([str(i), str(label)])
#--------------------------End Helper Functions -------------------------

def training():
    tweets, labels = gatherData()
    print('extracting features')
    xTrain, yTrain, xValid, yValid, vec, selector, xTest, yTest = extractFeatures(tweets, labels)
    print('training')
    model, trainingError, validationError = trainAndValidate(xTrain, yTrain, xValid, yValid)
    return model, vec, selector, int(trainingError*100), int(validationError*100)

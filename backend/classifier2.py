"""
Classifier to determine if tweets are from Trump or Staff.
Part of the Trump vs Staff web app

Alexander Li (afl59), Rohan Patel (rp442)
1/24/19
From Yurong You
"""

import numpy as np
import csv
from xgboost import XGBClassifier
import xgboost as xgb
import pandas as pd
from sklearn import svm, utils
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.preprocessing import normalize, MinMaxScaler

import itertools
from collections import Counter
import re

def datetime2int(datetime):
    # input example: "7/12/16 0:56"
    time = datetime.split(" ")[1]
    hour, minute = int(time.split(':')[0]), int(time.split(':')[1])
    return hour * 60 + minute

def get_data(filename, get_label=False):
    care_features = ['text', 'favoriteCount', 'created', 'retweetCount']
    data = pd.read_csv(filename, usecols=care_features)
    if get_label:
        label = pd.read_csv(filename, usecols=['label'])
        label = label['label'].tolist()
        return data, label
    return data

def getWords(text):
    return re.compile('\w+').findall(text)
def countwords(raw_data, train_label):
    android, iphone = [], []
    android.append([getWords(x.lower()) for i, x in enumerate(raw_data['text']) if train_label[i] == 1])
    iphone.append([getWords(x.lower()) for i, x in enumerate(raw_data['text']) if train_label[i] == -1])
    a = Counter(itertools.chain.from_iterable(tweet for tweet in android[0]))
    b = Counter(itertools.chain.from_iterable(tweet for tweet in iphone[0]))
    print('Andriod:\n',a, '\n Iphone:\n', b)


def process_raw_data(raw_data):
    # here we first try ignore text data just use favoriteCount, created, retweetCount
    # size = len(raw_data['favoriteCount'])
    raw_text = pd.DataFrame(raw_data['text']).to_dict('records')
    raw_data = raw_data.to_dict('list')
    num_data = len(raw_text)
    extract_feats = ['http', '#', '@', 'thank', '"', ' so ', ' - ', '&amp;', 'makeamericagreatagain',
                     'hillary', 'all', ' trump ', 'join']
    text_vec = np.zeros((num_data, len(extract_feats)))
    end_with_exc = [] # end with Exclamation !
    length = []
    # process text
#     vec = DictVectorizer()
#     text_vec = vec.fit_transform(raw_text).toarray()
#     hasher = FeatureHasher(n_features=300)
#     text_vec = hasher.transform(raw_text).toarray()
#     vec = CountVectorizer()
#     text_vec = vec.fit_transform(raw_data['text']).toarray()
#     vec = HashingVectorizer(stop_words='english', alternate_sign=False, n_features=2**6)
#     text_hashing = vec.fit_transform(raw_data['text']).toarray()
#     documents = [gensim.models.doc2vec.TaggedDocument(doc, str(label)) for doc, label in zip(raw_data['text'], train_label)]
#     model = gensim.models.doc2vec.Doc2Vec(documents, vector_size=10, window=2, min_count=1, workers=4)
#     text_vec = [model.infer_vector(text.split()) for text in raw_data["text"]]

    for i, text in enumerate(raw_data['text']):
        length.append(len(text))
        text_vec[i] = list(map(lambda x: x in text.lower(), extract_feats))
        end_with_exc.append(text[-1] == '!')
    # process metadata
    favCount = list(map(int, raw_data['favoriteCount']))
    created = list(map(datetime2int, raw_data['created']))
    rtwtCount = list(map(int, raw_data['retweetCount']))
    data = np.stack((favCount, created, rtwtCount, length, end_with_exc), axis=1)
    data = np.concatenate((data, text_vec), axis=1)
#     data = normalize(data)

    return data

raw_train_data, train_label = get_data('train.csv', get_label=True)
raw_test_data = get_data('test.csv', get_label=False)

train_data = process_raw_data(raw_train_data)
test_data = process_raw_data(raw_test_data)
FEATURE_SIZE = train_data.shape[-1]
print("# train data points: {}".format(train_data.shape[0]))
print("# test data points : {}".format(test_data.shape[0]))
print("feature size: {}".format(train_data.shape[-1]))

# split train to train and eval set
seed = 7
val_size = 0.1
split = int(val_size * len(train_data))
X_train, y_train = train_data[split:], train_label[split:]
X_val, y_val = train_data[:split], train_label[:split]
print("# train data points: {}".format(X_train.shape[0]))
print("# val data points : {}".format(X_val.shape[0]))


def gen_test_labels(model, test_data, name="output"):
    y_pred = model.predict(test_data)
    predictions = [round(value) for value in y_pred]
    with open('data/' + name + '.csv', 'w') as csvfile:
        fieldnames = ['ID', 'Label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, pred in enumerate(predictions):
            writer.writerow({'ID': str(i), 'Label': str(pred)})

model = XGBClassifier(n_estimators=2000, learning_rate=0.01, max_depth=2)
model.fit(X_train, y_train)

y_pred = model.predict(X_train)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_train, predictions)
print("Train Accuracy: %.2f%%" % (accuracy * 100.0))

y_pred = model.predict(X_val)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_val, predictions)
print("Test Accuracy: %.2f%%" % (accuracy * 100.0))

model = XGBClassifier(n_estimators=2500, learning_rate=0.01, max_depth=2)
model.fit(train_data, train_label)

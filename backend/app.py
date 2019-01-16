from flask import Flask, request, jsonify
from sklearn import svm
from sklearn import datasets
from sklearn.externals import joblib
from classifier import *
from tweets import *
import numpy as np


# declare constants
HOST = '0.0.0.0'
PORT = 8081

# initialize flask application
app = Flask(__name__)


@app.route('/api/train', methods=['POST'])
def train():
    # get parameters from request
    #parameters = request.get_json()

    #adaboost, vec, selector, trainingError, validationError = training()
    # persist model
    #joblib.dump(adaboost, 'model.pkl')
    #joblib.dump(vec, 'vectorizer.pkl')
    #joblib.dump(selector, 'selector.pkl')
    #use existing models
    adaboost = joblib.load('model.pkl')
    vec = joblib.load('vectorizer.pkl')
    selector = joblib.load('selector.pkl')
    # get tweets and predict
    recentTweets = getTweets()
    preds, predsNumber = predictTweets(recentTweets, adaboost, vec, selector)
    #return jsonify({'accuracy': round(clf.score(X, y) * 100, 2)})
    #1 is Trump
    #-1 is Staff
    return jsonify({'accuracy': 'not used',
    'tweetOne': recentTweets[0] + preds[0],
    'tweetOnePred': predsNumber[0],
    'tweetTwo': recentTweets[1] + preds[1],
    'tweetTwoPred': predsNumber[1],
    'tweetThree': recentTweets[2] + preds[2],
    'tweetThreePred': predsNumber[2],
    'tweetFour': recentTweets[3] + preds[3],
    'tweetFourPred': predsNumber[3],
    'tweetFive': recentTweets[4] + preds[4],
    'tweetFivePred': predsNumber[4],
    'tweetSix': recentTweets[5] + preds[5],
    'tweetSixPred': predsNumber[5],
    'tweetSeven': recentTweets[6] + preds[6],
    'tweetSevenPred': predsNumber[6],
    'tweetEight': recentTweets[7] + preds[7],
    'tweetEightPred': predsNumber[8],
    'tweetNine': recentTweets[8] + preds[8],
    'tweetNinePred': predsNumber[8],
    'tweetTen': recentTweets[9] + preds[9],
    'tweetTenPred': predsNumber[9]})


@app.route('/api/predict', methods=['POST'])
def predict():
    # get iris object from request
    X = request.get_json()
    X = str(X['text'])

    adaboost = joblib.load('model.pkl')
    vec = joblib.load('vectorizer.pkl')
    selector = joblib.load('selector.pkl')

    tweet = vec.transform([X])
    tweet = selector.transform(tweet)
    prob = adaboost.predict_proba(tweet)

    return jsonify([{'name': 'Trump', 'value': int(prob[0][1]*100)},
                    {'name': 'Staff', 'value': int(prob[0][0]*100)}])

if __name__ == '__main__':
    # run web server
    app.run(host=HOST,
            debug=True,  # automatic reloading enabled
            port=PORT)

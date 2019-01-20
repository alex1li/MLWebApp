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
    preds, predsNumber, predsUrl = predictTweets(recentTweets, adaboost, vec, selector)
    #return jsonify({'accuracy': round(clf.score(X, y) * 100, 2)})
    #1 is Trump
    #-1 is Staff
    return jsonify({'accuracy': 'not used',
    'tweetOne': recentTweets[0],
    'tweetOnePred': preds[0],
    'tweetOneUrl': predsUrl[0],

    'tweetTwo': recentTweets[1],
    'tweetTwoPred': preds[1],
    'tweetTwoUrl': predsUrl[1],

    'tweetThree': recentTweets[2],
    'tweetThreePred': preds[2],
    'tweetThreeUrl': predsUrl[2],

    'tweetFour': recentTweets[3],
    'tweetFourPred': preds[3],
    'tweetFourUrl': predsUrl[3],

    'tweetFive': recentTweets[4],
    'tweetFivePred': preds[4],
    'tweetFiveUrl': predsUrl[4],

    'tweetSix': recentTweets[5],
    'tweetSixPred': preds[5],
    'tweetSixUrl': predsUrl[5],

    'tweetSeven': recentTweets[6],
    'tweetSevenPred': preds[6],
    'tweetSevenUrl': predsUrl[6],

    'tweetEight': recentTweets[7],
    'tweetEightPred': preds[7],
    'tweetEightUrl': predsUrl[7],

    'tweetNine': recentTweets[8],
    'tweetNinePred': preds[8],
    'tweetNineUrl': predsUrl[8],

    'tweetTen': recentTweets[9],
    'tweetTenPred': preds[9],
    'tweetTenUrl': predsUrl[9]})


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

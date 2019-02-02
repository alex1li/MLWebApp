from flask import Flask, request, jsonify
from sklearn import svm
from sklearn import datasets
from sklearn.externals import joblib
from classifier import *
from classifier2 import *
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
    """
    adaboost = joblib.load('model.pkl')
    vec = joblib.load('vectorizer.pkl')
    selector = joblib.load('selector.pkl')
    # get tweets and predict
    recentTweets = getTweets()
    preds, predsNumber, predsUrl = predictTweets(recentTweets, adaboost, vec, selector)
    """

    #Classifier 2 stuff -------------------------------------------------------
    # raw_train_data, train_label = get_data('train.csv', get_label=True)
    # train_data = process_raw_data(raw_train_data)
    # FEATURE_SIZE = train_data.shape[-1]
    #
    # model = XGBClassifier(n_estimators=2500, learning_rate=0.01, max_depth=2)
    # model.fit(train_data, train_label)
    # joblib.dump(model, 'xgboost.pkl')
    model = joblib.load('xgboost.pkl')

    retry = 5
    data, recentTweets = getTweetsData()
    if len(recentTweets) < 2 and retry > 0:
        data, recentTweets = getTweetsData()
        retry -= 1
    processed = process_raw_data(data)

    preds = model.predict(processed).tolist()
    predsUrl = preds[:]
    for i in range(len(preds)):
        if preds[i] > 0:
            preds[i] = "Donald J. Trump"
            predsUrl[i] = "https://pbs.twimg.com/profile_images/874276197357596672/kUuht00m_400x400.jpg"
        else:
            preds[i] = "White House Staff"
            predsUrl[i] = "https://abs.twimg.com/sticky/default_profile_images/default_profile_400x400.png"
    #End Classifier 2 stuff ----------------------------------------------------

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
    'tweetTenUrl': predsUrl[9],

    'tweet11': recentTweets[10],
    'tweet11Pred': preds[10],
    'tweet11Url': predsUrl[10],

    'tweet12': recentTweets[11],
    'tweet12Pred': preds[11],
    'tweet12Url': predsUrl[11],

    'tweet13': recentTweets[12],
    'tweet13Pred': preds[12],
    'tweet13Url': predsUrl[12],

    'tweet14': recentTweets[13],
    'tweet14Pred': preds[13],
    'tweet14Url': predsUrl[13],

    'tweet15': recentTweets[14],
    'tweet15Pred': preds[14],
    'tweet15Url': predsUrl[14]})


@app.route('/api/predict', methods=['POST'])
def predict():
    # get iris object from request
    X = request.get_json()
    if X is None or str(X['text']) == "":
        prob = [[0,1]] #placeholder
        author = 'Please enter a tweet'
    else:
        X = str(X['text'])

        adaboost = joblib.load('model.pkl')
        vec = joblib.load('vectorizer.pkl')
        selector = joblib.load('selector.pkl')

        tweet = vec.transform([X])
        tweet = selector.transform(tweet)
        prob = adaboost.predict_proba(tweet)

        if int(prob[0][1]*100) > int(prob[0][0]*100):
            author = 'Donald J. Trump'
        else:
            author = 'White House Staff'

    # return jsonify([{'name': 'Trump', 'value': int(prob[0][1]*100)},
    #                 {'name': 'Staff', 'value': int(prob[0][0]*100)}])
    return jsonify({'prediction': author})

if __name__ == '__main__':
    # run web server
    app.run(host=HOST,
            debug=True,  # automatic reloading enabled
            port=PORT)

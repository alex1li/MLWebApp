from flask import Flask, request, jsonify
from sklearn import svm
from sklearn import datasets
from sklearn.externals import joblib
from classifier import *

# declare constants
HOST = '0.0.0.0'
PORT = 8081

# initialize flask application
app = Flask(__name__)


@app.route('/api/train', methods=['POST'])
def train():
    # get parameters from request
    parameters = request.get_json()

    # read iris data set
    #iris = datasets.load_iris()
    #X, y = iris.data, iris.target

    # fit model
    #clf = svm.SVC(C=float(parameters['C']),
    #             probability=True,
    #             random_state=1)
    #clf.fit(X, y)
    adaboost, vec, selector, trainingError, validationError = training()
    # persist model
    joblib.dump(adaboost, 'model.pkl')

    #return jsonify({'accuracy': round(clf.score(X, y) * 100, 2)})
    return jsonify({'accuracy': round(100-trainingError)})


@app.route('/api/predict', methods=['POST'])
def predict():
    # get iris object from request
    X = request.get_json()
    X = str(X['text'])

    # read model
    adaboost = joblib.load('model.pkl')
    probabilities = adaboost.predict(X)

    return jsonify([{'name': 'Iris-Setosa', 'value': 12},
                    {'name': 'Iris-Versicolour', 'value': 20},
                    {'name': 'Iris-Virginica', 'value': 15}])


if __name__ == '__main__':
    # run web server
    app.run(host=HOST,
            debug=True,  # automatic reloading enabled
            port=PORT)

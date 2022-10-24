from secrets import API_TOKEN, API_TOKEN_SANDBOX
from dataprep import PriceData
from datetime import datetime, timedelta
import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from iexfinance.stocks import get_historical_data

## SANDBOX TESTING
import os
os.environ['IEX_API_VERSION'] = 'iexcloud-sandbox'

class Predict:

    def __init__(self, intv, depth, classifier):

        self.intv = intv
        self.depth = depth
        self.classifier = classifier

    def get_price(date, ticker):

        try:
            return get_historical_data(ticker, date, close_only=True, token = PriceData.token)['close'].values[0]
        except IndexError:
            date = datetime.strptime(date, '%Y-%m-%d')
            date += -timedelta(days = 1)
            date = date.strftime('%Y-%m-%d')
            return Predict.get_price(date, ticker)

    def predict(self, ticker):

        priceList = []
        i = self.depth - 1
        while(i >= 0):
            date = datetime.today() - timedelta(days = i*self.intv)
            date = date.strftime('%Y-%m-%d')
            price = Predict.get_price(date, ticker)
            priceList += [price]
            i += -1

        rates = [
            priceList[i+1]/priceList[i] - 1 for i in range(len(priceList) - 1)
        ]
        rates = np.array([rates])
        ratesNormalised = StandardScaler().fit_transform(rates)

        return self.classifier.predict(ratesNormalised)[0]

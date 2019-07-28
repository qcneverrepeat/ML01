# linear model/elastic net/logit/LDA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class logit(object):

    def __init__(self):
        self.coef = None
        self.COST = []
        self.prob = None
        self.result = None

    def fit(self,x,y,alpha = 0.01,iter = 1000):
        '''
        x: shape(m,p)
        y: shape(m,)
        '''

        w = np.array([0.0]*x.shape[1]).T
        b = 0.0
        dw = np.array([0.0]*x.shape[1]).T
        db = 0.0

        for i in range(iter):
            y_ = 1/(1 + np.e**-(x.dot(w) + b))

            # if 0 in y_ or 1 in y_:
            #     print('warning')

            dw = np.dot(x.T, (y_ - y))/x.shape[0]
            db = np.sum(y_ - y)/x.shape[0]
            w -= alpha * dw
            b -= alpha * db
            cost = (-1/x.shape[0]) * sum(y*np.log(y_) + (1-y)*np.log(1-y_)) # attention that when y_ = 0,1
            self.COST.append(cost)

        self.prob = y_
        Y_ = pd.Series(list(map(lambda x: round(x,4), y_)))
        Y = pd.Series(y)
        self.result = pd.concat([Y,Y_],axis = 1)

    def predict(self,x,thres = 0.5):
        pass

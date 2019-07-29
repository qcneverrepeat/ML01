# linear model/elastic net/logit/LDA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class logit(object):

    def __init__(self):
        self.coef = None # np.array([w,b])
        self.COST = [] # record the training process by cost-function
        self.result = None # fitting result on the training set, containing 'Origin' label and fitting 'Prob'

    def fit(self,x,y,alpha = 0.01,iter = 1000):
        '''
        x: shape(m,p), input as np.array or pd.DataFrame
        y: shape(m,), input as np.array or pd.Series
        '''
        x = np.array(x)
        y = np.array(y)

        w = np.array([0.0]*x.shape[1]).T
        b = 0.0
        dw = np.array([0.0]*x.shape[1]).T
        db = 0.0

        for i in range(iter):
            y_ = 1/(1 + np.e**-(x.dot(w) + b))
            if 0 in y_ or 1 in y_:
                print('Warning!')
            dw = np.dot(x.T, (y_ - y))/x.shape[0]
            db = np.sum(y_ - y)/x.shape[0]
            w -= alpha * dw
            b -= alpha * db
            cost = (-1/x.shape[0]) * sum(y*np.log(y_) + (1-y)*np.log(1-y_)) # attention that when y_ = 0,1
            self.COST.append(cost)

        self.coef = np.append(w,b)
        self.result = pd.concat([pd.Series(y,name='Origin'), pd.Series(list(map(lambda x: round(x,4), y_)),name='Fit')],axis = 1)

    def trainProc(self):
        '''show the cost-function by plot'''
        plt.plot(self.COST)
        plt.title('Cost-function')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.show()

    def predict(self,x,thres = 0.5):
        '''
        input as np.array or pd.DataFrame
        output a DataFrame including predicting 'Prob' and 'Label' under a threshold
        '''
        y = 1/(1 + np.e**-(np.array(x).dot(self.coef[:-1]) + self.coef[-1]))
        frame = pd.Series(y, name = 'Prob')
        y[y>=thres] = 1
        y[y<thres] = 0
        frame = pd.concat([frame, pd.Series(y, name='Label')], axis = 1)
        return frame

# linear model/ridge/logit/LDA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class logitRegression(object):
    '''
    logistic regression with ridge regularization
    '''

    def __init__(self):
        self.coef = None # np.array([w,b])
        self.COST = [] # record the training process by cost-function
        self.result = None # fitting result on the training set, containing 'Origin' label and fitting 'Prob'

    def fit(self,x,y,alpha = 0.01,iter = 1000,lam=0):
        '''
        x: shape(m,p), input as np.array or pd.DataFrame
        y: shape(m,), input as np.array or pd.Series

        regularization : Ridge
        '''
        x = np.array(x)
        y = np.array(y)
        x = np.insert(x, 0, values=1, axis=1) # 首列插1合并w和b

        w = np.array([0.0]*x.shape[1]).T
        dw = np.array([0.0]*x.shape[1]).T

        for i in range(iter):
            y_ = 1/(1 + np.e**-(x.dot(w)))
            if 0 in y_ or 1 in y_:
                print('Warning!')
            dw = (x.T.dot(y_-y) + lam*w)/x.shape[0]
            w -= alpha * dw
            cost = (-1/x.shape[0]) * (sum(y*np.log(y_) + (1-y)*np.log(1-y_)) + lam/2 * sum(w**2)) # attention that when y_ = 0,1
            self.COST.append(cost)

        self.coef = w
        self.result = pd.concat([pd.Series(y,name='Origin'), pd.Series(list(map(lambda x: round(x,4), 1/(1 + np.e**-(x.dot(w))))),name='Fitting')],axis = 1)

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
        x = np.insert(x, 0, values=1, axis=1)
        y = 1/(1 + np.e**-(np.array(x).dot(self.coef)))
        frame = pd.Series(y, name = 'Prob')
        y[y>=thres] = 1
        y[y<thres] = 0
        frame = pd.concat([frame, pd.Series(y, name='Label')], axis = 1)
        return frame

class linearRegression(object):
    '''
    linear regression with ridge regularization
    '''

    def __init__(self):
        self.coef = None # np.array([w,b])
        self.COST = [] # record the training process by cost-function
        self.result = None # fitting result on the training set, containing 'Origin' and 'Fitting' value

    def fit(self,x,y,method=1,alpha=0.1,iter=1000,lam=0):
        '''
        method 1: GD (default)
        method 2: Normal Equation
        if #att. is too large, recommend to use GD

        regularization : Ridge
        '''
        x = np.array(x)
        y = np.array(y)
        x = np.insert(x, 0, values=1, axis=1) # 首列插1合并w和b
        print('Det(X.T*X) is',np.linalg.det(x.T.dot(x))) # 显示X.T*X行列式，若太小则考虑使用正则项

        if abs(np.linalg.det(x.T.dot(x))) < 0.1: # 检查X.T*X是否近似为奇异矩阵
            jud = input('X.T*X is roughly singular matirx: whether to use ridge regression ? (Y/N)')
            if jud == 'Y':
                lam = float(input('input your lambda : ')) # input接受为str
            elif jud == 'N':
                print('Must use method 1...')
                method = 1

        if method == 1:

            w = np.array([0.0]*x.shape[1]).T
            dw = np.array([0.0]*x.shape[1]).T

            for i in range(iter):
                y_ = x.dot(w)
                dw = (x.T.dot(y_-y) + lam*w)/x.shape[0]
                w -= alpha * dw
                cost = (0.5/x.shape[0]) * (sum((y_-y)**2) + lam*sum(w**2))
                self.COST.append(cost)
            self.coef = w

        elif method == 2:

            w = np.matrix(x.T.dot(x)+lam*np.identity(x.shape[1])).I.dot(x.T).dot(y).T
            self.coef = np.array(w).reshape((x.shape[1],)) # (n,) is different from (n,1), use reshape

        self.result = pd.concat([pd.Series(y,name='Origin'), pd.Series(list(map(lambda x: round(x,4), x.dot(self.coef))),name='Fitting')],axis = 1)

    def trainProc(self):
        '''show the cost-function by plot'''
        plt.plot(self.COST)
        plt.title('Cost-function')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.show()

    def predict(self,x):
        x = np.insert(x, 0, values=1, axis=1)
        y = np.array(x).dot(self.coef)
        frame = pd.Series(y, name = 'Predict')
        return frame

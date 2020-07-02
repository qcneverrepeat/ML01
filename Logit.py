'''
@Description: logit & gradient descent
@Version: 
@Autor: qc
@Date: 2020-01-09 21:04:09
@LastEditors  : qc
@LastEditTime : 2020-01-10 13:52:55
'''

import numpy as np

class logit(object):
    '''
    @description: multi-class logistic model
    @param {lr: float} {max_iter: int} 
    @return: logit object
    '''    
    def __init__(self, penalty='None', lr=0.001, epoch=20):
        self.penalty = penalty
        self.lr = lr
        self.epoch = epoch
        self.w = None
        self.b = None

    def fit(self, x, y):
        '''
        @description: gradient descent with lr and max_iter
        @param {x: np.array (obs, atts)} {y: np.array (obs, class_num) must be one_hot} 
        @return: self.w, self.b
        '''
        # randomly initialize parameters in standard Gaussian distribution N(0,1)
        self.w = np.random.randn(x.shape[1], y.shape[1]) 
        self.b = np.random.randn(y.shape[1])
        # self.w = np.zeros([x.shape[1], y.shape[1]])
        # self.b = np.zeros([y.shape[1]])

        for epoch in range(self.epoch):
            z = x.dot(self.w)+self.b
            output = (np.exp(z).T/(np.exp(z).sum(axis=1))).T
            grad = output - y 
            for i in range(y.shape[1]):
                self.b[i] -= grad[:,i].sum() * self.lr     
                self.w[:,i] -= (grad[:,i]*(x.T)).sum(axis=1) * self.lr
                print('Loss in %d epoch:'%epoch, self.cross_entropy(y, output))

    def cross_entropy(self, label, pred):
        # input should be 2 distributions
        ce = 0.0
        for row in range(label.shape[0]):
            ce += -(label[row,:]*np.log(pred[row,:])).sum()
        return ce

    def predict(self, x, type='score'):
        # type == 'class' / 'score'
        z = x.dot(self.w)+self.b
        y_out = (np.exp(z).T/(np.exp(z).sum(axis=1))).T
        if type == 'class': return np.argmax(y_out, axis=1)    
        return y_out
    


    
if __name__ == '__main__':
    from sklearn import datasets
    from keras.utils import to_categorical
    iris = datasets.load_iris()
    x = iris['data']
    y = iris['target']
    x_st = (x-x.mean(axis=0))/(x.std(axis=0)) 
    # learning rate for differents features are the same, so normalization is necessary
    # Indeed, Z-score normalization make the acc 66% to 95% 
    y_ca = to_categorical(y, num_classes=3) 

    md = logit(epoch=300)
    md.fit(x_st, y_ca)
    y_out = md.predict(x_st, type='class') 
    print('Acc:', (y_out == y).sum()/y.size) # around 96%
# coding: utf-8
'''
@author: qc
2019746
'''

'''
class sklearn.ensemble.AdaBoostClassifier(base_estimator=None, n_estimators=50,
learning_rate=1.0, algorithm=’SAMME.R’, random_state=None)
'''

import pandas as pd
import numpy as np
import decisionTree as DT

class adaBoost(object):
    '''adaptive boosting'''

    def __init__(self, base = None, n_estimators = 50):

        self.base = base
        self.n_estimators = n_estimators

        self.base_set = []
        self.base_weight_set = []

    def fit(self, x, y):

        sample_weight_set = pd.DataFrame([1/x.shape[0]] * x.shape[0], index = x.index)

        for i in range(self.n_estimators):

            base = self.base
            if self.base == None:
                base = DT.tree(max_depth = 1)
            base.fit(x, y, sample_weight = sample_weight_set) # train base model with sample weight
            self.base_set.append(base)

            error = sum(sample_weight_set[base.predict(x) == y][0]) # e t
            if error > 0.5:
                print('warning: the training process is stopped.')
                break
            base_weight = (1/2) * np.log((1-error)/error) # alpha t
            sample_weight_set[base.predict(x) == y][0]] = sample_weight_set[base.predict(x) == y][0]] * np.exp(- base_weight)
            sample_weight_set[base.predict(x) != y][0]] = sample_weight_set[base.predict(x) != y][0]] * np.exp(+ base_weight)
            sample_weight_set = sample_weight_set/sum(sample_weight_set[0])
            self.base_weight_set.append(base_weight)

    def predict(self, x):
        pass

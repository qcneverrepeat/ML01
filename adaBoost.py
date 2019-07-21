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
import random

class adaBoost(object):
    '''adaptive boosting'''

    def __init__(self, base = None, n_estimators = 50):

        self.base = base
        self.n_estimators = n_estimators

        self.base_set = []
        self.base_weight_set = []

    def fit(self, x, y):

        sample_weight_set = pd.DataFrame([1/y.size] * y.size)

        for i in range(self.n_estimators):
            base = self.base
            if self.base == None:
                base = DT.Tree(max_depth = 1)
            try:
                base.fit(x, y, sample_weight = sample_weight_set) # train base model with sample weight
            except TypeError: # if base model does not support training with sample_weight, then resampling
                # resampling
                resam_index = self.Roulette(sample_weight_set) # resam_index is coming from sample_weight_set.index
                resam_x = x.iloc[resam_index,:]
                resam_y = y.iloc[resam_index]
                sample_weight_set = sample_weight_set.iloc[resam_index]

                resam_x.index = range(y.size)
                resam_y.index = range(y.size)
                sample_weight_set.index = range(y.size)
                base.fit(resam_x, resam_y)
            self.base_set.append(base)

            # base weight calculating
            predict = base.predict(resam_x)
            error = sum(sample_weight_set.loc[predict != resam_y][0]) # e t
            # error cannot calculate error after resample the sample weight...






            # if error > 0.5:
            print(i,'error is %f'%error)
                # break
            base_weight = (1/2) * np.log((1-error)/error) # alpha t
            self.base_weight_set.append(base_weight)

            # sample weight updating
            # using loc can get rid of SettingWithCopy Warning
            sample_weight_set.loc[predict == resam_y] = sample_weight_set.loc[predict == resam_y] * np.exp(- base_weight)
            sample_weight_set.loc[predict != resam_y] = sample_weight_set.loc[predict != resam_y] * np.exp(+ base_weight)
            sample_weight_set = sample_weight_set/sum(sample_weight_set[0])


    def Roulette(self, weight):
        """
        input: weight_set in DataFrame
        output: resample index in list
        """
        weight = list(weight.iloc[:,0])
        accm_weight = []
        accm_for_sum = []
        for val in weight:
            accm_for_sum.append(val)
            accm_weight.append(sum(accm_for_sum))
        sumFit = sum(weight)
        resam_index = []
        for i in range(len(weight)):
            rand = random.uniform(0, sumFit)
            for ind, val in enumerate(accm_weight):
                if val >= rand:
                    resam_index.append(ind)
                    break
        return resam_index

    def predict(self, x):
        pass

# coding:utf-8

'''
class sklearn.ensemble.RandomForestClassifier(n_estimators=’warn’, criterion=’gini’,
max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0,
warm_start=False, class_weight=None)
'''

import decisionTree as DT
from collections import Counter
import numpy as np
import pandas as pd

class randomForest(object):
    '''base = 'ID3', 'C45', 'CART' '''
    '''max_features = 'log2', 'sqrt', int or float'''

    def __init__(self, n_estimators = 10, base = 'ID3',
                max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                max_features='log2', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0,
                warm_start=False, class_weight=None):

        # parameters check ...

        self.n_estimators = n_estimators
        self.base = base
        self.max_features = max_features
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.random_state = random_state
        # other parameters ...
        self.forest = []


    def fit(self,x,y):
        # generating trees to self.forest
        for i in range(self.n_estimators):

            tree = DT.Tree(tree_type = self.base,
                            max_depth = self.max_depth,
                            max_features = self.max_features,
                            random_state = self.random_state) # new a tree object

            if self.bootstrap:
                x_sampled = x.sample(n = x.shape[0], random_state = self.random_state, replace = True)
                y_sampled = y[x_sampled.index]
                x_sampled.index = range(x.shape[0])
                y_sampled.index = range(x.shape[0]) # resolve the index, or the Gain() will be broken
            else:
                x_sampled = x
                y_sampled = y

            tree.fit(x_sampled,y_sampled)
            self.forest.append(tree)

    def predict(self, x, show = 'ensemble'):
        '''
        show = 'ensemble' or 'all'
        '''
        result_frame = pd.DataFrame()
        for tree in self.forest:
            result = tree.predict(x)
            result_frame = pd.concat([result_frame, result], axis = 1)
        ensemble = []
        for index,row in result_frame.iterrows():
            ensemble.append(list(Counter(row).keys())[0])
        result_frame['ensemble'] = ensemble
        if show == 'all':
            return result_frame
        return pd.Series(ensemble, index = x.index)

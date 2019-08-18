# kmeans cluster

import numpy as np
import pandas as pd
import copy

class kmeans(object):

    def __init__(self, k):
        self.k = k

    def fit(self, x):
        '''
        input x: a np.array/matrix or pd.DataFrame, each element must be number (int/float/...)
        output : a new column "cluster" appended to x
        '''

        # randomly initialization: the cluster label is represented by center.index
        x = pd.DataFrame(x)
        center = x.sample(n=self.k, replace=False, axis=0).reset_index(drop=True)

        # Iteration
        while 1:
            # update cluster label for each point
            for index,row in x.iterrows(): # int,Series
                dis = ((row-center)**2).apply(lambda x: x.sum(),axis=1) # series: distance (row - each point in center)
                label = dis.sort_values().index[0]
                x.loc[index,'cluster'] = label # choose max distance index as the cluster label
            # update center
            pre_center = copy.deepcopy(center)
            for clu,group in x.groupby('cluster'):
                center.loc[clu] = group.mean()
            # convergence checking : when the center keeps still
            if (center.values == pre_center.values).all():
                break

            print('iterating...')

        return x # return a new DataFrame instead of modifying the original x

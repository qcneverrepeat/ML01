'''
kmeans cluster
issues:
    - Optimization objcet is minimizing the distortion/cost function (1/m)Sum||x-u||^2
    - Different initializations will get different cluster results (more seriously when k is small), sometimes gets local optimal
    - Repeat clustering for 50~1000 times to choose the lowest distortion
    - Choose k value usually use Elbow method (curve : distortion/other business metric -- k value)

    - Initialize randomly from R^n instead of D will probably cause cluster disappearing (the cluster cannot get a single point)
    - Initialize from D cause cluster disappearing ?

'''

import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt

class kmeans(object):

    def __init__(self):
        pass

    def fit(self, x, k = 3, repeat = 1):
        '''
        input x: a np.array/matrix or pd.DataFrame, each element must be number (int/float/...)
        output : a new column "cluster" appended to x
        '''

        for i in range(repeat):

            # randomly initialization: the cluster label is represented by center.index
            x_ = pd.DataFrame(x)
            center = x_.sample(n=k, replace=False, axis=0).reset_index(drop=True)

            # Iteration
            while 1:
                # update cluster label for each point
                for index,row in x_.iterrows(): # int,Series
                    dis = ((row-center)**2).sum(axis=1) # series: distance (row - each point in center)
                    label = dis.sort_values().index[0]
                    x_.loc[index,'cluster'] = label # choose max distance index as the cluster label
                # update center
                pre_center = copy.deepcopy(center)
                for clu,group in x_.groupby('cluster'):
                    center.loc[clu] = group.mean()
                # convergence checking : when the center keeps still
                if (center.values == pre_center.values).all():
                    break

            try:
                pre_J = J
            except:
                pre_J = float("inf")

            J = 0.0
            for clu,group in x_.groupby('cluster'):
                J += float(((group.iloc[:,:x_.shape[1]-1] - center.loc[clu])**2).sum(axis=1).sum()*(1/x_.shape[0]))
                # 如果不加float会报float+str的错误
            print('Distortion is',J)

            if J < pre_J:
                result = copy.deepcopy(x_)

        print('---- K = %d finished. ----'%k)

        return result,J # return a new DataFrame instead of modifying the original x

    def chooseK(self, x, repeat = 5, mink = 2, maxk = 10):
        '''
        Elbow method: draw the cruve (Distortion-K)
        Better way is to evaluating by (A business metric-K)

        return a curve : distortion -- k
        and the distortion vector
        '''
        Distortion = []
        for k in range(mink, maxk+1):
            Distortion.append(self.fit(x,repeat=repeat,k=k)[1])

        plt.plot(range(mink,maxk+1),Distortion)
        plt.xlabel('Cluster')
        plt.ylabel('Distortion')
        plt.show()

        return Distortion

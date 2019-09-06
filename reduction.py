# coding: utf-8

import pandas as pd
import numpy as np
from numpy import linalg

class PCA(object):

    def __init__(self, k = None):
        self.k = k

    def fit(self,x):
        '''x input as np.matrix or DataFrame'''

        x = np.matrix(x)
        x = np.apply_along_axis(lambda a: a-a.mean(),0,x) # 中心化

        lam,vec = linalg.eig(x.T.dot(x)) # 特征值分解
        # 特征向量 lam is [lam1,lam2,...]
        # 特征矩阵 vec is [eigvec1,eigvec2,...] 已单位正交化

        t = lam.cumsum(0)/sum(lam)
        print(list(map(lambda a: '%.2f%%'%(a*100),t))) # 显示方差(X.T*X特征值)累计百分比

        if self.k == None: # 若K缺失则可后续键入
            self.k = int(input('K='))

        y = x.dot(vec[:,:self.k])
        return y

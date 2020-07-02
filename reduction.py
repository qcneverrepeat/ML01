'''
@Description: 
@Version: 
@Autor: qc
@Date: 2019-08-20 21:57:46
@LastEditors  : qc
@LastEditTime : 2020-01-10 15:32:59
'''
# coding: utf-8

import numpy as np
from numpy import linalg

class PCA(object):

    def __init__(self, k = None):
        self.k = k
        self.lam = None 
        self.vec = None

    def reduce(self,x):
        '''x input as 2D np.array'''

        x = np.apply_along_axis(lambda a: a-a.mean(),0,x) # 中心化

        self.lam,self.vec = linalg.eig(x.T.dot(x)) # 特征值分解
        self.vec = np.real(self.vec) # 特征向量提取实部
        # 特征向量 self.lam is [lam1,lam2,...]
        # 特征矩阵 self.vec is [eigvec1,eigvec2,...] 已单位正交化

        t = self.lam.cumsum(0)/sum(self.lam)
        # print(list(map(lambda a: '%.2f%%'%(a*100),t))) # 显示方差(X.T*X特征值)累计百分比

        y = x.dot(self.vec[:,:self.k])
        return y

    def inverse(self, x):
        '''inverse transform to original shape'''
        y = x.dot(self.vec[:,:self.k].T) # np.matrix
        return np.array(y)



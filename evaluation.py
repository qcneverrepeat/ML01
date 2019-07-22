# coding:utf-8
'''
K-S
AUC
Lift
P-R
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

class evaluator(object):

    def __init__():
        pass

    @classmethod
    def ROC(cls, label, predict, pos_label, method = 1):
        '''
        input: label, predict score, both in series or DataFrame
        output: ROC curve
        method = 1: one-by-one change to positive
        method = 2: change threshold
        '''
        frame = pd.concat([label,predict],axis=1).sort_values(by=1)

        # identify the positive & negative label
        if pos_label == None:
            pos_label = list(Counter(frame[0].tail()).keys())[0]
        label_set = list(frame[0].drop_duplicates())
        label_set.remove(pos_label)
        neg_label = label_set[0]

        P = Counter(frame[0])[pos_label]
        N = Counter(frame[0])[neg_label]

        # method 2: change threshold
        if method == 2:
            #...
            pass

        # method 1: one-by-one change to positive
        TPRate = [0]
        FPRate = [0]
        frame['pre'] = neg_label
        for i in range(label.size):
            frame.iloc[i,2] = pos_label
            TPR = sum(frame[frame.iloc[:,0] == frame.iloc[:,2]].iloc[:,0] == pos_label)/P
            FPR = 1 - sum(frame[frame.iloc[:,0] == frame.iloc[:,2]].iloc[:,0] == neg_label)/N
            TPRate.append(TPR)
            FPRate.append(FPR)

        # calculating AUC value
        FPRate.append(0)
        TPRate.append(0)
        de_FPR = np.array(FPRate[1:])-np.array(FPRate[:-1])
        AUC = np.array(TPRate[1:]).dot(de_FPR.T)
        FPRate.pop()
        TPRate.pop()

        # draw ROC plot
        plt.plot(FPRate,TPRate)
        plt.title('ROC')
        plt.xlabel('FPRate')
        plt.ylabel('TPRate')
        plt.text(0.7,0.3,'AUC value: %0.3f'%AUC)
        plt.show()

        return AUC


    @classmethod
    def PR_cruve(cls, label, predict):
        pass

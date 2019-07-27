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
from scipy import stats
from scipy.integrate import quad,dblquad,nquad

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
        # if not given, choose the majority of labels with 5 highest score as pos_label
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
    def PR(cls, label, predict, pos_label, method = 2):
        '''
        input: label, predict score, both in series or DataFrame
        output: P-R curve
        method = 1: one-by-one change to positive
        method = 2: change threshold
        '''
        frame = pd.concat([label,predict],axis=1).sort_values(by=1)

        # identify the positive & negative label
        # if not given, choose the majority of labels with 5 highest score as pos_label
        if pos_label == None:
            pos_label = list(Counter(frame[0].tail()).keys())[0]
        label_set = list(frame[0].drop_duplicates())
        label_set.remove(pos_label)
        neg_label = label_set[0]

        P = Counter(frame[0])[pos_label]

        # less than ROC by 1 point
        Precision = []
        Recall = []
        frame['pre'] = neg_label
        # print(frame)

        # method 1: one-by-one change to positive
        if method == 1:
            for i in range(label.size):
                frame.iloc[i,2] = pos_label
                TP = sum(frame[frame.iloc[:,0] == frame.iloc[:,2]].iloc[:,0] == pos_label)
                recall = TP/P
                precision = TP/(i+1)
                Recall.append(recall)
                Precision.append(precision)

        # method 2: group-by-group change = change the threshold
        else:
            i = 0
            while i <= label.size-1:
                if i == label.size-1:
                    frame.iloc[i,2] = pos_label
                    TP = sum(frame[frame.iloc[:,0] == frame.iloc[:,2]].iloc[:,0] == pos_label)
                    precision = TP/(i+1)
                    recall = TP/P
                    Recall.append(recall)
                    Precision.append(precision)
                    # print('2')
                    break
                while frame.iloc[i,1] == frame.iloc[i+1,1]:
                    frame.iloc[i,2] = pos_label
                    i += 1
                    # print('3')
                    if i == label.size-1:
                        break
                frame.iloc[i,2] = pos_label
                TP = sum(frame[frame.iloc[:,0] == frame.iloc[:,2]].iloc[:,0] == pos_label)
                precision = TP/(i+1)
                recall = TP/P
                Recall.append(recall)
                Precision.append(precision)
                # print('1')
                i += 1

        # draw P-R plot
        plt.plot(Recall, Precision)
        plt.title('P-R')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # plt.text(0.7,0.3,'AUC value: %0.3f'%AUC)
        plt.show()

    @classmethod
    def EMP(cls, label, predict, pos_label, CLV = 200, contact = 1, incentive = 10, alpha = 6, beta = 14, accept = 0.5, expect = True):
        '''
        expect = True : EMP
        expect = False : MP, accept probability = 0.5 in default
        '''
        frame = pd.concat([label,predict],axis=1).sort_values(by=1)
        if pos_label == None:
            pos_label = list(Counter(frame[0].tail()).keys())[0]
        label_set = list(frame[0].drop_duplicates())
        label_set.remove(pos_label)
        neg_label = label_set[0]

        P = Counter(frame[0])[pos_label]
        P_pre = Counter(frame[1])[pos_label]

        frame['pre'] = neg_label

        def MP(acc = accept):
            MP = 0
            for i in range(label.size):
                frame.iloc[i,2] = pos_label
                TP = sum(frame[frame.iloc[:,0] == frame.iloc[:,2]].iloc[:,0] == pos_label)
                MP = max(TP*(CLV*acc + incentive*(1-acc)) - P_pre*(contact + incentive), MP)
            return MP

        if not expect:
            return MP(accept)
        else:
            EMP = quad(lambda acce: stats.beta(alpha, beta).pdf(acce)*MP(acce),0,1)
            return round(EMP[0])

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

    def __init__(self):
        pass

    @classmethod
    def ROC(label, predict, pos_label, method = 1):
        '''
        input: label, predict score, both in series or array
        output: show ROC curve, return AUC value

        method = 1: one-by-one change to positive, from larger score to little
        method = 2: group-by-group change to positive, from larger score to little (some predict scores are the same)

        only when the score order is true (i.e., label sorted as ---...---+++...+++ without chaos), AUC=1
        <=> there is a threshold under which the accuracy = 100%

        '''

        # 0-1 normalization; sorting by column predict
        frame = pd.DataFrame({'label':label, 'predict':predict})
        frame['predict'] = (frame['predict']-frame['predict'].min())/np.ptp(frame['predict'])
        frame = frame.sort_values(by='predict', ascending=False)
        frame.index = range(1,frame.shape[0]+1)

        # identify the positive & negative label
        # if not given, choose the majority of labels with 5 highest score as pos_label
        if pos_label == None:
            pos_label = list(Counter(frame[0].tail()).keys())[0]
        label_set = list(frame['label'].drop_duplicates())
        label_set.remove(pos_label)
        neg_label = label_set[0]

        P = Counter(frame['label'])[pos_label]
        N = Counter(frame['label'])[neg_label]

        frame['temp'] = neg_label
        TPRate = []
        FPRate = []

        # method 1: one-by-one change to positive,阶梯型
        if method == 1:
            TPRate.append(0)
            FPRate.append(0)
            for i in range(label.size):
                frame.iloc[i,2] = pos_label
                TPR = sum(frame[frame.iloc[:,0] == frame.iloc[:,2]].iloc[:,0] == pos_label)/P
                FPR = 1 - sum(frame[frame.iloc[:,0] == frame.iloc[:,2]].iloc[:,0] == neg_label)/N
                TPRate.append(TPR)
                FPRate.append(FPR)

        # method 2: group-by-group change to positive (some predict scores are the same)，可能出现梯形
        # 【推荐】绘图结果与sklearn等价,但sklearn速度更快优化更好，能在保持图像不变的前提下省略部分绘图点
        elif method == 2:
            TPRate.append(0)
            FPRate.append(0)
            for i in range(label.size):
                frame.iloc[i,2] = pos_label
                try:
                    if frame['predict'].iloc[i] == frame['predict'].iloc[i+1]:
                        continue
                except:
                    pass
                TPR = sum(frame[frame.iloc[:,0] == frame.iloc[:,2]].iloc[:,0] == pos_label)/P
                FPR = 1 - sum(frame[frame.iloc[:,0] == frame.iloc[:,2]].iloc[:,0] == neg_label)/N
                TPRate.append(TPR)
                FPRate.append(FPR)

        # calculating AUC value
        # score完全不同时结果与法2等价
        # score有相同值时，此方法相当于将梯形部分向上填充成阶梯型，通常比法2略大
        '''
        FPRate.append(0)
        TPRate.append(0)
        de_FPR = np.array(FPRate[1:])-np.array(FPRate[:-1]) # 错位相减
        AUC = np.array(TPRate[1:]).dot(de_FPR.T)
        FPRate.pop()
        TPRate.pop()
        '''
        # another method，与sklearn.metric中等价
        # score有相同值时，此方法相当于给相同score的样本赋予了顺序，将梯形随机转化成了阶梯型，通常比法1略小
        AUC = 1 - (sum(frame[frame['label'] == pos_label].index) - 0.5*P*(P+1))/(N*P)

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
        frame = pd.concat([label,predict],axis=1).sort_values(by=1, ascending=False)

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

    @classmethod
    def Lift(cls, label, predict, pos_label):
        pass

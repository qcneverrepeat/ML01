# coding:utf-8
'''
@autor : qc
@version : 0.1

ID3 FINISHED

TO DO LIST:
    if (all x_i the same) then return leaf or split : return leaf
    if trainset attribute value is not complete, then how to handle the empty branch
    how to visualize : contents
    predict method in tree
    C4.5 & CART
    pruning
    other control parameters in tree
'''

'''
class sklearn.tree.DecisionTreeClassifier(criterion=’gini’, splitter=’best’, max_depth=None,
min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
min_impurity_split=None, class_weight=None, presort=False)
'''

from collections import Counter
import numpy as np
import pandas as pd
import random

class Node(object):

    '''decision tree node'''

    def __init__(self, x, y, deep, max_deep, max_features, random_state):

        self.type = 'node'
        self.entroy = self.__entroy(y)
        self.samples = y.size
        self.label = None
        self.judge = 'Default'
        self.childset = []
        self.max_deep = max_deep
        self.deep = deep
        self.max_features = max_features
        self.random_state = random_state
        self.x = x
        self.y = y

    def split(self):

        # max_features : random feature selection
        feature_number = len(self.x.columns)

        if self.max_features == 'log2':
            feature_number = int(round(np.log2(max(len(self.x.columns),1))))
        elif self.max_features == 'sqrt':
            feature_number = int(np.sqrt(len(self.x.columns)))
        elif isinstance(self.max_features,int) and self.max_features < len(self.x.columns):
            feature_number = self.max_features
        elif isinstance(self.max_features,float):
            feature_number = int(max_features * len(self.x.columns))

        if len(Counter(self.y).keys()) == 1 or self.x.shape[1] == 0 or self.x.drop_duplicates().shape[0] == 1 or self.deep == self.max_deep or feature_number == 0:
            # conditions to return leaf
            # y has only one label ; attributes set is empty ; the value in every attributes of all samples are the same
            # max_deep ; max_feature
            self.type = 'leaf'
            self.samples = self.y.size
            self.label = list(Counter(self.y).keys())[0]
            return

        frame = pd.DataFrame(self.__Gain(self.x,self.y)).sample(n = feature_number,random_state = self.random_state)
        best_index = frame.index[list(frame[0]).index(max(list(frame[0])))]

        # generate subnodes
        for key in Counter(self.x[self.x.columns[best_index]]).keys():

            subset_x = self.x[self.x[self.x.columns[best_index]] == key]
            del subset_x[subset_x.columns[best_index]] # finally, subset_x will be a empty dataframe with only index ,in shape of (n,0)
            subset_y = self.y[subset_x.index]

            subnode = Node(subset_x,subset_y,
                            deep = self.deep + 1,max_features = self.max_features,
                            max_deep = self.max_deep,random_state = self.random_state)

            # subnode.judge = '%s = %s'%(self.x.columns[best_index], key)
            subnode.judge = {self.x.columns[best_index]: key}
            subnode.label = list(Counter(self.y).keys())[0]
            self.childset.append(subnode)
            subnode.split()

    def __Gain(self,x,y):
        gain = []
        entroy = 0
        if y.size != 0:
            for var in x.columns:
                for key,value in Counter(x[var]).items():
                    sub_entroy = self.__entroy(y[x[var]==key])
                    entroy += sub_entroy*(value/y.size)
                gain.append(entroy)
        return gain

    def __entroy(self,y):
        '''entroy value ~ [0,1]'''
        proportion = list(Counter(y).values())
        entroy = 0
        for i in proportion:
            entroy += -(i/sum(proportion))*np.log2(i/sum(proportion))
        return entroy

    def show(self):
        print('     '*self.deep,'|type:',self.type)
        print('     '*self.deep,'|entroy: %0.4f'%self.entroy)
        print('     '*self.deep,'|samples:',self.samples)
        print('     '*self.deep,'|label:',self.label)
        print('     '*self.deep,'|judge:',self.judge)
        print('     '*self.deep,'|deep:',self.deep)
        print('     '*self.deep,'|'+'='*10)

# =======================================================================

class Tree(object):

    '''decision tree'''

    def __init__(self, class_weight=None, tree_type='ID3', max_depth=None, max_features=None,
            max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False,
            random_state=None, splitter='best'):

        # control parameters
        self.__class_weight = class_weight
        self.__type = tree_type
        self.__max_depth = max_depth
        self.__max_features = max_features
        self.__max_leaf_nodes = max_leaf_nodes
        self.__random_state = random_state
        #...
        self.__root = None

        # input check
        pass

    def __treeGenerate_ID3(self,x,y):
        self.__root = Node(x,y, deep = 0,
                                max_deep = self.__max_depth,
                                max_features = self.__max_features,
                                random_state = self.__random_state)
        self.__root.type = 'root'
        self.__root.label = list(Counter(y).keys())[0] # choose the max class in y-label or the only class
        self.__root.split()

    def __treeGenerate_C45(self,x,y):
        pass

    def __treeGenerate_CART(self,x,y):
        pass

    def fit(self,x,y):

        if x.shape[0] != y.shape[0] or y.shape[0] != y.size:
            print('data shape error')
        elif y.size == 0:
            print('no input')
        elif self.__type == 'ID3':
            self.__treeGenerate_ID3(x,y)
        elif self.__type == 'C4.5':
            self.__treeGenerate_C45(x,y)
        else:
            self.__treeGenerate_CART(x,y)

    def traversal(self):
        self.__root.show()
        self.traversal_recursive(self.__root)

    def traversal_recursive(self,node):
        '''recursive model'''
        for sub_node in node.childset:
            sub_node.show()
            self.traversal_recursive(sub_node)

    def predict(self, x):
        label = []
        for index,row in x.iterrows(): # iteration by row
            label.append(self.getLabel(row))
        return pd.Series(label, index = x.index)

    def getLabel(self, row):
        '''row : input as a pd.series'''
        flag = self.__root
        return self.getLabel_recursive(flag_input = flag, row_input = row)

    def getLabel_recursive(self, flag_input, row_input):
        '''recursive model'''
        for subnode in flag_input.childset:
            if list(subnode.judge.values())[0] == dict(row_input)[list(subnode.judge.keys())[0]]:
                flag = subnode
                break
        if flag.type == 'leaf':
            return flag.label
        return self.getLabel_recursive(flag, row_input) # do not forget the return

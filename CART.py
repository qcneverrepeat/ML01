'''
@Description: ID3,C4.5,CART
@Version: 
@Autor: qc
@Date: 2019-12-09 23:11:15
@LastEditors  : qc
@LastEditTime : 2019-12-23 00:36:30
@blog: 
- cannot handle missing value
- digital variables only, automatically convert to float32
- only use gini criterion in classification
- fit(x,y) x is numpy.array in 2d/pd.DataFrame, y is numpy.array in 1d/pd.Series/list
只实现核心功能代码，丢弃过多的容错、检查、结构集成机制
速度：适当考虑
内存：适当考虑

待写：
剪枝
与sklearn比较泛化性能
'''

import numpy as np
from collections import Counter

class Tree(object):
    '''
    @description: TREE class contains initialization, fit, predict, visualization
    @param {task: str, 'classification','regression', default = 'classification'; 
            max_features: int/float/str, 'sqrt','log2', default = None;
            max_depth: int, maximun depth of the tree, default = None;
            }
    @return: a CART TREE Instance

    '''
    def __init__(self,  task = 'classification',
                        max_depth = np.inf,
                        min_split = 1):
        self.root = None
        self.task = task        
        self.max_depth = max_depth
        self.min_split = min_split

        self.dot = '' # the DOT script of the tree
    
    def fit(self, x, y):
        '''
        @description: fit the tree from numpy.array
        @param {x: numpy.array in 2d/pd.DataFrame; y: numpy.array in 1d/pd.Series/list}
        ''' 
        # automatically convert the input to np.ndarray in float32
        self.x = np.array(x, dtype = np.float32)
        self.y = np.array(y, dtype = np.float32)
        self.root = Node(type='root',
                         feature_ind = np.array(list(range(self.x.shape[1]))), 
                         ins_ind = np.array(list(range(self.x.shape[0]))))
        self.nodeSplit(self.root)

    def nodeSplit(self, node):
        '''
        node with (depth, ins/fea_index)
        ->
        node with all other attributes
        in which right/left is still node with (depth, ins/fea_index)
        '''

        node.feature_num = node.feature_ind.size
        node.ins_num = node.ins_ind.size
        # print(node.feature_ind) debug

        if self.task == 'classification':
            ct = Counter(self.y[node.ins_ind])
            ct_val = [i for i in ct.values()]
            node.label = [i for i in ct.keys()][ct_val.index(max(ct_val))]
        else:
            node.label = self.y[node.ins_ind].mean()
        
        node.impurity = self.getGini(self.y[node.ins_ind])

        if (node.depth >= self.max_depth) or (node.ins_num <= self.min_split) or (node.feature_num == 0):
            '''
            return or split
            max_depth/min_split: stop splitting and return
            '''
            node.type = 'leaf'
            return 

        gini_set = []
        sep_set = []
        for i in node.feature_ind:
            split = self.bestSplit(var = self.x[node.ins_ind,i], label = self.y[node.ins_ind])
            gini_set.append(split[1])
            sep_set.append(split[0])
        best = gini_set.index(max(gini_set))
        node.sep_point = sep_set[best]
        node.feature_judge = node.feature_ind[best] # feature_judge is an index

        node.feature_ind = np.delete(node.feature_ind, best) # delete the feature_judge from node.feature_ind

        left  = (self.x[node.ins_ind, node.feature_judge] <= node.sep_point)
        right = (self.x[node.ins_ind, node.feature_judge] > node.sep_point)
        left_ins_ind  = node.ins_ind[left]
        right_ins_ind = node.ins_ind[right]

        if min(left_ins_ind.size, right_ins_ind.size)==0 or (node.impurity-max(gini_set))<=0:
            '''
            early stop: stop splitting and return
            condition:
                one subtree is emtpy
                impurity decrease <= threshold (0 here)
            '''
            node.type = 'leaf'
            return
            
        node.left  = Node(depth=node.depth+1, 
                          feature_ind=node.feature_ind, 
                          feature_judged=node.feature_judge,
                          sep_point_ed=node.sep_point,
                          ins_ind=left_ins_ind, 
                          belong=0)
        node.right = Node(depth=node.depth+1, 
                          feature_ind=node.feature_ind, 
                          feature_judged=node.feature_judge,
                          sep_point_ed=node.sep_point,
                          ins_ind=right_ins_ind, 
                          belong=1)  
        self.nodeSplit(node.left)
        self.nodeSplit(node.right)      

    def bestSplit(self, var, label):
        '''
        var & label: 1d array
        return best sep_point, corresponding gini = (d1/d) * gini1 + (d2/d) * gini2
        '''
        val_set = np.unique(var)
        gini_set = []
        for sep in val_set:
            set1 = label[var <= sep]
            set2 = label[var > sep]
            gini = set1.size/label.size*self.getGini(set1) + set2.size/label.size*self.getGini(set2)
            gini_set.append(gini)
        best_ind = gini_set.index(min(gini_set))
        return val_set[best_ind], gini_set[best_ind]
    
    def getGini(self, label):
        '''
        label: 1d array
            classification: gini impurity
            regression: variance
        '''
        if self.task == 'classification':
            freq = np.array([i for i in Counter(label).values()])
            result = 1-((freq/label.size)**2).sum() # 1 - sum(p**2)
        else:
            result = np.var(label)
        return result
        
    def predict(self, x):
        x = np.array(x, dtype = np.float32)
        if x.shape[1] != self.x.shape[1]:
            raise Exception('shape of prediction set ERROR')
        predict = np.zeros([x.shape[0],])
        for i in range(x.shape[0]):
            ins = x[i,:]
            flag = self.root
            while 1:
                if flag.type == 'leaf':
                    label = flag.label
                    break
                if ins[flag.feature_judge] <= flag.sep_point:
                    flag = flag.left
                else:
                    flag = flag.right
            predict[i] = label
        return predict        

    def tree2DOT(self, color = 'pink'):
        '''
        head + node's self context + node's edge context
        traversal all the node, print each node's self context and edge context
        '''
        self.dot += '''digraph graph1 
                       {node [shape=box, style="filled, rounded", 
                        color="%s", fontname="Microsoft YaHei"]; 
                    ''' % color 
        # set up the Fonts to support Chinese character - Microsoft YaHei
        self.traversal(self.root)
        self.dot += '}'
               
        return self.dot
    
    def traversal(self, node):
        '''preorder'''
        self.dot += node.node2DOT()
        if not node.left:
            return
        self.traversal(node.left)
        self.traversal(node.right)

        

class Node(object):
    '''
    @description: node class 
    @return: node instance
    '''
    def __init__(self, type='normal',
                       depth=0,
                       feature_ind=None,
                       feature_judge=None,
                       feature_judged=None,
                       feature_num=None,
                       sep_point=None,
                       sep_point_ed=None,
                       ins_num=None,
                       ins_ind=None,
                       label=None,
                       impurity=None,
                       left=None,
                       right=None,
                       belong=None):

        self.series = id(self)                 # unique serial number of the node, using its address
        
        self.type = type
        self.depth = depth

        self.feature_ind = feature_ind       # feature index
        self.feature_judge = feature_judge   # judge condition for subnode. For leaf, it's None
        self.feature_judged = feature_judged # judge condition of self. For root, it's None
        self.feature_num = feature_num       # feature number
        self.sep_point = sep_point           # sep_point for subnode. For leaf, it's None
        self.sep_point_ed = sep_point_ed     # sep_point_ed of self. For root, it's None
        self.ins_ind = ins_ind               # instance index
        self.ins_num = ins_num               # instance number
        self.impurity = impurity             # impurity: gini/variance
        self.label = label                   # y label

        self.left = left
        self.right = right

        self.belong = belong                 # if it is left_child: 0, else 1. Root: None

    def node2DOT(self, feature_list=None):
        '''
        output the dot script as string for a node: self context & edge context
        e.g. 5 [label="Feature1 > 3.3\nimpurity = 0.6251\n
                ins_num = 113\nfeature_num = 3\ndepth = 1\nclass = 1.0"];
                5 -> 6; 5 -> 7;
        '''

        if not feature_list:
            feature_judged = 'Feature' + str(self.feature_judged)
        else:
            feature_judged = feature_list[self.feature_judged]

        dot = str(self.series) + ' [label="'
        dot += ( 'node_type = ' + self.type + '\n' )

        eq = ' <= ' if (self.belong == 0) else ' > '
        if self.type != 'root': 
            dot += ( feature_judged + eq + str(self.sep_point_ed) + '\n' )

        dot += ( 'impurity = ' + '%.4f' % self.impurity + '\n' )
        dot += ( 'ins_num = ' + str(self.ins_num) + '\n' )
        dot += ( 'feature_num = ' + str(self.feature_num) + '\n' )
        dot += ( 'depth = ' + str(self.depth) + '\n' )
        dot += ( 'class = ' + str(self.label) )
        dot += '"]; '
        
        if self.type != 'leaf':
            dot += ( str(self.series) + ' -> ' +  str(self.left.series) + '; ' )
            dot += ( str(self.series) + ' -> ' +  str(self.right.series) + '; ' )

        return dot




        














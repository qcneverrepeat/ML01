# coding:utf-8
'''
@autor : qc
@version : 0.1

ID3 FINISHED

TO DO LIST:
    if (all x_i the same) then return leaf or split : return leaf
    if trainset attribute value is not complete, then how to handle the empty branch
    how to visualize
    predict method in tree
    C4.5 & CART
    other control parameters in tree
'''

class Node(object):

    '''decision tree node'''

    def __init__(self, x, y, deep = 1):

        from collections import Counter as ct
        import numpy
        import pandas
        self.pd = pandas
        self.np = numpy
        self.Counter = ct

        self.type = 'node'
        self.entroy = self.__entroy(x)
        self.samples = y.size
        self.label = None
        self.judge = 'Default'
        self.childset = []

        self.x = x
        self.y = y

        self.deep = deep

    def split(self):

        if len(self.Counter(self.y).keys()) == 1 or self.x.shape[1] == 0 or self.x.drop_duplicates().shape[0] == 1:
            # 3 conditions : return leaf
            # y has only one label ; attributes set is empty ; the value in every attributes of all samples are the same
            self.type = 'leaf'
            self.samples = self.y.size
            self.entroy = self.__entroy(self.y)
            self.label = list(self.Counter(self.y).keys())[0]
            return

        att_index = self.__Gain(self.x,self.y).index(max(self.__Gain(self.x,self.y)))
        for key in self.Counter(self.x[self.x.columns[att_index]]).keys():

            subset_x = self.x[self.x[self.x.columns[att_index]] == key]
            del subset_x[subset_x.columns[att_index]] # finally, subset_x will be a empty dataframe with only index ,in shape of (n,0)
            subset_y = self.y[subset_x.index]

            subnode = Node(subset_x,subset_y,deep = self.deep + 1)
            subnode.judge = '%s = %s'%(self.x.columns[att_index], key)
            subnode.label = list(self.Counter(self.y).keys())[0]

            self.childset.append(subnode)
            subnode.split()

    def __Gain(self,x,y):
        gain = []
        entroy = 0
        if y.size != 0:
            for var in x.columns:
                for key,value in self.Counter(x[var]).items():
                    sub_entroy = self.__entroy(y[x[var]==key])
                    entroy += sub_entroy*(value/y.size)
                gain.append(entroy)
        return gain

    def __entroy(self,y):
        proportion = list(self.Counter(y).values())
        entroy = 0
        for i in proportion:
            entroy += -(i/sum(proportion))*self.np.log2(i/sum(proportion))
        return entroy

    def show(self):
        print('type:',self.type)
        print('entroy:',self.entroy)
        print('samples:',self.samples)
        print('label:',self.label)
        print('judge:',self.judge)
        print('deep:',self.deep)
        print('='*10)

# =======================================================================

class Tree(object):

    '''decision tree'''


    def __init__(self, class_weight=None, tree_type='ID3', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False,
            random_state=None,splitter='best'):

        from collections import Counter as ct
        import numpy
        import pandas
        self.pd = pandas
        self.np = numpy
        self.Counter = ct

        # control parameters
        self.__class_weight = class_weight
        self.__type = tree_type
        self.__max_depth = max_depth
        self.__max_features = max_features
        self.__max_leaf_nodes = max_leaf_nodes
        #...

        self.__root = None

    def __treeGenerate_ID3(self,x,y):

        self.__root = Node(x,y)
        self.__root.type = 'root'
        self.__root.label = list(self.Counter(y).keys())[0] # choose the max class in y-label or the only class
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

    def predict(self,x):
        pass

    def traversal(self):
        '''遍历'''
        self.__root.show()
        self.cycle(self.__root)

    def cycle(self,node):
        '''遍历中的递归模块'''
        for sub_node in node.childset:
            sub_node.show()
            self.cycle(sub_node)

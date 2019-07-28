import numpy as np
import pandas as pd
filename = 'D:/jupyter_dir/Python数据分析与挖掘实战/chapter5/demo/data/sales_data.xls' # 当前路径省略即可，间隔用正斜杠；R中的当前路径以.代替
filename2 = 'D:/jupyter_dir/Python数据分析与挖掘实战/chapter6/demo/data/model.xls'
raw = pd.read_excel(filename,index_col = '序号')
x = raw.iloc[:,:3]
y = raw.iloc[:,3] # pandas Series

'''
import decisionTree as DT

# tree = DT.Tree(max_features = None,max_depth = 2)
tree = DT.Tree()
tree.fit(x,y)
tree.traversal()
print(tree.predict(x))
'''

'''
import randomForest as RF
forest = RF.randomForest()
forest.fit(x,y)
print(forest.predict(x,show='all'))
'''

'''
import adaBoost as aB
import decisionTree as DT
adB = aB.adaBoost(base = DT.Tree(max_depth = 2),n_estimators = 10)
adB.fit(x,y)
print(adB.predict(x, show='all'))
'''

'''
import evaluation
predict = pd.Series(np.random.uniform(-1,1,size=500))
predict.iloc[1:6] = 0.3
predict.iloc[9:30] = 0.5
predict.iloc[50:70] = 0.7 # simulate that some instances share the same score
label = pd.Series(np.random.uniform(-1,1,size=500))
label[label<=0] = 'neg'
label[label!='neg'] = 'pos'
# a = evaluation.evaluator.ROC(label,predict,pos_label='pos')
# print(a)
evaluation.evaluator.PR(label,predict,pos_label='pos')
'''

'''
import lossFunction as LF
print(LF.Cost(y,y,type='binary'))
'''

'''
import evaluation
predict = pd.Series(np.random.uniform(-1,1,size=500))
label = pd.Series(np.random.uniform(-1,1,size=500))
label[label<=0] = 'neg'
label[label!='neg'] = 'pos'
a = evaluation.evaluator.EMP(label,predict,pos_label='pos',expect=True)
print(a)
'''


from sklearn import datasets
iris = datasets.load_iris()
x = iris.data
y = iris.target
y[y>1]=1

import linear
md = linear.logit()
md.fit(x,y)
print(md.result)
plt.plot(md.COST)
plt.show()

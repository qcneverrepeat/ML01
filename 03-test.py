
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


import adaBoost as aB
adB = aB.adaBoost(n_estimators = 10)
adB.fit(x,y)
# print(adB.base_weight_set)
print(adB.predict(x, show='all'))

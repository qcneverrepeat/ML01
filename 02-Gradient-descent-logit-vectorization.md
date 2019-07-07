

```python
import numpy as np
import pandas as pd
```


```python
raw = pd.read_excel('D:/jupyter_dir/Python数据分析与挖掘实战/chapter5/demo/data/bankloan.xls')
raw.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年龄</th>
      <th>教育</th>
      <th>工龄</th>
      <th>地址</th>
      <th>收入</th>
      <th>负债率</th>
      <th>信用卡负债</th>
      <th>其他负债</th>
      <th>违约</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41</td>
      <td>3</td>
      <td>17</td>
      <td>12</td>
      <td>176</td>
      <td>9.3</td>
      <td>11.359392</td>
      <td>5.008608</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>27</td>
      <td>1</td>
      <td>10</td>
      <td>6</td>
      <td>31</td>
      <td>17.3</td>
      <td>1.362202</td>
      <td>4.000798</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>40</td>
      <td>1</td>
      <td>15</td>
      <td>14</td>
      <td>55</td>
      <td>5.5</td>
      <td>0.856075</td>
      <td>2.168925</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>41</td>
      <td>1</td>
      <td>15</td>
      <td>14</td>
      <td>120</td>
      <td>2.9</td>
      <td>2.658720</td>
      <td>0.821280</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>28</td>
      <td>17.3</td>
      <td>1.787436</td>
      <td>3.056564</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = raw.iloc[:,:8].T # N*M = 8*700
Y = raw.iloc[:,8].values # 1*M, DATA.FRAME.values -> np.array dtype
```


```python
# initialization
w = np.zeros((1,8))
b = 1

# gradient-descent
for i in range(1000):
    Z = np.dot(w,X) + b
    A = 1/(1 + np.exp(-Z))
    dw = (1/700) * np.dot(X,(A-Y).T).T
    db = (1/700) * np.sum(A-Y).T
    w -= 0.01 * dw
    b -= 0.01 * db

print(w,b)
```

    [[-0.1033291  -0.05781102 -0.4589901  -0.14403647 -0.08745595  0.12336657
       0.90910763  0.16827845]] 0.9091907339221399
    


```python
from sklearn.linear_model.logistic import LogisticRegression
import statsmodels.api as sm
```


```python
# logit in sklearn
clf_1 = LogisticRegression()
clf_1.fit(X.T,Y)

print(clf_1,end='\n\n')
print(clf_1.coef_, clf_1.intercept_)
```

    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='warn',
              n_jobs=None, penalty='l2', random_state=None, solver='warn',
              tol=0.0001, verbose=0, warm_start=False)
    
    [[ 0.02557421  0.05707375 -0.25876387 -0.10047359 -0.00985556  0.05718765
       0.64194083  0.08326905]] [-1.13419814]
    

    D:\installation\anaconda\lib\site-packages\sklearn\linear_model\logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    


```python
# logit in statsmodels
clf_2 = sm.Logit(Y,X.T).fit()
print(clf_2.summary())
```

    Optimization terminated successfully.
             Current function value: 0.398567
             Iterations 7
                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                  700
    Model:                          Logit   Df Residuals:                      692
    Method:                           MLE   Df Model:                            7
    Date:                Fri, 21 Jun 2019   Pseudo R-squ.:                  0.3063
    Time:                        14:20:03   Log-Likelihood:                -279.00
    converged:                       True   LL-Null:                       -402.18
                                            LLR p-value:                 1.641e-49
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    年龄             0.0032      0.012      0.261      0.794      -0.021       0.027
    教育            -0.0313      0.114     -0.275      0.783      -0.254       0.192
    工龄            -0.2670      0.033     -8.054      0.000      -0.332      -0.202
    地址            -0.0906      0.023     -4.008      0.000      -0.135      -0.046
    收入            -0.0147      0.007     -1.983      0.047      -0.029      -0.000
    负债率            0.0264      0.026      1.017      0.309      -0.024       0.077
    信用卡负债          0.7238      0.112      6.448      0.000       0.504       0.944
    其他负债           0.1466      0.071      2.077      0.038       0.008       0.285
    ==============================================================================
    

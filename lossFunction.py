# coding: utf-8
# loss-function: L(y_hat,y) based on single instance
# cost-function: J(w,b) based on the whole sample

import numpy as np

def Loss(pre, real, type = 'square'):
    if type == 'square':
        return (pre-real)**2
    if type == 'binary':
        return sum([pre != real])
    if type == 'log':
        # real must be 1,-1
        # pre must be (0,1)
        return -(real*np.log(pre) + (1-real)*np.log(1-pre))
    if type == 'abs':
        return abs(pre-real)
    else:
        print('Loss function type should be: square, binary, log, abs.')

def Cost(pre, real, type = 'square'):
    # input: 2 pandas series
    return Loss(pre, real, type = type).mean()

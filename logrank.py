'''
monotonic transform of values to log of rank
(a kind of logarithmic equalization)
useful to visualize detector outputs,
'''

## should go somewhere! (rocu ? equalize ?)

import numpy as np

def logrank(a_values,/,base=10,lothresh=2.0):
    '''monotonic transform of values to log of rank'''
    ## base=0 corresponds to natural log
    ## higher values of low threshold enables
    ## higher contrast images (also more white)
    a_shape = a_values.shape
    a_values = a_values.reshape(-1) ## convert to 1d array
    a_rnk = np.argsort(np.argsort(-a_values)) ## argsort twice to get rank
    a_xformed = -np.log((1+a_rnk)/a_values.size)
    if base:
        a_xformed = a_xformed/np.log(base)
    if lothresh:
        np.clip(a_xformed,lothresh, None, out=a_xformed)
    a_xformed = a_xformed.reshape(a_shape)
    return a_xformed

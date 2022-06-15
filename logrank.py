'''
monotonic transform of values to log of rank
(a kind of logarithmic equalization)
useful to visualize detector outputs,
'''

## should go somewhere! (rocu ? equalize ?)

import numpy as np

def logrank(a,base=10,lothresh=2.0):
    '''monotonic transform of values to log of rank'''
    ## base=0 corresponds to natural log
    ## higher values of low threshold enables higher contrast images (also more white)
    a_shape = a.shape
    a = a.reshape(-1) ## convert to 1d array
    arnk = np.argsort(np.argsort(-a)) ## argsort twice to get rank
    ax = -np.log((1+arnk)/a.size)
    if base:
        ax = ax/np.log(base)
    if lothresh:
        np.clip(ax,lothresh, None, out=ax)
    ax = ax.reshape(a_shape)
    return ax

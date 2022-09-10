'''background estimation
   provides BackgroundEstimator class
'''

import numpy as np
from sklearn.linear_model import LinearRegression
import tqdm

import verbose as v
from . import annfeat

class FlatRegressor:
    '''
    another class that makes a regressor object, but this one is trivial
    and just takes 1 times first feature -- makes (some) sense for mean, median
    '''
    def fit(self,xfeatures,ytrue):
        '''no actual fitting'''
        pass

    def predict(self,xdata):
        '''prediction is just first component of xdata'''
        ypred = xdata[...,0]
        return ypred

def get_regressor(model):
    '''return a regressor class'''
    ## via scikit-learn
    if model == "lin":
        regressor = LinearRegression(fit_intercept=False)
    else:
        ## other models defined in original mkresid
        raise RuntimeError("Invalid model: "+model)

    v.vvprint("Regressor: ",regressor)
    return regressor

def get_predictor(y_ins,x_ins,model='lin',flat=False):
    '''return a prediction function fcn so that fcn(x) gives y'''
    if flat:
        regressor = FlatRegressor()
    else:
        regressor = get_regressor(model)
    y_ins = y_ins.reshape(-1)
    num_features = x_ins.shape[0]
    v.vprint_only(2,f'{num_features=}')
    x_ins = x_ins.reshape(num_features,-1)
    regressor.fit(x_ins.T,y_ins)
    #v.print('coef:',regressor.coef_)
    ## return function that can predict y, given x
    def predictor(x_oos):
        assert x_oos.shape[0] == num_features
        imshape = x_oos.shape[1:]
        x_oos = x_oos.reshape(num_features,-1)
        y_oos = regressor.predict(x_oos.T)
        y_oos = y_oos.reshape(imshape)
        return y_oos
    return predictor


class BackgroundEstimator:
    '''
    local background estimator based on symmetric regression on the
    pixels in the annulus surrounding the pixel under test
    '''
    ## NOTE: Assume xcube is BIP

    def __init__(self,rinn=1,rout=1,features='mean'):
        self.rinn = rinn
        self.rout = rout
        self.band_predictor = None ## None signals that it has not been fit, yet
        self.num_bands = None
        self.mean = None    #global mean used in the edges
        self.flat = False   #if True, use features as predictors
        def annfcn(ximage):
            '''convert 2d image to features'''
            assert ximage.ndim == 2
            return annfeat.AnnulusFunction[features](ximage,self.rout,self.rinn)
        self.annfcn = annfcn
        if features in annfeat.non_regressive_features():
            ## use mean/median/card as literal predictor, don't regress
            self.flat = True
        self.tqdmfilter = lambda x: x

    def use_tqdm(self,flag=True):
        '''if flat set, use tqdm progress bar during fit'''
        if flag:
            self.tqdmfilter = tqdm.tqdm
        else:
            self.tqdmfilter = lambda x: x

    def fit(self,xcube):
        '''fit a separate regressor for each band of the data cube'''
        assert xcube.ndim == 3
        self.num_bands = xcube.shape[-1]
        v.vprint('num_bands=',self.num_bands)
        self.band_predictor = [None]*self.num_bands ## set up empty array
        self.mean = [0]*self.num_bands
        for band in self.tqdmfilter(range(self.num_bands)):
            band_data = xcube[:,:,band].copy()
            self.mean = np.mean(band_data)
            ## Y is "true" center-pixel value
            y_true = band_data[self.rout:-self.rout,
                              self.rout:-self.rout].copy()
            x_features = self.annfcn(band_data)
            self.band_predictor[band] = get_predictor(y_true,x_features,flat=self.flat)

    def predict(self,xcube):
        '''apply the predictor that was obtained in the fit method'''
        ## Note that the edges are "fit" with global mean
        assert xcube.ndim == 3
        assert xcube.shape[-1] == self.num_bands
        assert self.num_bands is not None
        assert self.band_predictor is not None
        x_est = xcube.copy()
        x_est[:] = self.mean
        inx = slice(self.rout,-self.rout) ## inner-x, away from border
        for band in range(self.num_bands):
            band_data = xcube[:,:,band].copy()
            x_features = self.annfcn(band_data)
            y_est = self.band_predictor[band](x_features)
            x_est[inx,inx,band] = y_est[:,:]
        return x_est

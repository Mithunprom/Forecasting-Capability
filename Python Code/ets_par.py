#holts tunned model
import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from numpy import array

from multiprocessing import cpu_count
from joblib import Parallel,delayed,parallel_backend

from warnings import catch_warnings
from warnings import filterwarnings
from sklearn.metrics import mean_absolute_error as mae
def mad(pred,data_test):
    #MEAN=(np.mean(data_train)+np.mean(data_test))/2
    MEAN=np.mean(data_test)
    mad=0
    for i in range(len(pred)):
        mad+=abs(MEAN-pred[i])
    return mad/len(pred)
# one-step Holt Winter's Exponential Smoothing forecast
def ets_forecast(history, config,dim_test=5):
    e,t,s,d,p = config
    # define model model
    history = array(history)
    model=  ETSModel(history, error=e, trend=t, seasonal=s,
                damped_trend=d, seasonal_periods=p)
    # fit model
    model_fit = model.fit(maxiter=2000)
    # make one step forecast
    yhat = model_fit.predict(len(history), len(history)+dim_test-1)
    return yhat


# walk-forward validation for univariate data
def walk_forward_validation(data_train,data_test, n_test, cfg):
    #predictions = list()
    # split dataset
    train, test = data_train,data_test
    history = [x[0] for x in train.values]#[x for x in train]#[
    predictions = ets_forecast(history, cfg,dim_test=n_test)
    #predictions=yhat
    error = mae(predictions,test)
    return error

# score a model, return None on failure
def score_model(data_ttrain,data_test, n_test, cfg, debug=False):
    result = None
    # convert config to a key
    key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
        result = walk_forward_validation(data_ttrain,data_test, n_test, cfg)
    else:
        '''
        
        '''
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result = walk_forward_validation(data_ttrain,data_test, n_test, cfg)
        except:
            error = None
    # check for an interesting result
    if result is not None:
        print(' > Model[%s] %.3f' % (key, result))
    return (key, result)


# grid search configs
def grid_search(data_train,data_test, cfg_list, n_test, parallel=True):
    scores = None
    if parallel:
        # execute configs in parallel
        with parallel_backend("loky"):
            try:
                scores= Parallel(n_jobs=cpu_count())(delayed(score_model)(data_train,data_test, n_test, cfg) for cfg in cfg_list)
            except Exception as e:
                print('Fatal Error....')
                print(e)
    else:
        scores = [score_model(data_train,data_test, n_test, cfg) for cfg in cfg_list]
    # remove empty results
    scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores
def exp_smoothing_configs(seasonal=[2,3,4,5,6,7,8,9,10,11,12]):

    models = list()
    # define config lists
    e_params = ['add', 'mul']
    t_params = ['add', 'mul', None]
    s_params = ['add', 'mul', None]
    p_params = seasonal
    d_params = [True, False]
    
    # create config instances
    for e in e_params:
        for t in t_params:
            for s in s_params:
                for p in p_params:
                    for d in d_params:
                            cfg = [e,t,s,d,p]
                            models.append(cfg)
    return models
def sun_ets(data_train,data_test):
    data_train = data_train.astype('float64')
    n_test = len(data_test)
    # model configs
    cfg_list = exp_smoothing_configs()
    # grid search

    scores = grid_search(data_train,data_test, cfg_list, n_test)
    print(scores)
    print('done')
    print('The top three candidates are the following:')
    # list top 3 configs
    for cfg, error in scores[:3]:
        print(cfg, error)
    tup=[]
    for i in range(3):
        char=scores[i][0]
        char=char.replace(']','')
        char = char.replace('[', '')
        char = char.replace("'", '')
        char = char.replace(" ", '')
        tup.append(char.split(','))
    return tup

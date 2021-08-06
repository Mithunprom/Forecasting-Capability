#holts tunned model
#from Moldels_build_db import *
#from Models_build import *
#from Neural_net import *
from joblib import Parallel,delayed,parallel_backend

import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from numpy import array
from multiprocessing import cpu_count
from warnings import catch_warnings
from warnings import filterwarnings
from sklearn.metrics import mean_absolute_error as mae

# one-step Holt Winter's Exponential Smoothing forecast
def exp_smoothing_forecast(history, config,dim_test=5):
    t,d,s,p,b,r = config
    # define model model
    history = array(history)
    model = ExponentialSmoothing(history, trend=t, damped=d, seasonal=s, seasonal_periods=p)
    # fit model
    model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
    # make one step forecast
    yhat = model_fit.predict(len(history), len(history)+dim_test-1)
    return yhat


# walk-forward validation for univariate data
def walk_forward_validation(data_train,data_test, n_test, cfg):
    train, test = data_train,data_test
    history = [x[0] for x in train.values]#[x for x in train]#[x[0] for x in train.values]
    predictions = exp_smoothing_forecast(history, cfg,dim_test=n_test)
    error = mae(predictions,test)
    return error

# score a model, return None on failure
def score_model(data_train,data_test, n_test, cfg, debug=False):
    result = None
    # convert config to a key
    key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
        result = walk_forward_validation(data_train,data_test, n_test, cfg)
    else:
        print('error')
        '''
        
        '''
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result = walk_forward_validation(data_train,data_test, n_test, cfg)
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
                #score_model(data_train,data_test, n_test, cfg_list[0])
                scores= Parallel(n_jobs=cpu_count())(delayed(score_model)\
                    (data_train,data_test, n_test, cfg) for cfg in cfg_list)
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
    t_params = ['add', 'mul', None]
    d_params = [True, False]
    s_params = ['add', 'mul', None]
    p_params = seasonal
    b_params = [True, False]
    r_params = [True, False]
    # create config instances
    for t in t_params:
        for d in d_params:
            for s in s_params:
                for p in p_params:
                    for b in b_params:
                        for r in r_params:
                            cfg = [t,d,s,p,b,r]
                            models.append(cfg)
    return models
def sun_holts(data_train,data_test):
    n_test = len(data_test)
    # model configs
    cfg_list = exp_smoothing_configs()
    # grid search

    scores = grid_search(data_train,data_test, cfg_list, n_test)
    print(scores)
    print('done')
    print('top three candidates are the folowing:')
    # list top 3 configs
    for cfg, error in scores[:3]:
        print(cfg, error)
    tup=[]
    if len(scores)<1:
        print('###################################')
    for i in range(3):
        char=scores[i][0]
        char=char.replace(']','')
        char = char.replace('[', '')
        char = char.replace("'", '')
        char = char.replace(" ", '')
        tup.append(char.split(','))
    return tup
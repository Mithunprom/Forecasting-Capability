from numpy import array
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.ar_model import AutoReg
from random import random
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error as mae
import numpy as np
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from multiprocessing import cpu_count
from joblib import Parallel,delayed,parallel_backend
import pickle
import warnings
warnings.filterwarnings("ignore")

def evaluate_sarima_model(train, test, arima_order, seasonalOrder):

    try:
        # no need to calcuate if order as well as seasonal differencing is 0
        if (arima_order[1]+seasonalOrder[1])==0:
          print(f"##### Skipped modelling with: {arima_order}, {seasonalOrder} --> Both d & D are zeroes\n")
          # return a high value of RMSE so that it sits at the bottom of the list when sorted
          return 999999999, arima_order, seasonalOrder
          
        #y_hat = test.copy()
        model = SARIMAX(train, order=arima_order, seasonal_order=seasonalOrder)
        model_fit = model.fit(method='cg')
        predict = model_fit.predict(len(train), len(train)+len(test)-1, dynamic=True)
        #y_hat['model_prediction']=predict

        error = mae(predict.values,test)
        print(f"> Model: {error}, {arima_order}, {seasonalOrder}\n")
        return error, arima_order, seasonalOrder
    except Exception as e:
        # in case of convergence errors, non-invertible errors, etc.
        print(f"##### Skipped modelling with: {arima_order}, {seasonalOrder}\n")
        print(e)
        return 999999999, arima_order, seasonalOrder
def evaluate_models_parallely(train, test, p_values, d_values, q_values, P_values, D_values, Q_values, m_values):
    # utilize aall available cores using n_jobs = cpu_count()
    with parallel_backend("loky"):
        try:
            scor= Parallel(n_jobs=cpu_count())(
                delayed(evaluate_sarima_model)(train, test, (p,d,q), (P,D,Q,m)) for m in m_values for Q in Q_values for D in D_values for P in P_values for q in q_values for d in d_values for p in p_values)
        except Exception as e:
            print('Fatal Error....')
            print(e)
    return scor
#def fit_sarima(data_train,data_test):
    # specify the range of values we want ot try for the different hyperprameters
from Neural_net import *


def sarima_model(data_train,data_test):
    p_values = np.arange(0, 5)
    d_values = np.arange(1, 3)
    q_values = np.arange(1, 4)
    P_values = np.arange(0, 4)
    D_values = np.arange(1, 3)
    Q_values = np.arange(0, 3)
    m_values = np.arange(2,13)#np.arange(2,13)

    # total combinations being tried: 2*1*3*2*1*3*11 = 396

    scor=evaluate_models_parallely(data_train, data_test, p_values, d_values, q_values, P_values, D_values, Q_values, m_values)

    scores=[]
    for tup_list in scor:
        for tup in tup_list:
            scores.append(tup)

    # sort the results on basis of MAE scores (ascending)
    scor.sort(key=lambda x: x[0])

    #print('\nTop 5 SARIMA params with minimum MADs:\n')
    #for x in scores[:5]:
      #print(f'MAE={x[0]}  order={x[1]}  seasonal_order={x[2]}\n')
      #print(f'order is: {x[1]}')
      #print(f'seasonal order is: {x[2]}')

    print("DONE!")
    print(scor[0])
    p,d,q,P,D,Q,m=[],[],[],[],[],[],[]
    for i in range(5):
        res=scor[i][1]+scor[i][2]
        p.append(res[0])
        d.append(res[1])
        q.append(res[2])
        P.append(res[3])
        D.append(res[4])
        Q.append(res[5])
        m.append(res[6])
    return p,d,q,P,D,Q,m


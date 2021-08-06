from statsmodels.tsa.ar_model import AutoReg
from random import random
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error as mae
import numpy as np
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from fbprophet import Prophet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings
#%%
#'add', 'add', 'mul', False, 12
#ETS tunned model

from statsmodels.tsa.exponential_smoothing.ets import ETSModel
def ETS_model(data_train,data_test,e,t,s,p,d):
    data_train=data_train.astype('float64')
    model = ETSModel(data_train.values.reshape(-1,), error=e, trend=t, seasonal=s,
                damped_trend=d, seasonal_periods=p)
    fit = model.fit(maxiter=2000)
    pred=fit.forecast(steps=len(data_test))
    print("ETS error:" + str(mae(pred,data_test)))
    return fit, mae(pred,data_test),pred
#%%
#prophet method
def prophet_model(data_trained,data_tested):
    Map={'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06'
        ,'Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'}
    data_trained.reset_index(inplace=True)
    data_trained = data_trained.rename(columns={"Volumes":"y","Date_time":"ds"})
    data_tested.reset_index(inplace=True)
    data_tested = data_tested.rename(columns={"Volumes":"y","Date_time":"ds"})
    m=Prophet()
    # m = Prophet(yearly_seasonality=False, weekly_seasonality=False,seasonality_mode='multiplicative',
    #             daily_seasonality=False,growth='linear').add_seasonality(name='monthly', period=30.5, fourier_order=55).add_seasonality(name='yearly',period=365.25,fourier_order=20).add_seasonality(name='quarterly',period=365.25/4,fourier_order=15)
    model = m.fit(data_trained)
    dt=model.make_future_dataframe(periods=len(data_tested), freq = "M",include_history=False)
    forecast = m.predict(dt)
    #forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    yhat=forecast[['yhat']]
    print("The prophet model with MAD is: "+str(mae(yhat.values,data_tested['y'].values)))
    return model,mae(yhat.values,data_tested['y'].values),yhat.values
#MAE={}
#Models={}
#Data=pd.read_csv('C:/Users/AC91508/OneDrive - Lumen/Desktop/Forcasting_pro/data_trans.csv',header=0)
#Data=Data.set_index('Date')

# contrived dataset
#data=Data['A']
#autocorrelation_plot(data)
#pyplot.show()
#data_train=data[0:24]
#data_test=data[24:]
#%%
# fit model
def AR_model(data_train,data_test):
    Error=[]
    for i in range(1,10):
        try:
            model = AutoReg(data_train, lags=i)
            model_fit = model.fit()
            # make prediction
            y_hat = model_fit.predict(start=len(data_train), end=len(data_train)+len(data_test)-1)
            #print(mae(data_test,y_hat))
            Error.append(mae(y_hat,data_test))
        except:continue
    #finding the optimum lags
    lags=np.argmin(Error)+1
    model = AutoReg(data_train, lags=lags)
    model_fit1 = model.fit()
    y_hat = model_fit1.predict(start=len(data_train), end=len(data_train)+len(data_test)-1)
    print("The AR model have the lag: "+str(lags)+ ' & MAE: '+str(mae(y_hat,data_test)))
    return model_fit1,mae(y_hat,data_test),lags,y_hat
#mod_ar,err,lg=AR_model(data_train,data_test)
#Models['AR with lags '+str(lg)]=mod_ar
#MAE['AR with lags '+str(lg)]=err
#%%
#Moving Average (MA)
from statsmodels.tsa.arima.model import ARIMA
# fit model
def MA_model(data_train,data_test):
    Error=[]
    for i in range(1,6):
        try:
            model = ARIMA(data_train, order=(0, 0, i))
            model_fit = model.fit()
            # make prediction
            y_hat = model_fit.predict(start=len(data_train), end=len(data_train)+len(data_test)-1)
            #print(mae(data_test,y_hat))
            Error.append(mae(y_hat,data_test))
        except: continue
    #finding the optimum lags
    lags=np.argmin(Error)+1
    model = ARIMA(data_train, order=(0, 0, lags))
    model_fit2= model.fit()
    y_hat = model_fit2.predict(start=len(data_train), end=len(data_train)+len(data_test)-1)
    #MAE['MA with lags '+str(lags)]=min(Error)
    print("MA model have the lag: "+str(lags)+ ' & MAE: '+str(mae(y_hat,data_test)))
    return model_fit2,mae(y_hat,data_test),lags,y_hat
#mod_ma,err1,lg1=MA_model(data_train,data_test)
#Models['MA with lags '+str(lg1)]=mod_ma
#MAE['MA with lags '+str(lg1)]=err1
#%%
#ARMA
def ARMA_model(data_train,data_test):
    Error={}
    for i in range(10):
        for j in range(4):
            try:
                model = ARIMA(data_train, order=(i, 0, j))
                model_fit = model.fit()
                # make prediction
                yhat = model_fit.predict(len(data_train), len(data_train)+len(data_test)-1)
                Error[(i,j)]=mae(yhat,data_test)
            except:continue
    # get the indices in the restricted data
    lag1,lag2 = min(Error, key=Error.get)
    model = ARIMA(data_train, order=(lag1, 0, lag2))
    model_fit = model.fit()
    yhat = model_fit.predict(len(data_train), len(data_train)+len(data_test)-1)
    print('ARMA with lags ('+str(lag1)+" , "+str(lag2)+') & MAE: '+
          str(mae(yhat,data_test)))
    return model_fit,mae(yhat,data_test),lag1,lag2,yhat
# make prediction
#mod_arma,err2,lg21,lg22=ARMA_model(data_train,data_test)
#Models['ARMA with lags ('+str(lg21)+','+str(lg22)+')']=mod_arma
#MAE['ARMA with lags ('+str(lg21)+','+str(lg22)+')']=err2
#%%
#ARIMA
def ARIMA_model(data_train,data_test):
    Error={}
    # ARIMA example
    # fit model
    for i in range(10):
        for j in range(3):
            for k in range(3):
                try:
                    model = ARIMA(data_train, order=(i, k, j))
                    model_fit = model.fit()
                    # make prediction
                    yhat = model_fit.predict(len(data_train), len(data_train)+len(data_test)-1)
                    Error[(i,k,j)]=mae(yhat,data_test)
                    #print('error is: ' +str(mae(data_test,yhat))+' for order ('+str(i)+str(k)+str(j)+')')
                except:continue
    # get the indices in the restricted data
    lag1,diff,lag2 =  min(Error, key=Error.get)
    model = ARIMA(data_train, order=(lag1, diff, lag2))
    model_fit = model.fit()
    yhat = model_fit.predict(len(data_train), len(data_train)+len(data_test)-1)
    print('ARIMA with order ('+str(lag1)+str(diff)+str(lag2)+') MAE: '+str(mae(yhat,data_test)))
    return model_fit,mae(yhat,data_test),lag1,diff,lag2,yhat
# make prediction
#mod_arima,err3,lg31,diff3,lg32=ARIMA_model(data_train,data_test)
#Models['ARIMA with order ('+str(lg31)+','+str(diff3)+','+str(lg32)+')']=mod_arima
#MAE['ARIMA with order ('+str(lg31)+','+str(diff3)+','+str(lg32)+')']=err3
#%%
# SARIMA example
from statsmodels.tsa.statespace.sarimax import SARIMAX
def SARIMA_model(data_train,data_test,lg41,diff4,lg42,p,d,q,m):
    #lg41,diff4,lg42,p,d,q,m=0,2,2,0,1,2,3
    model =  SARIMAX(data_train, order=(lg41,diff4, lg42), seasonal_order=(p, d, q, m))
    model_fit = model.fit(method='cg')
    # make prediction
    yhat = model_fit.predict(len(data_train), len(data_train)+len(data_test)-1)
    Error=mae(yhat,data_test)
    print('SARIMA with order ('+str(lg41)+str(diff4)+str(lg42)+"),("+str(p)+str(d)+
          str(q)+str(m)+')')
    return model_fit,Error,lg41,diff4,lg42,p,d,q,m,yhat
#mod_sarima,err4,lg41,diff4,lg42,p,d,q,m = SARIMA_model(data_train,data_test)
#Models['SARMA with order ('+str(lg41)+str(diff4)+str(lg42)+'),('+str(p)+str(d)+str(q)+str(m)+')']=mod_sarima
#MAE['SARMA with order ('+str(lg41)+str(diff4)+str(lg42)+'),('+str(p)+str(d)+str(q)+str(m)+')']=err4
#%%
#vector  Autoregression

#%%
#Holt-linear trend no dmaped
from statsmodels.tsa.holtwinters import ExponentialSmoothing
def exp_add(data_train,data_test):
    model = ExponentialSmoothing(data_train, trend="additive", seasonal=None)
    fit = model.fit(optimized=True)
    pred = fit.forecast(len(data_test))
    error01=mae(pred,data_test)
    print('Holts add linear trend: '+str(error01))
    #yhat= fit.forecast(6)
    return fit, mae(pred,data_test),pred
#mod_exp_add,error06=exp_add(data_train,data_test)
#Models["Holts ES with additive trend"]=mod_exp_add
#MAE["Holts ES with additive trend"]= error06
#%%
#Holt-linear trend with dmaped
def mod_exp_add_d(data_train,data_test):
    model = ExponentialSmoothing(data_train, trend="additive", seasonal=None,damped=True)
    fit = model.fit(optimized=True)
    pred = fit.forecast(len(data_test))
    error01=mae(pred,data_test)
    #yhat= fit.forecast(6)
    print('Holts add linear dampen trend: '+str(error01))
    return fit,error01,pred
#mod_exp_add_d,error06=exp_add(data_train,data_test)
#Models["Holts ES with additive damped trend"]=mod_exp_add_d
#MAE["Holts ES with additive damped trend"]= error06
#%%
#Holt's exponential  trend model
def exp_mul(data_train,data_test):
    model = ExponentialSmoothing(data_train, trend="mul", seasonal=None)
    fit = model.fit(optimized=True)
    pred = fit.forecast(len(data_test))
    error01=mae(pred,data_test)
    print('Holts mul linear trend: '+str(error01))
    #yhat= fit.forecast(6)
    return fit,error01,pred
#mod_exp_mul,error07=exp_add_damped(data_train,data_test)
#Models["Holts ES with mul trend"]=mod_exp_mul
#MAE["Holts ES with mul trend " ]= error07
#%%
#Holt's exponential  trend model with damped
def exp_mul_damped(data_train,data_test):
    model = ExponentialSmoothing(data_train, trend="mul", seasonal=None,damped=True)
    fit = model.fit(optimized=True)
    pred = fit.forecast(len(data_test))
    error01=mae(pred,data_test)
    #yhat= fit.forecast(6)
    print('Holts mul linear dampen trend: '+str(error01))
    return fit,error01,pred
#mod_exp_mul_d,error07=exp_add_damped(data_train,data_test)
#Models["Holts ES with mul damped trend"]=mod_exp_mul_d
#MAE["Holts ES with mul damped trend " ]= error07
#%%
#Holt-Winter’s Seasonal Smoothing additive  model no dmaped
from statsmodels.tsa.holtwinters import ExponentialSmoothing
def exp_add_ssn(data_train,data_test):
    error01={}
    ln,an=divmod(len(data_train),12)
    if an==0:Ln=12
    else: Ln=an
    for i in range(2,Ln):
        try:
            model = ExponentialSmoothing(data_train, trend="additive", seasonal="additive", seasonal_periods=i)
            fit = model.fit(optimized=True)
            pred = fit.forecast(len(data_test))
            error01[i]=mae(pred,data_test)
        except:continue
    sp=  min(error01, key=error01.get) 
    model = ExponentialSmoothing(data_train, trend="additive", seasonal="additive", seasonal_periods=sp)
    fit = model.fit()
    yhat= fit.forecast(len(data_test))
    print('Holts add linear trend and seasonility: '+str(mae(pred,data_test)))
    return fit,mae(pred,data_test),sp,yhat
#mod_exp_add,error06,sp=exp_add(data_train,data_test)
#Models["Holts ES with additive trend and seasonality of " +str(sp)+' periods']=mod_exp_add
#MAE["Holts ES with additive trend and seasonality of " +str(sp)+' periods']= error06

#%%
#Holt-Winter’s Seasonal Smoothing additive model with damped 
def exp_add_ssn_damped(data_train,data_test):
    error01={}
    ln,an=divmod(len(data_train),12)
    if an==0:Ln=12
    else: Ln=an
    for i in range(2,Ln):
        try:
            model = ExponentialSmoothing(data_train, trend="additive", seasonal="additive", seasonal_periods=i, damped=True)
            fit = model.fit(optimized=True)
            pred = fit.forecast(len(data_test))
            error01[i]=mae(pred,data_test)
        except:continue
    sp=  min(error01, key=error01.get) 
    model = ExponentialSmoothing(data_train, trend="additive", seasonal="additive", seasonal_periods=sp, damped=True)
    fit = model.fit()
    yhat= fit.forecast(len(data_test))
    print('Holts add linear dampen trend and seasonility: '+str(mae(pred,data_test)))
    return fit,mae(pred,data_test),sp,yhat
#mod_exp_add_damped,error07,sp=exp_add_damped(data_train,data_test)
#Models["Holts ES with additive damped trend and seasonality of " +str(sp)+' periods']=mod_exp_add_damped
#MAE["Holts ES with additive damped trend and seasonality of " +str(sp)+' periods']= error07


#%%
#Holt-Winter’s Seasonal Smoothing multiplicative  model no dmaped
def exp_mul_ssn(data_train,data_test):
    error01={}
    ln,an=divmod(len(data_train),12)
    if an==0:Ln=12
    else: Ln=an
    for i in range(2,Ln):
        try:
            model = ExponentialSmoothing(data_train, trend="multiplicative", seasonal="multiplicative", seasonal_periods=i)
            fit = model.fit(optimized=True)
            pred = fit.forecast(len(data_test))
            error01[i]=mae(pred,data_test)
        except:continue
    sp=  min(error01, key=error01.get) 
    model = ExponentialSmoothing(data_train, trend="multiplicative", seasonal="multiplicative", seasonal_periods=sp)
    fit = model.fit()
    yhat= fit.forecast(len(data_test))
    print('Holts mul linear trend and seasonility: '+str(mae(pred,data_test)))
    return fit,mae(pred,data_test),sp,yhat
#mod_exp_mul,error08,sp=exp_mul(data_train,data_test)
#Models["Holts ES with multiplicative trend and seasonality of " +str(sp)+' periods']=mod_exp_mul
#MAE["Holts ES with multiplicative trend and seasonality of " +str(sp)+' periods']= error08

#%%
#Holt-Winter’s Seasonal Smoothing multiplicative model with damped 
def exp_mul_ssn_damped(data_train,data_test):
    error01={}
    ln,an=divmod(len(data_train),12)
    if an==0:Ln=12
    else: Ln=an
    for i in range(2,Ln):
        try:
            model = ExponentialSmoothing(data_train, trend="multiplicative", seasonal="multiplicative", seasonal_periods=i, damped=True)
            fit = model.fit(optimized=True)
            pred = fit.forecast(len(data_test))
            error01[i]=mae(pred,data_test)
        except:continue
    sp=  min(error01, key=error01.get) 
    model = ExponentialSmoothing(data_train, trend="multiplicative", seasonal="multiplicative", seasonal_periods=sp, damped=True)
    fit = model.fit()
    yhat= fit.forecast(len(data_test))
    print('Holts mul linear dampen trend and seasonility: '+str(mae(pred,data_test)))
    return fit,mae(pred,data_test),sp,yhat
#mod_exp_mul_damped,error09,sp=exp_mul_damped(data_train,data_test)
#Models["Holts ES with multiplicative damped trend and seasonality of " +str(sp)+' periods']=mod_exp_mul_damped
#MAE["Holts ES with multiplicative damped trend and seasonality of " +str(sp)+' periods']= error09
#%%
#holts tunned model
def tunned_holts(data_train,data_test,t,d,s,sp,bc,rb):
    model = ExponentialSmoothing(data_train, trend=t, damped=d, seasonal=s, seasonal_periods=sp)
    # fit model
    model_fit = model.fit(use_boxcox=bc,
                       remove_bias=rb)
    pred = model_fit.forecast(len(data_test))
    error01=mae(pred,data_test)
    print('Holts tunned model: '+str(error01))
    return model_fit, error01,pred
#%%
#Simple Exponential Smoothing (SES)
# SES example
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
# fit model
def Sexp_model(data_train,data_test):
    model = SimpleExpSmoothing(data_train)
    model_fit = model.fit(optimized=True)
    # make prediction
    yhat = model_fit.predict(len(data_train), len(data_train)+len(data_test)-1)
   # print(mae(data_test,yhat))  
    print('Simple EXP MAD: '+str(mae(data_test,yhat)))
    return model_fit,mae(yhat,data_test),yhat
#mod_sexp,error10=Sexp_model(data_train,data_test)
#Models["Simple Exponential Smoothing"]=mod_sexp
#MAE["Simple Exponential Smoothing"]=error10
#%%
'''
pip install git+https://github.com/jmetzen/gp_extras.git
'''
def gp_mod(data_train,data_test,df):
    from sklearn.gaussian_process.kernels import DotProduct,Matern, RBF, WhiteKernel, ExpSineSquared, ConstantKernel,  RationalQuadratic
    #from gp_extras.kernels import ManifoldKernel,HeteroscedasticKernel
    from sklearn.gaussian_process import GaussianProcessRegressor
    from itertools import permutations, combinations
    from sklearn.cluster import KMeans

    #X_train=np.linspace(0,data_train.shape[0]-1,data_train.shape[0]).reshape(-1,1)
    #X_test=np.linspace(data_train.shape[0],data_train.shape[0]+data_test.shape[0]-1,data_test.shape[0]).reshape(-1,1)
    try:data_trained,data_tested=data_train['Volumes'].values.reshape(-1,1),data_test['Volumes'].values.reshape(-1,1)
    except:data_trained,data_tested=data_train['Volumes'].reshape(-1,1),data_test['Volumes'].reshape(-1,1)
    # Get all permutations of [1, 2, 3]
    X_train,X_test=df.iloc[:len(data_train),:].values,df.iloc[len(data_train):,:].values   
    k0 = WhiteKernel(noise_level=0.3**2, noise_level_bounds=(0.01**2, 2**2))
    
    k1 = ConstantKernel(constant_value=1) * \
      ExpSineSquared()+k0
    k2=RBF()+k0#RationalQuadratic(alpha=.1,length_scale=1.0)
    k3=Matern()+k0
    k4 = ConstantKernel(constant_value=100, constant_value_bounds=(1, 500)) * \
      RationalQuadratic(length_scale=500, length_scale_bounds=(1, 1e4), alpha= 50.0, alpha_bounds=(1, 1e3))+k0
    k5=DotProduct()+k0
    n_samples = len(data_trained)
    n_features = X_train.shape[1]
    n_dim_manifold = 2
    n_hidden = 3
    architecture=((n_features, n_hidden, n_dim_manifold),)
    #kernel_nn = ConstantKernel (1.0, (1e-10, 100)) \
    #* ManifoldKernel.construct(base_kernel=RBF(0.1, (1.0, 100.0)),
    #                           architecture=architecture,
    #                           transfer_fct="tanh", max_nn_weight=1.0) \
    #+ WhiteKernel(1e-3, (1e-10, 1e-1))
    #prototypes = KMeans(n_clusters=4).fit(X_train).cluster_centers_
    #kernel_hetero = ConstantKernel(1.0, (1e-10, 1000)) * RBF(1, (0.01, 100.0)) \
    #+ HeteroscedasticKernel.construct(prototypes, 1e-3, (1e-10, 50.0),
     #                                 gamma=5.0, gamma_bounds="fixed")\
    #+ WhiteKernel(1e-3, (1e-10, 1e-1))
    
    kernel_1  = [k0]#[k0,k1,k2,k3,k4,k5]#,kernel_nn
    K=[]
    Error=[]
    for L in range(0, len(kernel_1)+1):
        for subset in combinations(kernel_1, L):
            K.append(subset)
    del K[0]
    #K.append(kernel_hetero)
    for i in range(len(K)):
        try:
            #print(sum(K[i]))
            gp1 = GaussianProcessRegressor(
            kernel=sum(K[i]),
            #kernel=kernel_hetero,
            n_restarts_optimizer=5, 
            normalize_y=True,
            alpha=0.0,random_state=0)
            gp1.fit(X_train, data_trained)
        except:
            continue
        y_pred= gp1.predict(X_test, return_std=False)
        Error.append(mae(y_pred,data_tested))
    Min_e=np.argmin(Error)
    gp1 = GaussianProcessRegressor(
            kernel=sum(K[Min_e]), 
            n_restarts_optimizer=5, 
            normalize_y=True,
            alpha=0.0,random_state=0)
        
    gp1.fit(X_train, data_trained)
    y_pred= gp1.predict(X_test, return_std=False)
    mae(y_pred,data_tested)
    print('GPY: '+str(mae(y_pred,data_tested)))
    return gp1,mae(y_pred,data_tested),y_pred

def Cart_mod(data_trained,data_tested,df):
    #df=ml_data(df)
    df_train_x,df_test_x=df.iloc[:len(data_trained),:].values,df.iloc[len(data_trained):,:].values 
    dtree = DecisionTreeRegressor(max_depth=2, min_samples_leaf=0.13, random_state=3)
    print(df_train_x,data_trained)
    dtree.fit(df_train_x, data_trained['Volumes']) 
    pred_train_tree= dtree.predict(df_test_x)
    return dtree, pred_train_tree,mae(pred_train_tree,data_tested['Volumes'])
def Rf_mod(data_trained,data_tested,df):
    #df=ml_data(df)
    df_train_x,df_test_x=df.iloc[:len(data_trained),:].values,df.iloc[len(data_trained):,:].values  
    model_rf = RandomForestRegressor(n_estimators=5000, oob_score=True, random_state=100)
    model_rf.fit(df_train_x, data_trained['Volumes']) 
    pred_train_tree= model_rf.predict(df_test_x)
    return model_rf, pred_train_tree,mae(pred_train_tree,data_tested['Volumes'])

#%%
def ml_data(Data):
    
    Map={'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06'
        ,'Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'}
    try:
        Data=Data.reset_index()
    except: 
        Data=pd.DataFrame(Data)
        Data=Data.reset_index()
    
    #Data['Year'] = Data['Date'].apply(lambda x: str(20)+str(x)[-2:])
    Data['Month'] = Data['Date'].apply(lambda x: Map[str(x)[:3]])
    df = Data['Month'] 
    df = pd.get_dummies(df, columns=['Month'], drop_first=True, prefix='month')
    check_name=['month_02','month_03','month_04', 'month_05','month_06', 'month_07','month_08','month_09','month_10','month_11','month_12']
    names=df.columns
    for i in check_name:
        if i not in names:
            df[i]=np.zeros((df.shape[0],1))
    return df
def ml_data_test(start='1-1-2021',end='1-1-2023'):
    Map={'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06'
        ,'Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'}
    rev_Map = {value : key for (key, value) in Map.items()}

    per1 = pd.date_range(start =start,
           end =end, freq ='M')
    ans=[i for i in range(len(per1))]
    d={"Date":per1,'val':ans}
    df = pd.DataFrame(d)
    
    df['Year'] = df['Date'].apply(lambda x: str(x)[2:4])
    df['Month'] = df['Date'].apply(lambda x: rev_Map[str(x)[5:7]])
    del df['Date']
    df['Date'] =df['Month']+'-'+df['Year']
    del df['Month']
    del df['Year']
    df=df.set_index('Date')
    return df
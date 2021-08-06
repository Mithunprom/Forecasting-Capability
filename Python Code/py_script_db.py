import distutils.util
import os
os.chdir('C:/Users/AC91508/OneDrive - Lumen/Desktop/Forcasting_pro')
from Moldels_build_db import *
from Neural_net import *
#from NN_tunned import *
import pandas as pd
from multiprocessing import cpu_count
from joblib import Parallel,delayed,parallel_backend
from sklearn.preprocessing import MinMaxScaler
from Sarima_par import *
from Holts_par import *
from ets_par import *
Maps={'Jan':'1','Feb':'2','Mar':'3','Apr':'4','May':'5','Jun':'6'
        ,'Jul':'7','Aug':'8','Sep':'9','Oct':'10','Nov':'11','Dec':'12'}
def NA(x):
    if x=='None':
        return None
    else: return x
def All_models(data_train,data_test,Data):
    MAE={}
    Models={}
    val_array=np.zeros((40,data_test.shape[0]))
    i=0
    st,ed=Data.index[0],Data.index[-1]
    start=Maps[st[:3]]+'-'+'1-'+'20'+st[-2:]
    end=str(int(Maps[ed[:3]])+1)+'-'+'1-'+'20'+ed[-2:]
    
    if ed=='Dec-21':end=str(1)+'-'+'1-'+'20'+str(int(ed[-2:])+1)
    else:end=str(int(Maps[ed[:3]])+1)+'-'+'1-'+'20'+ed[-2:]
    
    #AR model
    mod_ar,err,lg,pred=AR_model(data_train,data_test)
    Models['AR with lags '+str(lg)]=mod_ar
    MAE['AR with lags '+str(lg)]=err
    val_array[i,:]=pred
    i+=1
    #MA model
    mod_ma,err1,lg1,pred=MA_model(data_train,data_test)
    Models['MA with lags '+str(lg1)]=mod_ma
    MAE['MA with lags '+str(lg1)]=err1
    val_array[i,:]=pred
    i+=1
    #ARMA model
    mod_arma,err2,lg21,lg22,pred=ARMA_model(data_train,data_test)
    Models['ARMA with lags ('+str(lg21)+','+str(lg22)+')']=mod_arma
    MAE['ARMA with lags ('+str(lg21)+','+str(lg22)+')']=err2
    val_array[i,:]=pred
    i+=1
    #ARIMA
    mod_arima,err3,lg31,diff3,lg32,pred=ARIMA_model(data_train,data_test)
    Models['ARIMA with order ('+str(lg31)+','+str(diff3)+','+str(lg32)+')']=mod_arima
    MAE['ARIMA with order ('+str(lg31)+','+str(diff3)+','+str(lg32)+')']=err3
    val_array[i,:]=pred
    i+=1
    #Sarima
    if __name__ == '__main__':
        p,d,q,P,D,Q,m=sarima_model(data_train,data_test)
    mod_sarima,err4,lg41,diff4,lg42,a,b,c,f,pred = SARIMA_model(data_train,data_test,p[0],d[0],q[0],P[0],D[0],Q[0],m[0])
    Models['SARMA with order ('+str(lg41)+str(diff4)+str(lg42)+'),('+str(a)+str(b)+str(c)+str(f)+')']=mod_sarima
    MAE['SARMA with order ('+str(lg41)+str(diff4)+str(lg42)+'),('+str(a)+str(b)+str(c)+str(f)+')']=err4
    val_array[i,:]=pred
    i+=1
    
    mod_sarima,err4,lg41,diff4,lg42,a,b,c,f,pred = SARIMA_model(data_train,data_test,p[1],d[1],q[1],P[1],D[1],Q[1],m[1])
    Models['SARMA with order ('+str(lg41)+str(diff4)+str(lg42)+'),('+str(a)+str(b)+str(c)+str(f)+')']=mod_sarima
    MAE['SARMA with order ('+str(lg41)+str(diff4)+str(lg42)+'),('+str(a)+str(b)+str(c)+str(f)+')']=err4
    val_array[i,:]=pred
    i+=1
    
    mod_sarima,err4,lg41,diff4,lg42,a,b,c,f,pred = SARIMA_model(data_train,data_test,p[2],d[2],q[2],P[2],D[2],Q[2],m[2])
    Models['SARMA with order ('+str(lg41)+str(diff4)+str(lg42)+'),('+str(a)+str(b)+str(c)+str(f)+')']=mod_sarima
    MAE['SARMA with order ('+str(lg41)+str(diff4)+str(lg42)+'),('+str(a)+str(b)+str(c)+str(f)+')']=err4
    val_array[i,:]=pred
    i+=1
    
    mod_sarima,err4,lg41,diff4,lg42,a,b,c,f,pred = SARIMA_model(data_train,data_test,p[3],d[3],q[3],P[3],D[3],Q[3],m[3])
    Models['SARMA with order ('+str(lg41)+str(diff4)+str(lg42)+'),('+str(a)+str(b)+str(c)+str(f)+')']=mod_sarima
    MAE['SARMA with order ('+str(lg41)+str(diff4)+str(lg42)+'),('+str(a)+str(b)+str(c)+str(f)+')']=err4
    val_array[i,:]=pred
    i+=1
    
    mod_sarima,err4,lg41,diff4,lg42,a,b,c,f,pred = SARIMA_model(data_train,data_test,p[4],d[4],q[4],P[4],D[4],Q[4],m[4])
    Models['SARMA with order ('+str(lg41)+str(diff4)+str(lg42)+'),('+str(a)+str(b)+str(c)+str(f)+')']=mod_sarima
    MAE['SARMA with order ('+str(lg41)+str(diff4)+str(lg42)+'),('+str(a)+str(b)+str(c)+str(f)+')']=err4
    val_array[i,:]=pred
    i+=1

    #Holt's linear trend
    mod_exp_add,error06,pred=exp_add(data_train,data_test)
    Models["Holts ES with add trend"]=mod_exp_add
    MAE["Holts ES with add trend"]= error06
    val_array[i,:]=pred
    i+=1
    #Holt-linear trend with dmaped
    mod_exp_add_ds,error06,pred=mod_exp_add_d(data_train,data_test)
    Models["Holts ES with add damped trend"]=mod_exp_add_ds
    MAE["Holts ES with add damped trend"]= error06
    val_array[i,:]=pred
    i+=1
    ##Holt's exponential  trend model
    exp_muls,error07,pred=exp_mul(data_train,data_test)
    Models["Holts ES with mul trend"]=exp_muls
    MAE["Holts ES with mul trend" ]= error07
    val_array[i,:]=pred
    i+=1
    ##Holt's exponential  trend model with damped
    exp_mul_dampeds,error07,pred=exp_mul_damped(data_train,data_test)
    Models["Holts ES with mul damped trend"]=exp_mul_dampeds
    MAE["Holts ES with mul damped trend" ]= error07
    val_array[i,:]=pred
    i+=1
    ##Holt-Winter’s Seasonal Smoothing additive  model no dmaped
    exp_add_ssns,error06,sp,pred=exp_add_ssn(data_train,data_test)
    Models["Holts ES with add trend & seasonality: " +str(sp)+' periods']=exp_add_ssns
    MAE["Holts ES with add trend & seasonality: " +str(sp)+' periods']= error06
    val_array[i,:]=pred
    i+=1
    #Holt-Winter’s Seasonal Smoothing additive model with damped 
    exp_add_ssn_dampeds,error07,sp,pred=exp_add_ssn_damped(data_train,data_test)
    Models["Holts ES with add damped trend and seasonality of " +str(sp)+' periods']=exp_add_ssn_dampeds
    MAE["Holts ES with add damped trend & seasonality: " +str(sp)+' periods']= error07
    val_array[i,:]=pred
    i+=1
    #Holt-Winter’s Seasonal Smoothing multiplicative  model no dmaped
    exp_mul_ssns,error08,sp,pred=exp_mul_ssn(data_train,data_test)
    Models["Holts ES with mul trend and seasonality of " +str(sp)+' periods']=exp_mul_ssns
    MAE["Holts ES with mul trend & seasonality: " +str(sp)+' periods']= error08
    val_array[i,:]=pred
    i+=1
    #Holt-Winter’s Seasonal Smoothing multiplicative model with damped 
    exp_mul_ssn_dampeds,error09,sp,pred=exp_mul_ssn_damped(data_train,data_test)
    Models["Holts ES with mul damped trend & seasonality: " +str(sp)+' periods']=exp_mul_ssn_dampeds
    MAE["Holts ES with mul damped trend & seasonality: " +str(sp)+' periods']= error09
    val_array[i,:]=pred
    i+=1
    #holts tunned model    
    if __name__ == '__main__':
        res=sun_holts(data_train,data_test)
        t,d,s,sp,bc,rb=[],[],[],[],[],[]
        for i in range(len(res)):
            t.append(res[i][0])
            d.append(res[i][1])
            s.append(res[i][2])
            sp.append(res[i][3])
            bc.append(res[i][4])
            rb.append(res[i][5])
        t=[NA(i) for i in t]
        s=[NA(i) for i in s]
        d=[bool(distutils.util.strtobool(i)) for i in d]
        sp=[int(i) for i in sp]
        bc=[bool(distutils.util.strtobool(i)) for i in bc]
        rb=[bool(distutils.util.strtobool(i)) for i in rb]
        
    tunned_holtss,error011,pred=tunned_holts(data_train,data_test,t[0],d[0],s[0],sp[0],bc[0],rb[0])
    Models["Holts tunned model_1: "]=tunned_holtss
    MAE["Holts tunned model_1"]=error011
    val_array[i,:]=pred
    i+=1
    tunned_holtss,error011,pred=tunned_holts(data_train,data_test,t[1],d[1],s[1],sp[1],bc[2],rb[1])
    Models["Holts tunned model_2: "]=tunned_holtss
    MAE["Holts tunned model_2"]=error011
    val_array[i,:]=pred
    i+=1
    tunned_holtss,error011,pred=tunned_holts(data_train,data_test,t[2],d[2],s[2],sp[2],bc[2],rb[2])
    Models["Holts tunned model_3: "]=tunned_holtss
    MAE["Holts tunned model_3"]=error011
    val_array[i,:]=pred
    i+=1
    #Simple Exponential Smoothing (SES)
    Sexp_models,error10,pred=Sexp_model(data_train,data_test)
    Models["Simple Exponential Smoothing"]=Sexp_models
    MAE["Simple Exponential Smoothing"]=error10
    val_array[i,:]=pred
    i+=1
    #ETS model
    #ETS
    if __name__ == '__main__':
        e,t,s,d,p=[],[],[],[],[]
        res=sun_ets(data_train,data_test)
        for i in range(len(res)):
            e.append(res[i][0])
            t.append(res[i][1])
            s.append(res[i][2])
            d.append(res[i][3])
            p.append(res[i][4])
        e=[NA(i) for i in e]
        t=[NA(i) for i in t]
        s=[NA(i) for i in s]
        d=[bool(distutils.util.strtobool(i)) for i in d]
        p=[int(i) for i in p]
    ets_model,error12,pred=ETS_model(data_train,data_test,e[0],t[0],s[0],p[0],d[0])
    Models["ETS_1"]=ets_model
    MAE["ETS_1"]=error12
    val_array[i,:]=pred
    i+=1
    
    ets_model,error12,pred=ETS_model(data_train,data_test,e[1],t[1],s[1],p[1],d[1])
    Models["ETS_2"]=ets_model
    MAE["ETS_2"]=error12
    val_array[i,:]=pred
    i+=1
    
    ets_model,error12,pred=ETS_model(data_train,data_test,e[2],t[2],s[2],p[2],d[2])
    Models["ETS_3"]=ets_model
    MAE["ETS_3"]=error12
    val_array[i,:]=pred
    i+=1
    
    scaler=MinMaxScaler(feature_range=(-1, 1))
    scaler=scaler.fit(data_train.values.reshape(-1,1))
    data_train_n=scaler.transform(data_train.values.reshape(-1,1))
    data_test_n=scaler.transform(data_test.values.reshape(-1,1))
    
    #NN
    model_nn,error,pred=dense(data_train_n,data_test_n,sps=1,len_test=len(data_test))
    Models['NN-1']=model_nn
    MAE['NN-1']=error
    val_array[i,:]=scaler.inverse_transform(np.array(pred).reshape(-1,1)).reshape(1,-1)
    i+=1
    #LSTM
    model_lstm,error,pred=lstm(data_train_n,data_test_n,sps=1,len_test=len(data_test))
    Models['LSTM-1']=model_lstm
    MAE['LSTM-1']=error
    val_array[i,:]=scaler.inverse_transform(np.array(pred).reshape(-1,1)).reshape(1,-1)
    i+=1
    #NN
    model_nn,error,pred=dense(data_train_n,data_test_n,sps=2,len_test=len(data_test))
    Models['NN-2']=model_nn
    MAE['NN']=error
    val_array[i,:]=scaler.inverse_transform(np.array(pred).reshape(-1,1)).reshape(1,-1)
    i+=1
    #CNN
    model_cnn,error,pred=cnn(data_train_n,data_test_n,sps=2,len_test=len(data_test))
    Models['CNN-2']=model_cnn
    MAE['CNN-2']=error
    val_array[i,:]=scaler.inverse_transform(np.array(pred).reshape(-1,1)).reshape(1,-1)
    i+=1
    #LSTM
    model_lstm,error,pred=lstm(data_train_n,data_test_n,sps=2,len_test=len(data_test))
    Models['LSTM-2']=model_lstm
    MAE['LSTM-2']=error
    val_array[i,:]=scaler.inverse_transform(np.array(pred).reshape(-1,1)).reshape(1,-1)
    i+=1
    #NN
    model_nn,error,pred=dense(data_train_n,data_test_n,sps=3,len_test=len(data_test))
    Models['NN-3']=model_nn
    MAE['NN-3']=error
    val_array[i,:]=scaler.inverse_transform(np.array(pred).reshape(-1,1)).reshape(1,-1)
    i+=1
    #CNN
    model_cnn,error,pred=cnn(data_train_n,data_test_n,sps=3,len_test=len(data_test))
    Models['CNN-3']=model_cnn
    MAE['CNN-3']=error
    val_array[i,:]=scaler.inverse_transform(np.array(pred).reshape(-1,1)).reshape(1,-1)
    i+=1
    #LSTM
    model_lstm,error,pred=lstm(data_train_n,data_test_n,sps=3,len_test=len(data_test))
    Models['LSTM-3']=model_lstm
    MAE['LSTM-3']=error
    val_array[i,:]=scaler.inverse_transform(np.array(pred).reshape(-1,1)).reshape(1,-1)
    i+=1
    #NN
    model_nn,error,pred=dense(data_train_n,data_test_n,sps=4,len_test=len(data_test))
    Models['NN-4']=model_nn
    MAE['NN']=error
    val_array[i,:]=scaler.inverse_transform(np.array(pred).reshape(-1,1)).reshape(1,-1)
    i+=1
    #CNN
    model_cnn,error,pred=cnn(data_train_n,data_test_n,sps=4,len_test=len(data_test))
    Models['CNN-4']=model_cnn
    MAE['CNN-4']=error
    val_array[i,:]=scaler.inverse_transform(np.array(pred).reshape(-1,1)).reshape(1,-1)
    i+=1
    #LSTM
    model_lstm,error,pred=lstm(data_train_n,data_test_n,sps=4,len_test=len(data_test))
    Models['LSTM-4']=model_lstm
    MAE['LSTM-4']=error
    val_array[i,:]=scaler.inverse_transform(np.array(pred).reshape(-1,1)).reshape(1,-1)
    i+=1
    
    #LSTM-CNN
    model_lcn,error,pred=lstm_cnn(data_train_n,data_test_n,sps=4,len_test=len(data_test))
    Models['CNN-LSTM-4']=model_lcn
    MAE['CNN-LSTM-4']=error
    val_array[i,:]=scaler.inverse_transform(np.array(pred).reshape(-1,1)).reshape(1,-1)
    i+=1
    
    #Prophet
    model_pr,error,pred=prophet_model(Data)
    Models['prophet']=model_pr
    MAE['prophet']=error
    val_array[i,:]=pred.reshape(1,-1)
    i+=1
    
    #GPy
    dtt1=ml_data_test(start=start,end=end)
    data_ts=ml_data(dtt1)
    model_gpy,error,pred=gp_mod(data_train,data_test,data_ts)
    Models['GPy']=model_gpy
    MAE['GPy']=error
    val_array[i,:]=(np.array(pred).reshape(1,-1))
    i+=1
    
    #CART
    model_dt,error,pred=Cart_mod(data_train,data_test,data_ts)
    Models['RT']=model_dt
    MAE['RT']=error
    val_array[i,:]=(np.array(pred).reshape(1,-1))
    i+=1
    
    #RF
    model_rf,error,pred=Rf_mod(data_train,data_test,data_ts)
    Models['RF']=model_rf
    MAE['RF']=error
    val_array[i,:]=(np.array(pred).reshape(1,-1))
    i+=1
    
    print(MAE)
    return Models,MAE,val_array
#if __name__ == '__main__':
#    prudct_A_x_models,prudct_A_x_mad,val_array=All_models(data_train,data_test)

def get_forcasts(models,data_train,data_test,start,steps=24):
    names=list(models.keys())
    Res={}
    Flag={}
    ed=data_test.index[0]
    starts=str(int(Maps[ed[:3]]))+'-'+'1-'+'20'+ed[-2:]
    sts=int(Maps[ed[:3]])
    eds=2
    '''
    if int(Maps[ed[:3]])==12:
        eds=3
    else: 
        eds=2
    '''
    ends=str(sts)+'-'+'1-'+'20'+str(int(ed[-2:])+eds)
    
    dtt1=ml_data_test(start=starts,end=ends)
    data_ts=ml_data(dtt1)
    
    i=0
    #AR
    Res[names[i]]=models[names[i]].predict(start=start, end=start+steps)
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1
    #MA
    Res[names[i]]=models[names[i]].predict(start=start, end=start+steps)
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1    
    #ARMA
    Res[names[i]]=models[names[i]].predict(start=start, end=start+steps)
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1
    #ARIMA
    Res[names[i]]=models[names[i]].predict(start=start, end=start+steps)
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1
    #SARIMA
    Res[names[i]]=models[names[i]].predict(start=start, end=start+steps)
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1
    #SARIMA
    Res[names[i]]=models[names[i]].predict(start=start, end=start+steps)
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1
    #SARIMA
    Res[names[i]]=models[names[i]].predict(start=start, end=start+steps)
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1
    #SARIMA
    Res[names[i]]=models[names[i]].predict(start=start, end=start+steps)
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1
    #SARIMA
    Res[names[i]]=models[names[i]].predict(start=start, end=start+steps)
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1
    '''
    #VAR
    Res[names[5]]=models[names[5]].forecast(models[names[5]].y, steps=steps+1)[:,0]
    #VARMA
    Res[names[6]]=models[names[6]].forecast(steps=steps+1)[:,0]
    '''
    #add_Linear
    Res[names[i]]=models[names[i]].forecast(steps=steps+1)
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1
    #add_Linear_dampen
    Res[names[i]]=models[names[i]].forecast(steps=steps+1)
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1
    #mul_Linear
    Res[names[i]]=models[names[i]].forecast(steps=steps+1)
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1
    #mul_Linear_dampen
    Res[names[i]]=models[names[i]].forecast(steps=steps+1)
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1
    #add_Linear_ssn
    Res[names[i]]=models[names[i]].forecast(steps=steps+1)
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1
    #add_linear_dampen_ssn
    Res[names[i]]=models[names[i]].forecast(steps=steps+1)
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1
    #mul_linear_ssn
    Res[names[i]]=models[names[i]].forecast(steps=steps+1)
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1
    #mul_linear_dampen_ssn
    Res[names[i]]=models[names[i]].forecast(steps=steps+1)
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1
    #holt's_tunned_model
    Res[names[i]]=models[names[i]].predict(start=start, end=start+steps)
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1
    #holt's_tunned_model
    Res[names[i]]=models[names[i]].predict(start=start, end=start+steps)
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1
    #holt's_tunned_model
    Res[names[i]]=models[names[i]].predict(start=start, end=start+steps)
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1
    #SES
    Res[names[i]]=models[names[i]].predict(start=start, end=start+steps)
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1
    #ETS
    Res[names[i]]=models[names[i]].forecast(steps=steps+1)
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1
    #ETS
    Res[names[i]]=models[names[i]].forecast(steps=steps+1)
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1
    #ETS
    Res[names[i]]=models[names[i]].forecast(steps=steps+1)
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1
    
    
    scaler=MinMaxScaler(feature_range=(-1, 1))
    scaler=scaler.fit(data_train.values.reshape(-1,1))
    data_train_n=scaler.transform(data_train.values.reshape(-1,1))
    data_test_n=scaler.transform(data_test.values.reshape(-1,1))
    
    
    #Dense
    Res[names[i]]= scaler.inverse_transform(np.array(Forcast(models[names[i]],data_test_n,steps=steps+1,method="NN",sps=1)).reshape(-1,1))
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1
    #CNN
    #Res[names[i]]=np.array(Forcast(models[names[i]],data_test,steps=steps+1,method="CNN",sps=1))
    #i+=1
    #LSTM
    Res[names[i]]=scaler.inverse_transform(np.array(Forcast(models[names[i]],data_test_n,steps=steps+1,method='LSTM',sps=1)).reshape(-1,1))
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1
    #Dense
    Res[names[i]]=scaler.inverse_transform(np.array(Forcast(models[names[i]],data_test_n,steps=steps+1,method="NN",sps=2)).reshape(-1,1))
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1
    #CNN
    Res[names[i]]=scaler.inverse_transform(np.array(Forcast(models[names[i]],data_test_n,steps=steps+1,method="CNN",sps=2)).reshape(-1,1))
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1
    #LSTM
    Res[names[i]]=scaler.inverse_transform(np.array(Forcast(models[names[i]],data_test_n,steps=steps+1,method='LSTM',sps=2)).reshape(-1,1))
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]

    i+=1
    #Dense
    Res[names[i]]=scaler.inverse_transform(np.array(Forcast(models[names[i]],data_test_n,steps=steps+1,method="NN",sps=3)).reshape(-1,1))
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]

    i+=1
    #CNN
    Res[names[i]]=scaler.inverse_transform(np.array(Forcast(models[names[i]],data_test_n,steps=steps+1,method="CNN",sps=3)).reshape(-1,1))
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1
    #LSTM
    Res[names[i]]=scaler.inverse_transform(np.array(Forcast(models[names[i]],data_test_n,steps=steps+1,method='LSTM',sps=3)).reshape(-1,1))
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1
    #Dense
    Res[names[i]]=scaler.inverse_transform(np.array(Forcast(models[names[i]],data_test_n,steps=steps+1,method="NN",sps=4)).reshape(-1,1))
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1
    #CNN
    Res[names[i]]=scaler.inverse_transform(np.array(Forcast(models[names[i]],data_test_n,steps=steps+1,method="CNN",sps=4)).reshape(-1,1))
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1
    #LSTM
    Res[names[i]]=scaler.inverse_transform(np.array(Forcast(models[names[i]],data_test_n,steps=steps+1,method='LSTM',sps=4)).reshape(-1,1))
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1
    #CNN-LSTM
    Res[names[i]]=scaler.inverse_transform(np.array(Forcast(models[names[i]],data_test_n,steps=steps+1,method="LSTM_CNN",sps=4)).reshape(-1,1))
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1
    #Prophet
    dt=models[names[i]].make_future_dataframe(periods=steps+1, freq = "M",include_history=False)
    Res[names[i]]=models[names[i]].predict(dt)['yhat']
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1
    #Gpy
    
    #dt1=np.linspace(data_train.shape[0],data_train.shape[0]+steps,steps+1).reshape(-1,1)
    
    
    dtt1=ml_data_test(start=starts,end=ends)
    data_ts=ml_data(dtt1)
    Res[names[i]]=models[names[i]].predict(data_ts)
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1
    #CART
    Res[names[i]]=models[names[i]].predict(data_ts)
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1
    #RF
    Res[names[i]]=models[names[i]].predict(data_ts)
    #Flag[names[i]]=[sum(1 for number in Res[names[i]] if number < 0)<0 for _ in range(len(Res[names[i]]))]
    i+=1
    return Res,[starts,ends]#,Flag

def Assert(x,y):
    for i in range(len(x)):
        if x[i]==y[i]:
            continue
        else:
            False
    return True
#assert val_array[0,:==]
def get_excel_pred(prudct_A_x_models,prudct_A_x_mad,val_array,data_train,data_test,start,steps=24):
    #Res,Flag=get_forcasts(prudct_A_x_models,data_train,data_test,start,steps=steps-1)
    Res,send=get_forcasts(prudct_A_x_models,data_train,data_test,start,steps=steps-1)
    names=list(prudct_A_x_models.keys())
    n_methods=len(names)
    array=np.zeros((n_methods,steps))
    #flag_array=np.zeros((n_methods,steps))
    for i in range(n_methods):
        try:
            array[i,:]=Res[names[i]].values.reshape(1,-1)
            #flag_array[i,:]=Flag[names[i]].values
            print(Assert(val_array[i,:],array[i,:len(data_test)]))
            print('ok: '+str(i))
        except Exception as e:
            array[i,:]=Res[names[i]].reshape(1,-1)
            #flag_array[i,:]=Flag[names[i]]
            print(Assert(val_array[i,:],array[i,:len(data_test)]))
            print('ok: '+str(i))
    
    #df = pd.DataFrame(array)
    #df_flag=pd.DataFrame(flag_array)
    #df.to_excel(excel_writer = "array.xlsx", sheet_name=sname)
    #df_flag.to_excel(excel_writer = "flag_array.xlsx", sheet_name=sname)
    return array,send#,flag_array
#if __name__ == '__main__':
#    prudct_A_x_models,prudct_A_x_mad,val_array=All_models(data_train,data_test)

def par_loop(tableResult):
    scenario=tableResult['Scenario'].unique()
    product=tableResult.loc[tableResult['Scenario']==scenario[0]]['Product'].unique()
    subproduct=tableResult.loc[(tableResult['Scenario']==scenario[0]) \
            & (tableResult['Product']==product[0])]['Sub-Product'].unique()

    Data=tableResult.loc[(tableResult['Scenario']==scenario[0]) \
            & (tableResult['Product']==product[0])\
            & (tableResult['Sub-Product']==subproduct[0])][['Date_time','Date','Volumes']]
    #page.append(names)    
    Data=Data.sort_values(by='Date_time')
    Data.reset_index(drop=True)
    Data=Data[['Date','Volumes']]
    Data=Data.set_index('Date')
    Colnames=Data.columns.values.tolist()
    Database=[]
    metadata=[]
    Deleted_dat=[]
    for isc in range(len(scenario)):
        product=tableResult.loc[tableResult['Scenario']==scenario[isc]]['Product'].unique()
        for jpr in range(len(product)):
            subproduct=tableResult.loc[(tableResult['Scenario']==scenario[isc]) \
            & (tableResult['Product']==product[jpr])]['Sub-Product'].unique()
            for ksub in range(len(subproduct)):
                print([scenario[isc],product[jpr],subproduct[ksub]])
                Data=tableResult.loc[(tableResult['Scenario']==scenario[isc]) \
                                    & (tableResult['Product']==product[jpr])\
                                        & (tableResult['Sub-Product']==subproduct[ksub])][['Date_time','Date','Volumes']]
               #page.append(names)    
                Data=Data.sort_values(by='Date_time')
                Data.reset_index(drop=True)
                Data=Data[['Date','Volumes']]
                Data=Data.set_index('Date')
                Ln=int(len(Data)*.8)
                data_train,data_test=Data[:Ln],Data[Ln:]
                print('train data shape  {} and test shape is {}'.format(data_train.shape[0],data_test.shape[0]))
                if len(data_train)<12:
                   Deleted_dat.append([scenario[isc],product[jpr],subproduct[ksub]])
                   continue
                if __name__ == '__main__':
                   prudct_A_x_models,prudct_A_x_mad,val_array=All_models(data_train,data_test,Data)
                   arr,send=get_excel_pred(prudct_A_x_models,prudct_A_x_mad,val_array,data_train,data_test,start=len(data_train),steps=24)
                   Database.append(arr)
                   metadata.append([scenario[isc],product[jpr],subproduct[ksub],send,data_test.shape[0]])

    return Database,metadata,Deleted_dat

import pyodbc 
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=usodcvsql0255;'
                      'Database=NAO_PMO_Analytics;'
                      'Trusted_Connection=yes;')

cursor = conn.cursor()

tableResult = pd.read_sql("SELECT * FROM [NAO_PMO_Analytics].[dbo].[zzz_Forecast_Input]", conn) 

df=tableResult['Date'].values
def mod_data(df):
    Map={'01':'Jan','02':'Feb','03':'Mar','04':'Apr','05':'May','06':'Jun'
        ,'07':'Jul','08':'Aug','09':'Sep','10':'Oct','11':'Nov','12':'Dec'}
    arr=[]
    arr1=[]
    for i in range(len(df)):
        yr=str(df[i])[2:4]
        mn=str(df[i])[4:6]
        dt=str(df[i])[6:]
        map_mn=Map[mn]
        arr.append(map_mn+'-'+yr)
        arr1.append(str(df[i])[:4]+'-'+mn+'-'+dt)
    return np.array(arr),np.array(arr1)

tableResult['Date'],tableResult['Date_time']=mod_data(df)
tableResult['Date_time']=tableResult['Date_time'].apply(pd.to_datetime)
'''
tableResult.columns
scenario=tableResult['Scenario'].unique()
product=tableResult.loc[tableResult['Scenario']==scenario[0]]['Product'].unique()
subproduct=tableResult.loc[(tableResult['Scenario']==scenario[0]) \
            & (tableResult['Product']==product[0])]['Sub-Product'].unique()

data=tableResult.loc[(tableResult['Scenario']==scenario[0]) \
            & (tableResult['Product']==product[0])\
            & (tableResult['Sub-Product']==subproduct[0])][['Date_time','Date','Volumes']]
'''
dbase,metadat,Deleted_dat=par_loop(tableResult)
print(dbase,metadat)
#%%
#from sql_conn import *
import dill                            #pip install dill --user

import pickle
with open('dbase', 'wb') as f:
    pickle.dump(dbase, f)
with open('metadata', 'wb') as f:
    pickle.dump(metadat, f)    
with open('deleted_data', 'wb') as f:
    pickle.dump(Deleted_dat, f)     
with open('tableResult', 'wb') as f:
    pickle.dump(tableResult, f) 

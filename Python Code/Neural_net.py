# univariate cnn-lstm example
from numpy import array
from sklearn.metrics import mean_absolute_error as mae
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

def mad(pred,data_test):
    #MEAN=(np.mean(data_train)+np.mean(data_test))/2
    MEAN=np.mean(data_test)
    mad=0
    for i in range(len(pred)):
        mad+=abs(MEAN-pred[i])
    return mad/len(pred)

# define dataset
def Forcast(model,data_train,steps,method,sps=4):
        X,y=datamod(data_train,sps)
        m=len(y)
        for i in range(steps):
            x=X[-1][0][1:]
            x=np.append(x,y[-1])
            X+=[[x]]
            #X.append(x)
            if method=='LSTM_CNN':
                x = x.reshape((1, 2, 2, 1))
                y.append(model.predict(x, verbose=0)[0][0])
            if method=='NN':
                x = x.reshape((1, sps))
                y.append(model.predict(x, verbose=0)[0][0])
            if method=='CNN':
                x = x.reshape((1,sps,1))
                y.append(model.predict(x, verbose=0)[0][0])
            if method=='LSTM':
                try:
                    x = x.reshape((1,sps,1))
                    y.append(model.predict(x, verbose=0)[0][0])
                except Exception as e:
                    x = x.reshape((1,sps,))
                    y.append(model.predict(x, verbose=0)[0][0])
        return y[m:]
    
    
def datamod(data,n):
        try:data=data.values
        except:pass
        x=[]
        y=[]
        for i in range(len(data)-n):
            x.append([data[i:i+n]])
            y.append(data[i+n])
        return x,y
def lstm_cnn(data_train,data_test,sps=4,len_test=6):
    X,y=datamod(data_train,sps)
    X,y=np.array(X),np.array(y)
    #X=X..reshape(X.shape[0],X.shape[2])
    # reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
    if sps==4:
        X = X.reshape((X.shape[0], 2, 2, 1))
    #X_test = X_test.reshape((X_test.shape[0], 2, 2, 1))
    
    # define model
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=16, kernel_size=1, activation='relu'), input_shape=(None, 2, 1)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(20, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mae')
    # fit model
    model.fit(X, y, epochs=500, batch_size=1,verbose=0)
    
    yhat = Forcast(model,data_train, steps=len_test,method="LSTM_CNN",sps=sps)
    print('LSTM+CNN Net with MAE is: '+str(mae(yhat,data_test)))
    return model, mae(yhat,data_test),yhat
#model_nn,error=Net(data_train,data_test)
#Models['CNN-LSTM']=model_nn
#MAE['CNN-LSTM']=error
def dense(data_train,data_test,sps=4,len_test=6):
    X,y=datamod(data_train,sps)
    X,y=np.array(X),np.array(y)
    X=X.reshape(X.shape[0],X.shape[2])
    model = Sequential()
    model.add(Dense(50, activation='relu', input_dim=sps))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mae')
    # fit model
    model.fit(X, y, epochs=2000, verbose=0)
    yhat = Forcast(model,data_train, steps=len_test,method='NN',sps=sps)
    print('Neural Net with MAD is: '+str(mae(yhat,data_test)))
    return model, mae(yhat,data_test),yhat

def cnn(data_train,data_test,sps=4,len_test=6):
    X,y=datamod(data_train,sps)
    X,y=np.array(X),np.array(y)
    #X=X..reshape(X.shape[0],X.shape[2])
    # reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[2], 1))
    # define model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(sps, 1)))
    if sps>3:model.add(MaxPooling1D(pool_size=2))
    else: model.add(MaxPooling1D(pool_size=1))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mae')
    # fit model
    model.fit(X, y, epochs=1000, verbose=0)
    # demonstrate prediction
    yhat = Forcast(model,data_train, steps=len_test,method='CNN',sps=sps)
    print('CNN with MAD is: '+str(mae(yhat,data_test)))
    return model, mae(yhat,data_test),yhat
def lstm(data_train,data_test,sps=4,len_test=6):
    import tensorflow as tf
    X,y=datamod(data_train,sps)
    X,y=np.array(X),np.array(y)
    #X=X..reshape(X.shape[0],X.shape[2])
    # reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[2], 1))
    X=tf.convert_to_tensor(X, dtype=None, dtype_hint=None, name=None)
    # define model
    model = Sequential()
    model.add(LSTM(256, activation='relu', input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mae')
    # fit model
    model.fit(X, y, epochs=1000, verbose=0)
    # demonstrate prediction
    yhat = Forcast(model,data_train, steps=len_test,method='LSTM',sps=sps)
    print('LSTM with MAD is: '+str(mae(yhat,data_test)))
    return model, mae(yhat,data_test),yhat
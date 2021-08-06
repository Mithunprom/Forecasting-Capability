
# univariate cnn-lstm example
from numpy import array
from sklearn.metrics import mean_absolute_error as mae
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense,Dropout
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.optimizers import RMSprop,SGD,Adadelta,Adagrad,Adam
from keras.wrappers.scikit_learn import KerasRegressor
from keras.constraints import maxnorm
from sklearn.model_selection import GridSearchCV
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
seed = 7
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.autograph.set_verbosity(3)

np.random.seed(seed)

def tensorflow_shutup():
    """
    Make Tensorflow less verbose
    """
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # noinspection PyPackageRequirements
        import tensorflow as tf
        from tensorflow.python.util import deprecation

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        # Monkey patching deprecation utils to shut it up! Maybe good idea to disable this once after upgrade
        # noinspection PyUnusedLocal
        def deprecated(date, instructions, warn_once=True):  # pylint: disable=unused-argument
            def deprecated_wrapper(func):
                return func
            return deprecated_wrapper

        deprecation.deprecated = deprecated

    except ImportError:
        pass
tensorflow_shutup()
'''
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam']
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
weight_constraint = [1, 2, 3, 4, 5]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4]
neurons = [1, 5, 10, 15, 20, 25, 30]
# define the grid search parameters
batch_size = [4, 12, 16,20]
epochs = [10, 50, 100,200,500]
'''

# define dataset
def Forcast(model,data_trained,steps,method,sps=4):
        X,y=datamod(data_trained,sps)
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
        except: pass
        x=[]
        y=[]
        for i in range(len(data)-n):
            x.append([data[i:i+n]])
            y.append(data[i+n])
        return x,y
def lstm_cnn_tuned(data_trained,data_tested,sps=4,len_test=6):
    optimizer = ['Adam']
    learn_rate = [0.0001, 0.001, 0.01]
    #momentum = [0]
    #init_mode = ['uniform',  'normal']
    activation = ['relu', 'tanh']
    #weight_constraint = [0]
    #dropout_rate = [0.0,.15]
    neurons = [16,32,64,120]
    # define the grid search parameters
    #batch_size = [1,10]
    epochs = [500]
    def create_lcm_model(optimizer='adam',learn_rate=0.01, momentum=0,init_mode='uniform',
                 activation='relu',dropout_rate=0.0, weight_constraint=0,neurons=1):
    # create model
        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=16, kernel_size=1, activation=activation,#kernel_initializer=init_mode,kernel_constraint=maxnorm(weight_constraint),
                                   input_shape=(None, 2, 1))))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(neurons, activation=activation))#,kernel_initializer=init_mode
        #model.add(Dropout(dropout_rate))
        model.add(Dense(1,activation='linear'))#kernel_initializer=init_mode,
        # Compile model
        if optimizer == 'SGD':
            optimizer = SGD(lr=learn_rate, momentum=momentum)
        if optimizer == 'RMSprop':
            optimizer = RMSprop(lr=learn_rate, momentum=momentum)
        if optimizer == 'Adagrad':
            optimizer = Adagrad(lr=learn_rate)
        if optimizer == 'Adadelta':
            optimizer = Adadelta(lr=learn_rate)
        if optimizer == 'Adam':
            optimizer = Adam(lr=learn_rate)
        model.compile(loss='mae', optimizer=optimizer, metrics=['accuracy'])
        return model
    
    X,y=datamod(data_trained,sps)
    X,y=np.array(X,dtype='float32'),np.array(y,dtype='float32')
    #X=X..reshape(X.shape[0],X.shape[2])
    # reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
    if sps==4:
        X = X.reshape((X.shape[0], 2, 2, 1))
    #X_test = X_test.reshape((X_test.shape[0], 2, 2, 1))
    
    param_grid = dict(epochs=epochs,optimizer=optimizer,
                  learn_rate=learn_rate, 
                      #momentum=momentum,init_mode=init_mode,
                  activation=activation,
                      #dropout_rate=dropout_rate, 
                  #weight_constraint=weight_constraint,batch_size=batch_size, 
                      neurons=neurons)
    
    
    model = KerasRegressor(build_fn=create_lcm_model, verbose=0)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=20, cv=3)
    grid_result = grid.fit(X, y)
    activation=grid_result.best_params_['activation']
    #batch_size=grid_result.best_params_['batch_size']
    #dropout_rate=grid_result.best_params_['dropout_rate']
    epochs=grid_result.best_params_['epochs']
    #init_mode=grid_result.best_params_['init_mode']
    learn_rate=grid_result.best_params_['learn_rate']
    #momentum=grid_result.best_params_['momentum']
    neurons=grid_result.best_params_['neurons']
    optimizer=grid_result.best_params_['optimizer']
    #weight_constraint=grid_result.best_params_['weight_constraint']

    model=create_lcm_model(optimizer=optimizer,learn_rate=learn_rate,# momentum=momentum,
                       #init_mode=init_mode,
                            activation=activation,
                       #dropout_rate=dropout_rate, weight_constraint=weight_constraint,
                       neurons=neurons)
    model.fit(X, y, epochs=epochs, verbose=0)#batch_size=batch_size
    
    yhat = Forcast(model,data_trained, steps=len_test,method="LSTM_CNN",sps=sps)
    print('LSTM+CNN Net with MAE is: '+str(mae(yhat,data_tested)))
    return model, mae(yhat,data_tested),yhat
#model_nn,error=Net(data_trained,data_tested)
#Models['CNN-LSTM']=model_nn
#MAE['CNN-LSTM']=error
def dense_tuned(data_trained,data_tested,sps=4,len_test=6):
    optimizer = ['Adam']
    learn_rate = [0.0001, 0.001, 0.01]
    #momentum = [0]
    #init_mode = ['uniform',  'normal']
    activation = ['relu', 'tanh']
    #weight_constraint = [0]
    #dropout_rate = [0.0,.15]
    neurons = [16,32,64,120]
    # define the grid search parameters
    #batch_size = [1,10]
    epochs = [500]
    def create_model(optimizer='adam',learn_rate=0.01, momentum=0,init_mode='uniform',
                 activation='relu',dropout_rate=0.0, weight_constraint=0,neurons=1):
    # create model
        model = Sequential()
        model.add(Dense(neurons, input_dim=sps, #kernel_initializer=init_mode,
                        activation=activation)) #,kernel_constraint=maxnorm(weight_constraint)))
        #model.add(Dropout(dropout_rate))
        model.add(Dense(1, kernel_initializer=init_mode,activation='linear'))
        # Compile model
        if optimizer == 'SGD':
            optimizer = SGD(lr=learn_rate, momentum=momentum)
        if optimizer == 'RMSprop':
            optimizer = RMSprop(lr=learn_rate, momentum=momentum)
        if optimizer == 'Adagrad':
            optimizer = Adagrad(lr=learn_rate)
        if optimizer == 'Adadelta':
            optimizer = Adadelta(lr=learn_rate)
        if optimizer == 'Adam':
            optimizer = Adam(lr=learn_rate)
        model.compile(loss='mae', optimizer=optimizer, metrics=['accuracy'])
        return model
    X,y=datamod(data_trained,sps)
    X,y=np.array(X),np.array(y)
    X=X.reshape(X.shape[0],X.shape[2])
    param_grid = dict(epochs=epochs,optimizer=optimizer,
                  learn_rate=learn_rate, 
                      #momentum=momentum,init_mode=init_mode,
                  activation=activation,
                      #dropout_rate=dropout_rate, 
                  #weight_constraint=weight_constraint,batch_size=batch_size, 
                      neurons=neurons)
 
    
    model = KerasRegressor(build_fn=create_model, verbose=0)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=20, cv=3)
    grid_result = grid.fit(X, y)
    print("#######################################################################")
    activation=grid_result.best_params_['activation']
    #batch_size=grid_result.best_params_['batch_size']
    #dropout_rate=grid_result.best_params_['dropout_rate']
    epochs=grid_result.best_params_['epochs']
    #init_mode=grid_result.best_params_['init_mode']
    learn_rate=grid_result.best_params_['learn_rate']
    #momentum=grid_result.best_params_['momentum']
    neurons=grid_result.best_params_['neurons']
    optimizer=grid_result.best_params_['optimizer']
    #weight_constraint=grid_result.best_params_['weight_constraint']

    model=create_model(optimizer=optimizer,learn_rate=learn_rate,# momentum=momentum,
                       #init_mode=init_mode,
                            activation=activation,
                       #dropout_rate=dropout_rate, weight_constraint=weight_constraint,
                       neurons=neurons)
    model.fit(X, y, epochs=epochs, verbose=0)#batch_size=batch_size
    # pred of model
    yhat = Forcast(model,data_trained, steps=len_test,method='NN',sps=sps)
    print('Neural Net with MAD is: '+str(mae(yhat,data_tested)))
    return model, mae(yhat,data_tested),yhat

def cnn_tuned(data_trained,data_tested,sps=4,len_test=6):
    optimizer = ['Adam']
    learn_rate = [0.0001, 0.001, 0.01]
    #momentum = [0]
    #init_mode = ['uniform',  'normal']
    activation = ['relu', 'tanh']
    #weight_constraint = [0]
    #dropout_rate = [0.0,.15]
    neurons = [16,32,64,120]
    # define the grid search parameters
    #batch_size = [1,10]
    epochs = [500]
    def create_cnn_model(optimizer='adam',learn_rate=0.01, momentum=0,init_mode='uniform',
                 activation='relu',dropout_rate=0.0, weight_constraint=0,neurons=1):
        # create model
        model = Sequential()
        
        model.add(Conv1D(filters=neurons, kernel_size=2, input_shape=(sps, 1),
                         #kernel_initializer=init_mode,
                         activation=activation,
                         #kernel_constraint=maxnorm(weight_constraint)
                        ))
        if sps>3:model.add(MaxPooling1D(pool_size=2))
        else: model.add(MaxPooling1D(pool_size=1))
        model.add(Flatten())
        model.add(Dense(neurons, activation=activation))#kernel_initializer=init_mode
        #model.add(Dropout(dropout_rate))
        model.add(Dense(1,activation='linear'))#kernel_initializer=init_mode
    
        # Compile model
        if optimizer == 'SGD':
            optimizer = SGD(lr=learn_rate, momentum=momentum)
        if optimizer == 'RMSprop':
            optimizer = RMSprop(lr=learn_rate, momentum=momentum)
        if optimizer == 'Adagrad':
            optimizer = Adagrad(lr=learn_rate)
        if optimizer == 'Adadelta':
            optimizer = Adadelta(lr=learn_rate)
        if optimizer == 'Adam':
            optimizer = Adam(lr=learn_rate)
        model.compile(loss='mae', optimizer=optimizer, metrics=['accuracy'])
        return model
    
    X,y=datamod(data_trained,sps)
    X,y=np.array(X),np.array(y)
    #X=X..reshape(X.shape[0],X.shape[2])
    # reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[2], 1))
    
    param_grid = dict(epochs=epochs,optimizer=optimizer,
                  learn_rate=learn_rate, 
                      #momentum=momentum,init_mode=init_mode,
                  activation=activation,
                      #dropout_rate=dropout_rate, 
                  #weight_constraint=weight_constraint,batch_size=batch_size, 
                      neurons=neurons)
    
    
    model = KerasRegressor(build_fn=create_cnn_model, verbose=0)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=20, cv=3)
    grid_result = grid.fit(X, y)
    
    activation=grid_result.best_params_['activation']
    #batch_size=grid_result.best_params_['batch_size']
    #dropout_rate=grid_result.best_params_['dropout_rate']
    epochs=grid_result.best_params_['epochs']
    #init_mode=grid_result.best_params_['init_mode']
    learn_rate=grid_result.best_params_['learn_rate']
    #momentum=grid_result.best_params_['momentum']
    neurons=grid_result.best_params_['neurons']
    optimizer=grid_result.best_params_['optimizer']
    #weight_constraint=grid_result.best_params_['weight_constraint']

    model=create_cnn_model(optimizer=optimizer,learn_rate=learn_rate,# momentum=momentum,
                       #init_mode=init_mode,
                            activation=activation,
                       #dropout_rate=dropout_rate, weight_constraint=weight_constraint,
                       neurons=neurons)
    model.fit(X, y, epochs=epochs, verbose=0)#batch_size=batch_size
    # pred of model
    # demonstrate prediction
    yhat = Forcast(model,data_trained, steps=len_test,method='CNN',sps=sps)
    print('CNN with MAD is: '+str(mae(yhat,data_tested)))
    return model, mae(yhat,data_tested),yhat
def lstm_tuned(data_trained,data_tested,sps=4,len_test=6):
    optimizer = ['Adam']
    learn_rate = [0.0001, 0.001, 0.01]
    #momentum = [0]
    #init_mode = ['uniform',  'normal']
    activation = ['relu', 'tanh']
    #weight_constraint = [0]
    #dropout_rate = [0.0,.15]
    neurons = [16,32,64,120]
    # define the grid search parameters
    #batch_size = [1,10]
    epochs = [500]
    #import tensorflow as tf
    X,y=datamod(data_trained,sps)
    X,y=np.array(X,dtype='float32'),np.array(y,dtype='float32')
    X = X.reshape((X.shape[0], X.shape[2], 1))
    #X = X.reshape((X.shape[0], X.shape[2]))
    #X=tf.convert_to_tensor(X, dtype=tf.float32, dtype_hint=None, name=None)
    #X=tf.Tensor(X, dtype=tf.float32,shape=(X.shape[0], X.shape[2]))
    #y=tf.convert_to_tensor(y, dtype=None, dtype_hint=None, name=None)
    def create_lstm_model(optimizer='adam',learn_rate=0.01, momentum=0,init_mode='uniform',
                 activation='relu',dropout_rate=0.0, weight_constraint=0,neurons=1):
    # create model
        model = Sequential()
        model.add(LSTM(units =50, input_shape=(X.shape[1], 1),
                       #kernel_initializer=init_mode,
                       activation=activation, return_sequences = True
                         #kernel_constraint=maxnorm(weight_constraint)
                      ))
        model.add(Dropout(0.2))
        model.add(LSTM(units = neurons, return_sequences = True))
        model.add(Dropout(dropout_rate))
        #model.add(Dense(neurons, activation=activation))#,kernel_initializer=init_mode
        #model.add(Dropout(dropout_rate))
        model.add(Dense(1,activation='linear'))#,kernel_initializer=init_mode
    
        # Compile model
        if optimizer == 'SGD':
            optimizer = SGD(lr=learn_rate, momentum=momentum)
        if optimizer == 'RMSprop':
            optimizer = RMSprop(lr=learn_rate, momentum=momentum)
        if optimizer == 'Adagrad':
            optimizer = Adagrad(lr=learn_rate)
        if optimizer == 'Adadelta':
            optimizer = Adadelta(lr=learn_rate)
        if optimizer == 'Adam':
            optimizer = Adam(lr=learn_rate)
        model.compile(loss='mae', optimizer=optimizer, metrics=['accuracy'])
        return model
    
    
    param_grid = dict(epochs=epochs,optimizer=optimizer,
                  learn_rate=learn_rate, 
                      #momentum=momentum,init_mode=init_mode,
                  activation=activation,
                      #dropout_rate=dropout_rate, 
                  #weight_constraint=weight_constraint,batch_size=batch_size, 
                      neurons=neurons)
    
    
    model = KerasRegressor(build_fn=create_lstm_model, verbose=0)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=20, cv=4)
    grid_result = grid.fit(X, y)
    
    activation=grid_result.best_params_['activation']
    #batch_size=grid_result.best_params_['batch_size']
    #dropout_rate=grid_result.best_params_['dropout_rate']
    epochs=grid_result.best_params_['epochs']
    #init_mode=grid_result.best_params_['init_mode']
    learn_rate=grid_result.best_params_['learn_rate']
    #momentum=grid_result.best_params_['momentum']
    neurons=grid_result.best_params_['neurons']
    optimizer=grid_result.best_params_['optimizer']
    #weight_constraint=grid_result.best_params_['weight_constraint']

    model=create_lstm_model(optimizer=optimizer,learn_rate=learn_rate,# momentum=momentum,
                       #init_mode=init_mode,
                            activation=activation,#dropout_rate=dropout_rate, weight_constraint=weight_constraint,
                       neurons=neurons)
    model.fit(X, y, epochs=epochs, verbose=0)#batch_size=batch_size
    # demonstrate prediction
    yhat = Forcast(model,data_trained, steps=len_test,method='LSTM',sps=sps)
    print('LSTM with MAD is: '+str(mae(yhat,data_tested)))
    return model, mae(yhat,data_tested),yhat
#%%
'''
Data=pd.read_csv('data_trans.csv',header=0)
Data=Data.set_index('Date')
data=Data['A']
data_train=data[:24]
sps=1
seed = 7
np.random.seed(seed)
optimizer = ['SGD']
learn_rate = [0.001]
momentum = [0.2]
init_mode = ['uniform']
activation = ['softmax']
weight_constraint = [1]
dropout_rate = [0.0]
neurons = [5]
# define the grid search parameters
batch_size = [20]
epochs = [10]
'''
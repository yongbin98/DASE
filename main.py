import os, psutil
import numpy as np
import scipy.io as sio
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error,  accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, ConfusionMatrixDisplay
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV,KFold
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import lightgbm as lgb
from sklearn.svm import SVR
import sklearn.preprocessing
from pycaret.regression import *
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf
import xgboost as xgb
import smogn
import math
from keras import backend as K
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor

# def log_cosh_loss(y_true, y_pred):
#     error = y_pred - y_true
#     loss = tf.math.log1p(tf.exp(2 * error)) - 2 * tf.math.log(2.0)
#     return tf.reduce_mean(loss, axis=-1)

# def huber_loss(y_true, y_pred, delta=1.0):
#     error = y_true - y_pred
#     quadratic_part = K.minimum(K.abs(error), delta)
#     linear_part = K.abs(error) - quadratic_part
#     loss = 0.5 * K.square(quadratic_part) + delta * linear_part
#     return K.mean(loss, axis=-1)

def loadMatlabData():
    fileName = os.getcwd() + '/DASE/daseData.mat'
    
    ###============= Load Matlab files
    contentsMat = sio.loadmat(fileName)
    x_train = contentsMat['x_train']
    y_train = contentsMat['y_train']
    x_test = contentsMat['x_test']
    y_test = contentsMat['y_test']
    
    return x_train, y_train, x_test, y_test

def loadManualData():
    fileName = os.getcwd() + '/DASE/daseDataManual.mat'
    
    ###============= Load Matlab files
    contentsMat = sio.loadmat(fileName)
    train = contentsMat['train_data']
    test = contentsMat['test_data']
    valid_data1 = contentsMat['valid_data1']
    valid_data2= contentsMat['valid_data2']
    valid_data3 = contentsMat['valid_data3']
    valid_data4 = contentsMat['valid_data4']
    valid_data5 = contentsMat['valid_data5']
    
    return train, test, valid_data1, valid_data2, valid_data3, valid_data4, valid_data5

def loadxlsxData():
    filename = os.getcwd() + '/DASE/KOBIO_data_final(changed).xlsx'
    
    ###============= Load xlsx files
    df = pd.read_excel(filename)    
    return df

def loadbestPred(i):    
    filename = os.getcwd() + '/DASE/result/best_pred'+str(i+1)+'.mat'
    
    contentsMat = sio.loadmat(filename)
    x_train = contentsMat['x_train']
    y_train = contentsMat['y_train']
    x_test = contentsMat['x_test']
    y_test = contentsMat['y_test']
    y_pred = contentsMat['y_pred']
    x_valid = contentsMat['x_valid']
    y_valid = contentsMat['y_valid']
    
    return x_train, x_valid, x_test, y_train, y_valid, y_test, y_pred
    
def weighted_mse_loss(y_true, y_pred, sample_weights):
    squared_errors = np.square(y_true - y_pred)
    weighted_errors = squared_errors * sample_weights
    return np.mean(weighted_errors)

def standarizeInput(x_train, x_valid, x_test, y_train, y_valid, y_test):    
    key = ['AGE','BWT','HGT','BMI','CIGP','CIGY','PTGA','PHGA','ESR','CRP','BTIME','CDOSE','RF','ACCP']
    
    ssl = sklearn.preprocessing.StandardScaler()    
    ssl.fit(x_train[key])
    x_train[key] = ssl.transform(x_train[key])
    x_valid[key] = ssl.transform(x_valid[key])    
    x_test[key] = ssl.transform(x_test[key])    
    
    ssl = sklearn.preprocessing.MinMaxScaler()    
    ssl.fit(x_train[key])
    x_train[key] = ssl.transform(x_train[key])
    x_valid[key] = ssl.transform(x_valid[key])
    x_test[key] = ssl.transform(x_test[key])
        
    return x_train, x_valid, x_test, y_train, y_valid, y_test

def standarizeInput4(x_train, x_test, y_train, y_test):     
    key = ['AGE','BWT','HGT','BMI','CIGP','CIGY','PTGA','PHGA','ESR','CRP','BTIME','CDOSE','RF','ACCP']
    
    ssl = sklearn.preprocessing.StandardScaler()    
    ssl.fit(x_train[key])
    x_train[key] = ssl.transform(x_train[key])
    x_test[key] = ssl.transform(x_test[key])    
    
    ssl = sklearn.preprocessing.MinMaxScaler()    
    ssl.fit(x_train[key])
    x_train[key] = ssl.transform(x_train[key])
    x_test[key] = ssl.transform(x_test[key])
        
    return x_train, x_test, y_train, y_test

def applySMOGN(x_train, y_train):    
    np.random.seed(99)
    df_smogn = pd.concat([x_train, y_train], axis=1)
    
    rg_mtrx = [
        [3, 1, 0],  ## over-sample ("minority")
        [7, 0, 0],  ## under-sample ("majority")
        [9, 1, 0],  ## over-sample ("minority")
    ]
    
    smogned = smogn.smoter(
        data=df_smogn.reset_index(drop=True),
        y='DASE',
        k=10,
        samp_method='extreme',
        rel_thres=0.9,
        rel_method = 'auto',    
        rel_xtrm_type = 'high',  
        rel_coef = 0.9
        # rel_method='manual',
        # rel_ctrl_pts_rg=rg_mtrx
    )
    # smogned = newSmoteR(df_smogn, target='DASE', th = 0.999, o = 300, u = 100, k = 10, categorical_col = x_train.columns)    
    
    x_train_smogned = smogned.drop("DASE", axis=1)
    y_train_smogned = smogned["DASE"]

    print(x_train.shape)
    print(x_train_smogned.shape)
    
    return x_train_smogned, y_train_smogned
    
# def get_synth_cases(D, target, o=200, k=3, categorical_col = []):
#     '''
#     Function to generate the new cases.
#     INPUT:
#         D - pd.DataFrame with the initial data
#         target - string name of the target column in the dataset
#         o - oversampling rate
#         k - number of nearest neighbors to use for the generation
#         categorical_col - list of categorical column names
#     OUTPUT:
#         new_cases - pd.DataFrame containing new generated cases
#     '''
#     new_cases = pd.DataFrame(columns = D.columns) # initialize the list of new cases 
#     ng = o // 100 # the number of new cases to generate
#     for index, case in D.iterrows():
#         # find k nearest neighbors of the case
#         knn = KNeighborsRegressor(n_neighbors = k+1) # k+1 because the case is the nearest neighbor to itself
#         knn.fit(D.drop(columns = [target]).values, D[[target]])
#         neighbors = knn.kneighbors(case.drop(labels = [target]).values.reshape(1, -1), return_distance=False).reshape(-1)
#         neighbors = np.delete(neighbors, np.where(neighbors == index))
#         for i in range(0, ng):
#             # randomly choose one of the neighbors
#             x = D.iloc[neighbors[np.random.randint(k)]]
#             attr = {}          
#             for a in D.columns:
#                 # skip target column
#                 if a == target:
#                     continue;
#                 if a in categorical_col:
#                     # if categorical then choose randomly one of values
#                     if np.random.randint(2) == 0:
#                         attr[a] = case[a]
#                     else:
#                         attr[a] = x[a]
#                 else:
#                     # if continious column
#                     diff = case[a] - x[a]
#                     attr[a] = case[a] + np.random.randint(2) * diff
#             # decide the target column
#             new = np.array(list(attr.values()))
#             d1 = cosine_similarity(new.reshape(1, -1), case.drop(labels = [target]).values.reshape(1, -1))[0][0]
#             d2 = cosine_similarity(new.reshape(1, -1), x.drop(labels = [target]).values.reshape(1, -1))[0][0]
#             attr[target] = (d2 * case[target] + d1 * x[target]) / (d1 + d2)
            
#             # append the result
#             new_cases = new_cases.append(attr,ignore_index = True)
                    
#     return new_cases

# def newSmoteR(D, target, th = 0.999, o = 200, u = 100, k = 3, categorical_col = []):
#     '''
#     The implementation of SmoteR algorithm:
#     https://core.ac.uk/download/pdf/29202178.pdf
#     INPUT:
#         D - pd.DataFrame - the initial dataset
#         target - the name of the target column in the dataset
#         th - relevance threshold
#         o - oversampling rate
#         u - undersampling rate
#         k - the number of nearest neighbors
#     OUTPUT:
#         new_D - the resulting new dataset
#     '''
#     # median of the target variable
#     y_bar = D[target].median()
    
#     # find rare cases where target less than median
#     rareL = D[(relevance(D[target]) > th) & (D[target] > y_bar)]  
#     # generate rare cases for rareL
#     new_casesL = get_synth_cases(rareL, target, o, k , categorical_col)
    
#     # find rare cases where target greater than median
#     rareH = D[(relevance(D[target]) > th) & (D[target] < y_bar)]
#     # generate rare cases for rareH
#     new_casesH = get_synth_cases(rareH, target, o, k , categorical_col)
    
#     new_cases = pd.concat([new_casesL, new_casesH], axis=0)
    
#     # undersample norm cases
#     norm_cases = D[relevance(D[target]) <= th]
#     # get the number of norm cases
#     nr_norm = int(len(norm_cases) * u / 100)
    
#     norm_cases = norm_cases.sample(min(len(D[relevance(D[target]) <= th]), nr_norm))
    
#     # get the resulting dataset
#     new_D = pd.concat([new_cases, norm_cases], axis=0)
    
#     return new_D
    
# def relevance(x):
#     x = np.array(x)
#     return sigmoid(x - 50)

# def sigmoid(x):
#     return 1 / (1 +np.exp(-x))

# def log_cosh_loss(y_true, y_pred):
#     error = y_pred - y_true
#     loss = tf.math.log1p(tf.exp(2 * error)) - 2 * tf.math.log(2.0)
#     return tf.reduce_mean(loss, axis=-1)

# def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    quadratic_part = K.minimum(K.abs(error), delta)
    linear_part = K.abs(error) - quadratic_part
    loss = 0.5 * K.square(quadratic_part) + delta * linear_part
    return K.mean(loss, axis=-1)

def modelDNN(x_train, x_valid, x_test, y_train, y_valid, y_test):
    
    input = tf.keras.layers.Input(shape=(111,1,1))
    x = tf.keras.layers.Flatten(name='Flatten')(input)
    x = tf.keras.layers.Dense(32, activation='relu', name='FC2')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(16, activation='relu', name='FC3')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(8, activation='relu', name='FC4')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    output = tf.keras.layers.Dense(1, name='Output')(x)
    
    model= tf.keras.models.Model(inputs=input, outputs=output)
    model.summary()
    
    callback_list = [
            tf.keras.callbacks.EarlyStopping(monitor='mae', mode='min', verbose=0, patience=10),
            tf.keras.callbacks.ModelCheckpoint(filepath=os.getcwd()+'/DASE/model/model_dnn.h5', monitor='mae', mode='min', verbose=0, save_best_only=True, save_weights_only=True),
        ]
    
    # sample_weight = np.ones_like(y_train)  
    # sample_weight[:] = 0.5
    # sample_weight[y_train<4] = 1
    # sample_weight[y_train>7] = 1        
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,loss='mae',metrics=['mae','mse'])
    x_train = tf.expand_dims(x_train, axis=-1) 
    x_valid = tf.expand_dims(x_valid, axis=-1) 
    # model.fit(x_train, y_train, batch_size=16, epochs=300, validation_data = (x_valid,y_valid), callbacks=callback_list)
    model.fit(x_train, y_train, batch_size=16, epochs=100)
    # model.load_weights(filepath=os.getcwd()+'/DASE/model/model_dnn.h5')            
    x_test = tf.expand_dims(x_test, axis=-1) 
    
    y_pred = model.predict(x_valid)
    evaluation(y_valid,y_pred)
    y_pred = model.predict(x_test)
    evaluation(y_test,y_pred)
    
    sio.savemat(os.getcwd()+'/DASE/result/single_pred.mat',{'y_test' : y_test.values, 'y_pred' : y_pred})
    return y_pred
    
def modelGridSearch(x_train, x_test, y_train, y_test):
    param_grid = {
    # ## XGB
    # 'max_depth': [5, 7, 9, 11],
    # 'learning_rate': [0.1, 0.05, 0.01],
    # 'n_estimators': [100, 200, 300],
    # 'subsample': [0.7, 0.8, 0.9],
    # 'colsample_bytree': [0.7, 0.9, 1],
    # 'booster': ['gbtree', 'gblinear'],
    ### Random Forest
        # 'n_estimators': [700],
        # 'max_depth': [None],
        # 'min_samples_split': [2],
        # 'min_samples_leaf': [1],
        # 'max_features': ['auto'],
        # 'bootstrap': [True],
        # 'random_state': [99],
    ### SVR
        # 'C': [0.1, 1, 10, 100, 1000], 
        # 'gamma': ['scale','auto',1, 0.1, 0.01, 0.001, 0.0001],
        # 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        # 'epsilon': [0.1, 0.2, 0.5, 1.0],
    # ### LGBM
    #     'boosting_type': ['gbdt', 'dart', 'goss'],
    #     'num_leaves': [10, 20, 30, 40, 50],
    #     'learning_rate': [0.1, 0.01, 0.001],
    #     'subsample': [0.8, 0.9, 1.0],
    #     'colsample_bytree': [0.8, 0.9, 1.0],
    #     'reg_alpha': [0.0, 0.1, 0.5],
    #     'reg_lambda': [0.0, 0.1, 0.5],
    #     'n_estimators': [100, 200, 300],
    #     'min_child_samples': [20, 50, 100],
    #     'max_depth': [-1, 5, 10]
    ### ADA
        # 'n_estimators': [50, 100, 200, 300],
        # 'learning_rate': [0.1, 0.05, 0.01],
        # 'loss': ['linear', 'square', 'exponential'],
        # 'base_estimator': [None, DecisionTreeRegressor(max_depth=3), DecisionTreeRegressor(max_depth=5)],
        # 'random_state': [42]
    ### CB
        'iterations': [300, 400 ,500],
        'learning_rate': [0.05],
        'depth': [10,12,14],
        'l2_leaf_reg': [1]
    }
    
    model = CatBoostRegressor()
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose = 2)
    grid_search.fit(x_train, y_train)
    
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    y_pred = best_model.predict(x_test)
    print(best_params)
    evaluation(y_test,y_pred)
    return y_pred
   
def modelCB(x_train, x_test, y_train, y_test):
    model = CatBoostRegressor(iterations=300,depth=10,l2_leaf_reg=1,learning_rate=0.05).fit(x_train,y_train)
    
    # iterations=1000,
    #                              learning_rate=0.1,
    #                              depth=4,
    #                              l2_leaf_reg=20,
    #                              bootstrap_type='Bernoulli',
    #                              subsample=0.6,
    #                              eval_metric='RMSE',
    #                              metric_period=50,
    #                              od_type='Iter',
    #                              od_wait=45,
    #                              random_seed=17,
    #                              allow_writing_files=False
                                 
    # cb_model.fit(trn_x, trn_y,
    #          eval_set=(val_x, val_y),
    #          cat_features=[],
    #          use_best_model=True,
    #          verbose=True)
    
    train_pred = model.predict(x_train.squeeze())
    test_pred = model.predict(x_test.squeeze())    
    evaluation(y_test.squeeze(),test_pred)
    return test_pred, train_pred

def modelXGB(x_train, x_test, y_train, y_test):    
    model = xgb.XGBRegressor(objective='reg:squarederror',n_estimators=100, learning_rate=0.05, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7).fit(x_train,y_train)
    
    train_pred = model.predict(x_train.squeeze())
    test_pred = model.predict(x_test.squeeze())
    print(model)
    evaluation(y_test.squeeze(),test_pred)
    return test_pred, train_pred

def modelRF(x_train, x_test, y_train, y_test):    
    model = RandomForestRegressor(bootstrap=True,max_depth=None,max_features='auto',min_samples_leaf=1,min_samples_split=2,n_estimators=400,random_state=99).fit(x_train,y_train)
    
    train_pred = model.predict(x_train.squeeze())
    test_pred = model.predict(x_test.squeeze())
    print(model)
    evaluation(y_test.squeeze(),test_pred)
    return test_pred, train_pred

def modelGBR(x_train, x_test, y_train, y_test):        
    model = GradientBoostingRegressor().fit(x_train.squeeze(),y_train.squeeze())
    
    train_pred = model.predict(x_train.squeeze())
    test_pred = model.predict(x_test.squeeze())
    print(model)
    evaluation(y_test.squeeze(),test_pred)
    return test_pred, train_pred

def modelADA(x_train, x_test, y_train, y_test):    
    model = AdaBoostRegressor(learning_rate=0.05,loss='square',random_state=99,n_estimators=100,base_estimator=DecisionTreeRegressor(max_depth=7)).fit(x_train.squeeze(),y_train.squeeze())
    
    train_pred = model.predict(x_train.squeeze())
    test_pred = model.predict(x_test.squeeze())
    print(model)
    evaluation(y_test.squeeze(),test_pred)
    return test_pred, train_pred

def modelLGB(x_train, x_test, y_train, y_test):    
    model = lgb.LGBMRegressor(boosting_type='gbdt',colsample_bytree=0.8,learning_rate=0.05,max_depth=-1,min_child_samples=20,n_estimators=300,num_leaves=50,reg_alpha=0.5,
                              reg_lambda=0.5,subsample=0.8).fit(x_train.squeeze(),y_train.squeeze())
    
    train_pred = model.predict(x_train.squeeze())
    test_pred = model.predict(x_test.squeeze())
    print(model)
    evaluation(y_test.squeeze(),test_pred)
    return test_pred, train_pred

def modelSVR(x_train, x_test, y_train, y_test):    
    
    model = SVR(C=10,epsilon=1).fit(x_train.squeeze(),y_train.squeeze())
    
    train_pred = model.predict(x_train.squeeze())
    test_pred = model.predict(x_test.squeeze())
    print(model)
    evaluation(y_test.squeeze(),test_pred)
    return test_pred, train_pred

def modelMLMerge(x_train, x_valid, x_test, y_train, y_valid, y_test):
    input1 = tf.keras.layers.Input(shape=(1))  
    input2 = tf.keras.layers.Input(shape=(1))  
    input3 = tf.keras.layers.Input(shape=(1))  
    input4 = tf.keras.layers.Input(shape=(1))  
    input5 = tf.keras.layers.Input(shape=(1))  
    input6 = tf.keras.layers.Input(shape=(1))  
        
    x = tf.keras.layers.Concatenate()([input1,input2,input3,input4,input5,input6])
    x = tf.keras.layers.Dense(32, activation='relu', name='FC1')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(16, activation='relu', name='FC2')(x)
    x = tf.keras.layers.Dense(8, activation='relu', name='FC3')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(4, activation='relu', name='FC4')(x)
    output = tf.keras.layers.Dense(1, name='Output')(x)
    
    callback_list = [
            tf.keras.callbacks.EarlyStopping(monitor='mae', mode='min', verbose=0, patience=10),
            tf.keras.callbacks.ModelCheckpoint(filepath=os.getcwd()+'/DASE/model/model_dnn.h5', monitor='mae', mode='min', verbose=0, save_best_only=True, save_weights_only=True),
        ]
    
    model= tf.keras.models.Model(inputs=[input1,input2,input3,input4,input5,input6], outputs=output)
    model.summary()

    # sample_weight = np.ones_like(y_train)  
    # sample_weight[:] = 0.5
    # sample_weight[y_train<5] = 1
    # sample_weight[y_train>7] = 1        
        
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    #model.compile(optimizer=optimizer,loss=weighted_mse_loss(sample_weights = sample_weight), metrics=['mae'])
    model.compile(optimizer=optimizer,loss='mae', metrics=['mae'])
    model.fit(x_train, y_train, batch_size=16, epochs=60)
    # model.fit(x_train, y_train, batch_size=16, epochs=300, validation_data = (x_valid,y_valid), callbacks=callback_list)
    # model.load_weights(filepath=os.getcwd()+'/DASE/model/model_dnn.h5')        
    y_pred = model.predict(x_valid)
    evaluation(y_valid,y_pred)
    
    return y_pred

def modelClassification(x_train, x_test, y_train, y_test):  
    category = [0,1,2]  
    # input = tf.keras.layers.Input(shape=(111))
    # x = tf.keras.layers.Flatten(name='Flatten')(input)
    # x = tf.keras.layers.Dense(64, activation='relu', name='FC1')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Dropout(0.2)(x)
    # x = tf.keras.layers.Dense(32, activation='relu', name='FC2')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Dropout(0.2)(x)
    # x = tf.keras.layers.Dense(16, activation='relu', name='FC3')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Dense(8, activation='relu', name='FC4')(x)
    # output = tf.keras.layers.Dense(3, activation='softmax', name='Output')(x)
    
    # model= tf.keras.models.Model(inputs=input, outputs=output)
    # model.summary()
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    # model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    # model.fit(x_train, y_train, batch_size=16, epochs=50)
    
    # test_pred = model.predict(x_test.squeeze())
    # print(model)
    # Pred = decode_one_hot(test_pred, len(category))
    # get_clf_eval(y_test, Pred.flatten(), test_pred, classes=category)
    
    
    model = xgb.XGBClassifier(booster = 'gbtree', colsample_bytree = 0.9, learning_rate = 0.05, max_depth = 5, n_estimators = 300, subsample = 0.7).fit(x_train,y_train)
    
    test_pred = model.predict(x_test.squeeze())
    print(model)
    Pred = one_hot(test_pred, len(category))
    get_clf_eval(y_test, test_pred.flatten(), Pred, classes=category)
    
    return test_pred

def one_hot(y_, n_classes=6):
    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS 

def decode_one_hot(y_, n_classes=3):
    new_y = np.zeros([int(y_.size/n_classes)])
    for i in range(0,int(y_.size/n_classes)):
        max = 0
        for j in range(0,n_classes):
            if(y_[i,max] < y_[i,j]):
                max = j
        new_y[i] = max;
        
    return new_y  # Returns FLOATS 

def randomOversampler(x_train, y_train):    
    oversample = RandomOverSampler(sampling_strategy='minority')
    y_train = round(y_train)
    overx_train, overy_train = oversample.fit_resample(x_train,y_train)
    return overx_train, overy_train

def evaluation(y_test,y_pred):
    mse = mean_squared_error(y_test, y_pred)    
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2Score = r2_score(y_test,y_pred)
    print("MAE :", mae)
    print("MSE :", mse)
    print("RMSE :", rmse)    
    print("MAPE :", mape)  
    print("R2 :", r2Score)
    return mae,mse,rmse,mape,r2Score

def evaluationWithstd(mae,mse,rmse,mape,r2Score):    
    print("MAE : ", np.mean(mae), " + ",np.std(mae))
    print("MSE : ", np.mean(mse), " + ",np.std(mse))
    print("RMSE : ", np.mean(rmse), " + ",np.std(rmse))
    print("MAPE : ", np.mean(mape), " + ",np.std(mape))
    print("R2 : ", np.mean(r2Score), " + ",np.std(r2Score))
    return mae,mse,rmse,mape,r2Score

def qunatileSplit(x_data,y_data,df_columns):
    sorted_indices = np.argsort(y_data)
    sorted_x_data = x_data[sorted_indices]
    sorted_y_data = y_data[sorted_indices]
    num_quantiles = 45
    
    bins = np.linspace(0, len(y_data), num_quantiles+1, dtype=int)
    bin_indices = np.digitize(range(len(y_data)), bins)
    x_train, x_test, y_train, y_test = train_test_split(sorted_x_data, sorted_y_data, test_size=0.2, stratify=bin_indices, random_state=99)
    
    x_train = numpyToDataFrame(x_train,df_columns[1:-1])
    x_test = numpyToDataFrame(x_test,df_columns[1:-1])
    y_train = numpyToDataFrame(y_train,['DASE'])
    y_test = numpyToDataFrame(y_test,['DASE'])
    
    return x_train, x_test, y_train, y_test

def numpyToDataFrame(x,column):    
    df_x = pd.DataFrame(x)
    df_x.columns = column
    return df_x

def pltFeature(importance, feature_names):
    indices = importance.argsort()[::-1]
    sorted_importance = importance[indices]
    sorted_feature_names = [feature_names[i] for i in indices]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_importance)), sorted_importance)
    plt.xticks(range(len(sorted_importance)), sorted_feature_names, rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.show()

def get_clf_eval(y_true, pred=None, pred_proba=None, classes=[0, 1]):
    confusion = confusion_matrix(y_true, pred, labels=classes)
    accuracy = accuracy_score(y_true, pred)
    precision = precision_score(y_true, pred,average=None)
    recall = recall_score(y_true, pred,average=None)
    f1 = f1_score(y_true, pred,average=None)
    # roc_auc = roc_auc_score(y_true, pred_proba, average=None, multi_class='ovr')
    
    TP = np.zeros(len(classes))
    FP = np.zeros(len(classes))
    FN = np.zeros(len(classes))
    TN = np.zeros(len(classes))
    spec = np.zeros(len(classes))
    for i in classes:
        for j in classes:
            for z in classes:
                if i == j and j == z :
                    TP[i] += confusion[j,z]
                elif i == j and i != z:
                    FN[i] += confusion[j,z]
                elif i != j and i == z:
                    FP[i] += confusion[j,z]                    
                else : 
                    TN[i] += confusion[j,z]               
        spec[i] = TN[i]/(FP[i]+TN[i])
        
    bal_acc = ((recall + spec) / 2)
    
    print('오차 행렬')
    print(confusion)
    print(f'정확도 : {accuracy}')
    print(f'정밀도 : {precision}')
    print(f'재현율 : {recall}')
    print(f'특이성 : {spec}')
    print(f'F1 : {f1}')
    # print(f'AUC : {roc_auc}')
    print(f'bal_acc : {bal_acc}')
    return accuracy, precision, recall, spec, bal_acc, f1

def trainvalidSplit(train, test, valid_data1, valid_data2, valid_data3, valid_data4, valid_data5,i):
    if i == 0:
        x_train = pd.concat([valid_data1.drop(['DASE'],axis=1),valid_data2.drop(['DASE'],axis=1),valid_data3.drop(['DASE'],axis=1),valid_data4.drop(['DASE'],axis=1)])
        y_train = pd.concat([valid_data1['DASE'],valid_data2['DASE'],valid_data3['DASE'],valid_data4['DASE']])
        x_valid = valid_data5.drop(['DASE'],axis=1)
        y_valid = valid_data5['DASE']
        x_test = test.drop(['DASE'],axis=1)
        y_test = test['DASE']
    elif i == 1:
        x_train = pd.concat([valid_data1.drop(['DASE'],axis=1),valid_data2.drop(['DASE'],axis=1),valid_data3.drop(['DASE'],axis=1),valid_data5.drop(['DASE'],axis=1)])
        y_train = pd.concat([valid_data1['DASE'],valid_data2['DASE'],valid_data3['DASE'],valid_data5['DASE']])
        x_valid = valid_data4.drop(['DASE'],axis=1)
        y_valid = valid_data4['DASE']
        x_test = test.drop(['DASE'],axis=1)
        y_test = test['DASE']
    elif i == 2:
        x_train = pd.concat([valid_data1.drop(['DASE'],axis=1),valid_data2.drop(['DASE'],axis=1),valid_data4.drop(['DASE'],axis=1),valid_data5.drop(['DASE'],axis=1)])
        y_train = pd.concat([valid_data1['DASE'],valid_data2['DASE'],valid_data4['DASE'],valid_data5['DASE']])
        x_valid = valid_data3.drop(['DASE'],axis=1)
        y_valid = valid_data3['DASE']
        x_test = test.drop(['DASE'],axis=1)
        y_test = test['DASE']
    elif i == 3:
        x_train = pd.concat([valid_data1.drop(['DASE'],axis=1),valid_data3.drop(['DASE'],axis=1),valid_data4.drop(['DASE'],axis=1),valid_data5.drop(['DASE'],axis=1)])
        y_train = pd.concat([valid_data1['DASE'],valid_data3['DASE'],valid_data4['DASE'],valid_data5['DASE']])
        x_valid = valid_data2.drop(['DASE'],axis=1)
        y_valid = valid_data2['DASE']
        x_test = test.drop(['DASE'],axis=1)
        y_test = test['DASE']
    elif i == 4:
        x_train = pd.concat([valid_data2.drop(['DASE'],axis=1),valid_data3.drop(['DASE'],axis=1),valid_data4.drop(['DASE'],axis=1),valid_data5.drop(['DASE'],axis=1)])
        y_train = pd.concat([valid_data2['DASE'],valid_data3['DASE'],valid_data4['DASE'],valid_data5['DASE']])
        x_valid = valid_data1.drop(['DASE'],axis=1)
        y_valid = valid_data1['DASE']
        x_test = test.drop(['DASE'],axis=1)
        y_test = test['DASE']
        
    
    return x_train, x_valid, x_test, y_train, y_valid, y_test

df = loadxlsxData()
x_data = df.drop(['DASE','Subject'],axis=1)
y_data = df['DASE']
# y_data[y_data < 4] = 4
# y_data[y_data > 7] = 7
# x_train, x_test, y_train, y_test = qunatileSplit(x_data.values,y_data.values,df.columns)
train, test, valid_data1, valid_data2, valid_data3, valid_data4, valid_data5 = loadManualData()
train = numpyToDataFrame(train,df.columns[1:])
test = numpyToDataFrame(test,df.columns[1:])
valid_data1 = numpyToDataFrame(valid_data1,df.columns[1:])
valid_data2 = numpyToDataFrame(valid_data2,df.columns[1:])
valid_data3 = numpyToDataFrame(valid_data3,df.columns[1:])
valid_data4 = numpyToDataFrame(valid_data4,df.columns[1:])
valid_data5 = numpyToDataFrame(valid_data5,df.columns[1:])

### grid search
# x_train = train.drop(['DASE'],axis=1);
# x_test = test.drop(['DASE'],axis=1);
# y_train = train['DASE']
# y_test = test['DASE']

# x_train, x_test, y_train, y_test = standarizeInput4(x_train, x_test, y_train, y_test)
# x_train, y_train = applySMOGN(x_train, y_train)

# y_pred = modelGridSearch(x_train, x_test, y_train, y_test)
# y_pred = modelXGB(x_train, x_test, y_train, y_test)
# evaluation(y_pred,y_test)

### single model dnn
# x_train, x_test, y_train, y_test = train_test_split(x_data.values, y_data.values, test_size=0.2, shuffle=True, random_state=99)
# x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, shuffle=True, random_state=99)

# x_train = numpyToDataFrame(x_train,df.columns[1:-1])
# x_valid = numpyToDataFrame(x_valid,df.columns[1:-1])
# x_test = numpyToDataFrame(x_test,df.columns[1:-1])
# y_train = numpyToDataFrame(y_train,['DASE'])
# y_valid = numpyToDataFrame(y_valid,['DASE'])
# y_test = numpyToDataFrame(y_test,['DASE'])

# x_train, x_valid, x_test, y_train, y_valid, y_test = standarizeInput(x_train, x_valid, x_test, y_train, y_valid, y_test)
# x_train, y_train = applySMOGN(x_train, y_train)

### Classification
x_train = train.drop(['DASE'],axis=1);
x_test = test.drop(['DASE'],axis=1);
y_train = train['DASE']
y_test = test['DASE']

filtered_indices1 = y_train[y_train < 7].index
filtered_indices2 = y_test[y_test < 7].index

x_train = x_train.drop(index=filtered_indices1, axis=0)
y_train = y_train.drop(index=filtered_indices1, axis=0)
x_test = x_test.drop(index=filtered_indices2, axis=0)
y_test = y_test.drop(index=filtered_indices2, axis=0)

x_train, x_test, y_train, y_test = standarizeInput4(x_train, x_test, y_train, y_test)
x_train, y_train = randomOversampler(x_train, y_train)
test_pred, train_pred1 = modelXGB(x_train, x_test, y_train, y_test)
sio.savemat(os.getcwd()+'/DASE/result/single_pred.mat',{'y_test' : y_test.values, 'y_pred' : test_pred})

# # x_train, y_train = applySMOGN(x_train, y_train)
# y_train[y_train < 4] = 0
# y_train[y_train > 7] = 2
# y_train[y_train >= 4] = 1
# y_test[y_test < 4] = 0
# y_test[y_test > 7] = 2
# y_test[y_test >= 4] = 1
# modelClassification(x_train, x_test, y_train, y_test)

### 5 fold
# valid_Preds = []
# test_Preds = []
# validmaes = []
# validmses = []
# validrmses = [] 
# validmapes = []
# validr2scores = []
# testmaes = []
# testmses = []
# testrmses = [] 
# testmapes = []
# testr2scores = []

# for i in range(5):
#     x_train, x_valid, x_test, y_train, y_valid, y_test = trainvalidSplit(train.copy(), test.copy(), valid_data1.copy(), valid_data2.copy(), valid_data3.copy(), valid_data4.copy(), valid_data5.copy(),i)
        
#     x_train, x_valid, x_test, y_train, y_valid, y_test = standarizeInput(x_train, x_valid, x_test, y_train, y_valid, y_test)
#     ### augmentation
#     # x_train, y_train = applySMOGN(x_train, y_train)    
#     x_train, y_train = randomOversampler(x_train, y_train)
    
#     valid_pred, train_pred = modelXGB(x_train, x_valid, y_train, y_valid) # CB XGB RF ADA LGB SVR
#     # valid_pred2, train_pred = modelCB(x_train, x_valid, y_train, y_valid)
#     # valid_pred = (valid_pred1+valid_pred2)/2
#     valid_Preds.append(valid_pred)   
     
#     test_pred, train_pred1 = modelXGB(x_train, x_test, y_train, y_test)
#     # test_pred2, train_pred2 = modelCB(x_train, x_test, y_train, y_test)
#     # test_pred = (test_pred1+test_pred2)/2
#     test_Preds.append(test_pred)    
    
#     mae,mse,rmse,mape,r2Score = evaluation(y_valid,valid_pred)    
#     validmaes.append(mae)
#     validmses.append(mse)
#     validrmses.append(rmse)
#     validmapes.append(mape)
#     validr2scores.append(r2Score)
    
#     mae,mse,rmse,mape,r2Score = evaluation(y_test,test_pred)    
#     testmaes.append(mae)
#     testmses.append(mse)
#     testrmses.append(rmse)
#     testmapes.append(mape)
#     testr2scores.append(r2Score)    

# # x_train = train.drop(['DASE'],axis=1)
# # x_test = test.drop(['DASE'],axis=1)
# # y_train = train['DASE']
# # y_test = test['DASE']
# # x_train, x_test, y_train, y_test = standarizeInput4(x_train, x_test, y_train, y_test)
# # # x_train, y_train = applySMOGN(x_train, y_train)
# # x_train, y_train = randomOversampler(x_train, y_train)
# # single_pred, train_pred = modelSVR(x_train, x_test, y_train, y_test)

# # print('SINGLE\n')
# # evaluation(y_test,single_pred)

# print('VALID\n')
# mae,mse,rmse,mape,r2Score = evaluationWithstd(validmaes,validmses,validrmses,validmapes,validr2scores)

# print('TEST\n')
# mae,mse,rmse,mape,r2Score = evaluationWithstd(testmaes,testmses,testrmses,testmapes,testr2scores)
# sio.savemat(os.getcwd()+'/DASE/result/single_pred.mat',{'y_test' : y_test.values, 'y_pred' : test_Preds})
### ensemble softvoting

# vPreds = []
# for i in range(5):
#     x_train, x_valid, x_test, y_train, y_valid, y_test, y_pred = loadbestPred(i)    
#     vPreds.append(y_pred)
#     evaluation(y_test,y_pred)
    
# vPred = (vPreds[0] + vPreds[1] + vPreds[2] + vPreds[3] + vPreds[4])/5
# evaluation(y_test,vPred)
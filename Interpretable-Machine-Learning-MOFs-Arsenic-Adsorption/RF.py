path1 = '.../training/...'
path2 = '.../MF/...'

import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import ensemble
from sklearn import tree
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction

df = pd.read_csv(path1 + 'trainingdata.csv')
MOF_construction = df.loc[:,'name':'topo']
Structure_info = df.loc[:,'GLD':'AV']
Absorb_env = df.loc[:,'Tem':'Concentration']
Capacity = df.loc[:,'Capacity']
Mfs1 = pd.read_csv(path2 + 'Morgan.csv').loc[:, '0':] 
Mfs2 = pd.read_csv(path2 + 'MACCS.csv').loc[:, '0':] 

Metal1 = pd.get_dummies(MOF_construction.loc[:,'metal_smiles'])
Topo = pd.get_dummies(MOF_construction.loc[:,'topo'])
Data = pd.concat([MOF_construction, Structure_info, Absorb_env, Mfs1, Metal1, Topo],axis=1) 
X = Data.drop(['name', 'metal_smiles', 'organ_smiles', 'topo', 'GLD'],axis=1)
Y = Capacity

X = np.array(X)
Y = np.array(Y)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

for train_index, test_index in kf.split(X):
    print(train_index, test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

#Hyperparameter optimization
rf_run = ensemble.RandomForestRegressor(random_state=0)
rf_run.get_params()

def bo_params_rf(max_depth, min_samples_leaf, min_samples_split, max_samples,n_estimators):
    params = {'max_depth': int(max_depth),
              'min_samples_leaf': int(min_samples_leaf),
              'min_samples_split': int(min_samples_split),
              'max_samples': max_samples,
              'n_estimators': int(n_estimators)}
    reg = ensemble.RandomForestRegressor(random_state=0,
                                         **params)
    reg.fit(X_train, y_train)
    preds = reg.predict(X_train)
    mse = mean_squared_error(y_train, preds)#
    rmse = np.sqrt(mse)#
    r2 = r2_score(y_train, preds)#
    save_path = '.../...'
    file_path = os.path.join(save_path, 'RF1optimization_output.txt')
    with open(file_path, 'a') as f:
        f.write(out + '\n')

    return -rmse

uf = UtilityFunction(kind='ei', kappa=1.5, xi=0.01) 

rf_BO = BayesianOptimization(bo_params_rf, {'max_depth': (60,100),
                                            'min_samples_leaf': (1,4),
                                            'min_samples_split': (2,10),
                                            'max_samples': (0.5,1),
                                            'n_estimators': (100, 250)},
                              random_state=1)
                              
results = rf_BO.maximize(n_iter=200, init_points=20,  
                         acquisition_function=uf)

params = rf_BO.max['params']
params['max_depth']= int(params['max_depth'])
params['min_samples_leaf']= int(params['min_samples_leaf'])
params['min_samples_split']= int(params['min_samples_split'])
params['n_estimators']= int(params['n_estimators'])

rf_run = ensemble.RandomForestRegressor(random_state=0,
                                        max_depth=params['max_depth'],
                                        min_samples_leaf=params['min_samples_leaf'],
                                        min_samples_split=params['min_samples_split'],
                                        max_samples=params['max_samples'],
                                        n_estimators=params['n_estimators'])

tr_rmse, te_rmse = [], [] 
tr_r2, te_r2 = [], []
tr_mae, te_mae = [], []


kf = KFold(n_splits=10, shuffle=True, random_state=42)

for i, (tr_idx,te_idx) in enumerate(kf.split(X, Y)):
    

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    rf_run.fit(X_train, y_train)
    y_tr_pred = rf_run.predict(X_train)
    y_te_pred = rf_run.predict(X_test)


    tr_rmse.append(mean_squared_error(y_tr_pred, y_train) ** 0.5)
    te_rmse.append(mean_squared_error(y_te_pred, y_test) ** 0.5)
    tr_mae.append(mean_absolute_error(y_tr_pred, y_train))
    te_mae.append(mean_absolute_error(y_te_pred, y_test))  

    tr_r2.append(r2_score(y_tr_pred, y_train))
    te_r2.append(r2_score(y_te_pred, y_test))
    
    print("Fold:", i+1)
     
best_idx1 = np.argmax(tr_rmse)
best_idx2 = np.argmax(te_rmse)

train = pd.DataFrame({'Experimental': y_train, 'Predicted': y_tr_pred, 'TrainTest': ['Train']*len(y_train)})
test = pd.DataFrame({'Experimental': y_test, 'Predicted': y_te_pred, 'TrainTest': ['Test']*len(y_test)})

results = pd.concat([train, test], ignore_index=True) 

save_path = '.../...'
file_path = os.path.join(save_path, 'RF1results.csv')
results.to_csv(file_path, index=False)

file_path = os.path.join(save_path, 'RF1_model.pkl')
with open(file_path, 'wb') as f:
    pickle.dump(rf_run, f)
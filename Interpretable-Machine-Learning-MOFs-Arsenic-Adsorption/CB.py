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
from catboost import CatBoostRegressor
import catboost as cb

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

def cb_reg_objective(learning_rate, depth, l2_leaf_reg, rsm):
    params = {'learning_rate': learning_rate,
              'depth': int(depth),
              'l2_leaf_reg': int(l2_leaf_reg), 
              'rsm': rsm}

    model = CatBoostRegressor(**params, random_state=0)
    model.fit(X_train, y_train)  
    preds = model.predict(X_train)
    mse = mean_squared_error(y_train, preds)#
    rmse = np.sqrt(mse)#
    r2 = r2_score(y_train, preds)#
    save_path = '.../...'
    file_path = os.path.join(save_path, 'CB2optimization_output.txt')
    with open(file_path, 'a') as f:
        f.write(out + '\n')

    return -rmse

bounds = {'learning_rate': (0.001, 1.0),
          'depth': (1, 16),
          'l2_leaf_reg': (1, 100),
          'rsm': (0.001, 0.999)}

optimizer = BayesianOptimization(
    f=cb_reg_objective,
    pbounds=bounds,
    random_state=1)
 
matern_kernel = Matern()
optimizer.set_gp_params(kernel=matern_kernel, alpha=1e-5)

optimizer._acq = 'ei' 

optimizer.maximize(
    init_points=20,
    n_iter=200)

params = optimizer.max['params']
params['depth'] = int(params['depth'])
params['l2_leaf_reg'] = int(params['l2_leaf_reg']) 

cb_run = cb.CatBoostRegressor(random_state=0,
                              learning_rate=params['learning_rate'],
                              depth=params['depth'],
                              l2_leaf_reg=params['l2_leaf_reg'],
                              rsm=params['rsm'])

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
    
    cb_run.fit(X_train, y_train)
    y_tr_pred = cb_run.predict(X_train)
    y_te_pred = cb_run.predict(X_test)

    tr_rmse.append(mean_squared_error(y_tr_pred, y_train) ** 0.5)
    te_rmse.append(mean_squared_error(y_te_pred, y_test) ** 0.5)
    tr_mae.append(mean_absolute_error(y_tr_pred, y_train))
    te_mae.append(mean_absolute_error(y_te_pred, y_test))  

    tr_r2.append(r2_score(y_tr_pred, y_train))
    te_r2.append(r2_score(y_te_pred, y_test))
    
    print("Fold:", i+1)
     
best_idx1 = np.argmax(tr_rmse)
best_idx2 = np.argmax(te_rmse)
print("Best Fold:", best_idx1+1)

train = pd.DataFrame({'Experimental': y_train, 'Predicted': y_tr_pred, 'TrainTest': ['Train']*len(y_train)})
test = pd.DataFrame({'Experimental': y_test, 'Predicted': y_te_pred, 'TrainTest': ['Test']*len(y_test)})

results = pd.concat([train, test], ignore_index=True) 

save_path = '.../...'
file_path = os.path.join(save_path, 'CB2results.csv')
results.to_csv(file_path, index=False)

file_path = os.path.join(save_path, 'CB2_model.pkl')
with open(file_path, 'wb') as f:
    pickle.dump(cb_run, f)
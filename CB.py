import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
from catboost import CatBoostRegressor
import random

path1 = '.../training/...'
path2 = '.../MF/...'

def load_data():
    df = pd.read_csv(os.path.join(path1, 'trainingdata.csv'))
    MOF_construction = df.loc[:, 'name':'topo']
    Structure_info = df.loc[:, 'GLD':'AV']
    Absorb_env = df.loc[:, 'Tem':'Concentration']
    Capacity = df.loc[:, 'Capacity']
    Mfs1 = pd.read_csv(os.path.join(path2, 'Morgan.csv')).loc[:, '0':] 
    Mfs2 = pd.read_csv(os.path.join(path2, 'MACCS.csv')).loc[:, '0':] 

    Metal1 = pd.get_dummies(MOF_construction.loc[:, 'metal_smiles'])
    Topo = pd.get_dummies(MOF_construction.loc[:, 'topo'])
    Data = pd.concat([MOF_construction, Structure_info, Absorb_env, Mfs1, Metal1, Topo], axis=1)
    X = Data.drop(['name', 'metal_smiles', 'topo', 'GLD'], axis=1)
    Y = Capacity
    return np.array(X), np.array(Y)

def standardize_data(X_train, X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train.astype(np.float32), X_test.astype(np.float32)

def cb_reg_objective(learning_rate, depth, l2_leaf_reg, rsm):
    random_state = random.randint(0, 1000)
    params = {'learning_rate': learning_rate,
              'depth': int(depth),
              'l2_leaf_reg': int(l2_leaf_reg), 
              'rsm': rsm}

    model = CatBoostRegressor(**params, random_state=random_state)
    model.fit(X_train, y_train)
    preds = model.predict(X_train)
    mse = mean_squared_error(y_train, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_train, preds)
    
    save_path = '.../...'
    file_path = os.path.join(save_path, 'CB2optimization_output.md')
    with open(file_path, 'a') as f:
        f.write(f"Random State: {random_state}, RMSE: {rmse}, R2: {r2}\n")

    return -rmse

def run_bayesian_optimization(X_train, y_train):
    bounds = {'learning_rate': (0.001, 1.0),
              'depth': (1, 16),
              'l2_leaf_reg': (1, 100),
              'rsm': (0.001, 0.999)}

    optimizer = BayesianOptimization(
        f=cb_reg_objective,
        pbounds=bounds,
        random_state=1)

    optimizer.maximize(init_points=20, n_iter=200)

    params = optimizer.max['params']
    params['depth'] = int(params['depth'])
    params['l2_leaf_reg'] = int(params['l2_leaf_reg'])
    return params

def cross_validate_model(X, Y, params):
    cb_run = CatBoostRegressor(random_state=0,
                               learning_rate=params['learning_rate'],
                               depth=params['depth'],
                               l2_leaf_reg=params['l2_leaf_reg'],
                               rsm=params['rsm'])

    tr_rmse, te_rmse = [], [] 
    tr_r2, te_r2 = [], []
    tr_mae, te_mae = [], []

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    for i, (tr_idx, te_idx) in enumerate(kf.split(X, Y)):
        random_state = random.randint(0, 1000)
        X_train, X_test = X[tr_idx], X[te_idx]
        y_train, y_test = Y[tr_idx], Y[te_idx]
        
        X_train, X_test = standardize_data(X_train, X_test)
        
        cb_run.set_params(random_state=random_state)
        cb_run.fit(X_train, y_train)
        y_tr_pred = cb_run.predict(X_train)
        y_te_pred = cb_run.predict(X_test)

        tr_rmse.append(mean_squared_error(y_tr_pred, y_train) ** 0.5)
        te_rmse.append(mean_squared_error(y_te_pred, y_test) ** 0.5)
        tr_mae.append(mean_absolute_error(y_tr_pred, y_train))
        te_mae.append(mean_absolute_error(y_te_pred, y_test))  

        tr_r2.append(r2_score(y_tr_pred, y_train))
        te_r2.append(r2_score(y_te_pred, y_test))
        
        print(f"Fold: {i+1}, Random State: {random_state}")

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

if __name__ == "__main__":
    X, Y = load_data()
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

    X_train, X_test = standardize_data(X_train, X_test)
    params = run_bayesian_optimization(X_train, y_train)
    cross_validate_model(X, Y, params)
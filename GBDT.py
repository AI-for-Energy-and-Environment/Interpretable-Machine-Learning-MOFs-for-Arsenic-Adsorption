import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import ensemble
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
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

def bo_params_gb(max_depth, min_samples_leaf, min_samples_split, n_estimators, learning_rate):
    random_state = random.randint(0, 1000)
    params = {
        'max_depth': int(max_depth),
        'min_samples_leaf': int(min_samples_leaf),
        'min_samples_split': int(min_samples_split),
        'n_estimators': int(n_estimators),
        'learning_rate': learning_rate
    }
    reg = ensemble.GradientBoostingRegressor(random_state=random_state, **params)
    reg.fit(X_train, y_train)
    preds = reg.predict(X_train)
    mse = mean_squared_error(y_train, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_train, preds)
    
    save_path = '.../...'
    file_path = os.path.join(save_path, 'GBDT2optimization_output.md')
    with open(file_path, 'a') as f:
        f.write(f"Random State: {random_state}, RMSE: {rmse}, R2: {r2}\n")

    return -rmse

def run_bayesian_optimization(X_train, y_train):
    uf = UtilityFunction(kind='ei', kappa=2.5, xi=0.01)
    gb_BO = BayesianOptimization(
        f=bo_params_gb,
        pbounds={
            'max_depth': (60, 100),
            'min_samples_leaf': (1, 4),
            'min_samples_split': (2, 10),
            'n_estimators': (100, 250),
            'learning_rate': (0.001, 1.0)
        },
        random_state=1
    )
    gb_BO.maximize(n_iter=200, init_points=20, acquisition_function=uf)
    params = gb_BO.max['params']
    params['max_depth'] = int(params['max_depth'])
    params['min_samples_leaf'] = int(params['min_samples_leaf'])
    params['min_samples_split'] = int(params['min_samples_split'])
    params['n_estimators'] = int(params['n_estimators'])
    return params

def cross_validate_model(X, Y, params):
    gb_run = ensemble.GradientBoostingRegressor(
        random_state=0, **params
    )

    tr_rmse, te_rmse = [], [] 
    tr_r2, te_r2 = [], []
    tr_mae, te_mae = [], []

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    for i, (tr_idx, te_idx) in enumerate(kf.split(X, Y)):
        random_state = random.randint(0, 1000)
        X_train, X_test = X[tr_idx], X[te_idx]
        y_train, y_test = Y[tr_idx], Y[te_idx]
        
        X_train, X_test = standardize_data(X_train, X_test)
        
        gb_run.set_params(random_state=random_state)
        gb_run.fit(X_train, y_train)
        y_tr_pred = gb_run.predict(X_train)
        y_te_pred = gb_run.predict(X_test)

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
    file_path = os.path.join(save_path, 'GBDT2results.csv')
    results.to_csv(file_path, index=False)

    file_path = os.path.join(save_path, 'GBDT2_model.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(gb_run, f)

if __name__ == "__main__":
    X, Y = load_data()
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

    X_train, X_test = standardize_data(X_train, X_test)
    params = run_bayesian_optimization(X_train, y_train)
    cross_validate_model(X, Y, params)
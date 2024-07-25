import pandas as pd
import numpy as np
import os
import time
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import partial_dependence
from sklearn.inspection import PartialDependenceDisplay
from sklearn import ensemble
from sklearn import tree
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import pearsonr
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
from catboost import CatBoostRegressor
import catboost as cb

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys

import shap
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import FixedLocator, FixedFormatter
from mpl_toolkits.mplot3d import Axes3D

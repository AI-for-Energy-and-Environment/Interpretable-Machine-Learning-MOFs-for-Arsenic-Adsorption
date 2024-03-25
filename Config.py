import csv
import os
import random
import time

import numpy as np
import pandas as pd
import rdkit.Chem as Chem
import torch

from typing import List
from typing import Text
from torch_geometric.data import Data
from tqdm.auto import tqdm
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch import nn
from torch.nn import GRU
from torch_geometric.nn import Set2Set
import torch.nn.functional as F

import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler

def get_params(case):
    params = {}
    params["input_data"] = '.../metadata.csv'
    params["output_path"] = '.../outputs/'
    params["prop"] = ["Capacity"]
    params["struc"] = ["GLD", "PLD", "LCD", "q", "VSA", "GSA", "VF", "AV"]
    params["fact"] = ["Tem", "pH", "Dosage", "Concentration"] 

    params["rand_seed"] = 7

     # Data loader
    params["y_scaler"] = False
    params["run_state"] = False
    params["run_number"] = 12000

    # Neural network
    params["use_chem"] = True
    if not params["use_chem"]:
        params["output_path"] = ".../outputs/"
    params["use_struc"] = True
    params["use_fact"] = True

    # Random test state
    params["rand_test"] = False
    params["rand_cycle"] = 10
    if params["rand_test"]:
        params["output_path"] = ".../outputs/"

    # node2vec
    params["adj_gen_state"] = False
    params["n2v_emb_state"] = False
    params["use_n2v_emb"] = False

    params["embed_dim"] = 2

    # optimizer
    params["lr"] = 0.005
    params["epoch"] = 2000
    params["scheduler_patience"] = 5
    params["scheduler_factor"] = 0.9
    params["earlystop_er_patience"] = 10

    return params
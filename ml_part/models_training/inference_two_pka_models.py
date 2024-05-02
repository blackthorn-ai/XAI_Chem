# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import time
import json
import pickle
import pandas as pd
import numpy as np
import torch
import sys
import os
sys.path.append("..") 
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR

from utils import setup, get_configure, collate_molgraphs, load_dataset
from My_Pka_Model import Pka_acidic_view,Pka_basic_view

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt

from tqdm import tqdm


def load_pKa_acidic_model(args):
    pka1_model = Pka_acidic_view(node_feat_size = 74,
                                 edge_feat_size = 12,
                                 output_size = 1,
                                 num_layers= 6,
                                 graph_feat_size=200,
                                 dropout=0).to(args['device'])
    # pka1_model.load_state_dict(torch.load(r'ml_part\weights\pKa\site_acidic.pkl',map_location='cpu'))
    pka1_model.load_state_dict(torch.load(r'ml_part\weights\pKa\acid_best_loss_blooming-deluge-52.pkl',map_location='cpu'))
    return pka1_model


def load_pKa_basic_model(args):
    pka2_model = Pka_basic_view(node_feat_size = 74,
                                 edge_feat_size = 12,
                                 output_size = 1,
                                 num_layers= 6,
                                 graph_feat_size=200,
                                 dropout=0).to(args['device'])
    pka2_model.load_state_dict(torch.load(r'ml_part\weights\pKa\basic_best_loss_glamorous-totem-39.pkl',map_location='cpu'))
    return pka2_model


def calculate_metrics(true_values, pred_values):
    mse = round(sqrt(mean_squared_error(true_values, pred_values)),3)
    mae = round(mean_absolute_error(true_values, pred_values),3)
    r_score = round(r2_score(true_values, pred_values),3)

    return {"mse": mse,
            "mae": mae,
            "r_score": r_score,}


def train(model, _test_set):

    exp_config['batch_size'] = 32
    test_loader = DataLoader(dataset=_test_set, batch_size=exp_config['batch_size'],
                             collate_fn=collate_molgraphs, num_workers=args['num_workers'])

    model.eval()
    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            _, bg, labels, masks = batch_data
            model_prediction, _ = model(bg,bg.ndata['h'], bg.edata['e'])

    true_val_values = labels.view(-1).tolist()
    pred_val_values = model_prediction.view(-1).tolist()
    

    print("Pred:", pred_val_values)
    print("True pka:", true_val_values)

    val_metrics = calculate_metrics(true_val_values, pred_val_values)
    print(val_metrics)

   
if __name__ == '__main__':
    with open(r'ml_part\configs\args.pickle', 'rb') as file:
        args =pickle.load(file)
        args['task'] = ['pKa']
    with open(r'ml_part\configs\configure.json', 'r') as f:
        exp_config = json.load(f)
    args['device'] = torch.device('cpu')
    args = setup(args)

    acidic_model = load_pKa_basic_model(args)

    csv_path = r'C:\work\DrugDiscovery\main_git\XAI_Chem\data\init_data\pKa_Prediction_Starting data_2024.01.25.csv'
    csv_logP_test_path = r'C:\work\DrugDiscovery\main_git\XAI_Chem\data\logP_lipophilicity_data\gnn_cv\test.csv'

    df = pd.read_csv(csv_path, index_col=0)
    df_test_splited_by_tanimoto = pd.read_csv(csv_logP_test_path, index_col=0)

    logP_smiles = df_test_splited_by_tanimoto['Smiles'].tolist()
    pKa_smiles, pKa_values = [], []

    for index, row in df.iterrows():
        if row['Amides for LogP'] in logP_smiles:
            pKa_smiles.append(row['Smiles'])
            pKa_values.append(row['pKa'])

    test_set = pd.DataFrame({"smiles": pKa_smiles,
                             "pKa": pKa_values})

    test_set = load_dataset(args,test_set,"test")

    train(acidic_model, test_set)

# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import time
import json
import pickle
import pandas as pd
import numpy as np
import operator
import torch
import sys
import os
sys.path.append("..") 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR

from utils import setup, get_configure, collate_molgraphs, load_dataset
from My_Pka_Model import Pka_acidic_view,Pka_basic_view

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import top_k_accuracy_score
from math import sqrt

from rdkit import Chem
from rdkit.Chem import rdmolops
import dgl
from dgllife.utils import smiles_to_bigraph

from tqdm import tqdm


def load_pKa_acidic_model(args):
    pka1_model = Pka_acidic_view(node_feat_size = 74,
                                 edge_feat_size = 12,
                                 output_size = 1,
                                 num_layers= 6,
                                 graph_feat_size=200,
                                 dropout=0).to(args['device'])
    pka1_model.load_state_dict(torch.load(r'ml_part\weights\pKa\site_acidic_best_loss_distinctive-butterfly-11.pkl',map_location='cpu'))
    # pka1_model.load_state_dict(torch.load(r'ml_part\weights\pKa\site_acidic.pkl',map_location='cpu'))
    return pka1_model


def load_pKa_basic_model(args):
    pka2_model = Pka_basic_view(node_feat_size = 74,
                                 edge_feat_size = 12,
                                 output_size = 1,
                                 num_layers= 6,
                                 graph_feat_size=200,
                                 dropout=0).to(args['device'])
    pka2_model.load_state_dict(torch.load(r'ml_part\weights\pKa\site_amine_best_loss_sweet-capybara-11.pkl',map_location='cpu'))
    # pka2_model.load_state_dict(torch.load(r'ml_part\weights\pKa\site_basic.pkl',map_location='cpu'))
    return pka2_model


def evaluate(model, smiles, true_value, args, exp_config):
    bg = convert_smiles_to_dglgraph(smiles, args)
    
    model_prediction, _ = model(bg,bg.ndata['h'], bg.edata['e'])
    # print(f"SMILES: {smiles}, pred: {model_prediction}, true: {true_value}")
    return model_prediction.item()


def convert_smiles_to_dglgraph(smiles: str, args):
    dglgraph = smiles_to_bigraph(smiles=smiles,
                                 node_featurizer=args['node_featurizer'],
                                 edge_featurizer=args['edge_featurizer'])

    return dglgraph
    

def extract_smiles(csv_path):
    df = pd.read_csv(csv_path)
    return df['smiles'], df['pKa']


def top_k_error_metric(smiles, true, pred, k=5):
    error_dict = {smiles[i] : abs(true[i]-pred[i]) for i in range(len(true))}
    error_dict_sorted = dict( sorted(error_dict.items(), key=operator.itemgetter(1), reverse=True))

    top_k_error_dict = {}
    for i, key in enumerate(error_dict_sorted):
        if i >= k:
            break

        top_k_error_dict[key] = error_dict_sorted[key]

    return top_k_error_dict


def calculate_metrics(true_values, pred_values):
    mse = round(sqrt(mean_squared_error(true_values, pred_values)),3)
    mae = round(mean_absolute_error(true_values, pred_values),3)
    r_score = round(r2_score(true_values, pred_values),3)

    return {"mse": mse,
            "mae": mae,
            "r_score": r_score,}


pKa_amine_data_csv = r"C:\work\DrugDiscovery\RT_LogP_with_pKa_model\RTlogD\RD_dataset\test_pKa_amine_data.csv"
pKa_acid_data_csv = r"C:\work\DrugDiscovery\RT_LogP_with_pKa_model\RTlogD\RD_dataset\test_pKa_acid_data.csv"
# SMILES = "FC([H])(F)C1CNC1"
   
if __name__ == '__main__':
    with open(r'C:\work\DrugDiscovery\RT_LogP_with_pKa_model\RTlogD\final_model/RTlogD/args.pickle', 'rb') as file:
        args =pickle.load(file)
        args['task'] = ['pKa']
    with open(r'C:\work\DrugDiscovery\RT_LogP_with_pKa_model\RTlogD\final_model/RTlogD/configure.json', 'r') as f:
        exp_config = json.load(f)
    args['device'] = torch.device('cpu')
    args = setup(args)

    amine_model = load_pKa_basic_model(args)
    acid_model = load_pKa_acidic_model(args)

    amine_model.eval()
    acid_model.eval()

    smiles_amine, true_amine_pKa = extract_smiles(pKa_amine_data_csv)

    pred = []
    for smile_index in range(len(smiles_amine)):
        pred.append(evaluate(amine_model, smiles_amine[smile_index], true_amine_pKa[smile_index], args, exp_config))
    
    smiles_acid, true_acid_pKa = extract_smiles(pKa_acid_data_csv)
    for smile_index in range(len(smiles_acid)):
        pred.append(evaluate(acid_model, smiles_acid[smile_index], true_acid_pKa[smile_index], args, exp_config))
    
    smiles = [y for x in [smiles_amine, smiles_acid] for y in x]
    true_pKa = [y for x in [true_amine_pKa, true_acid_pKa] for y in x]

    print(calculate_metrics(true_pKa, pred))
    print(top_k_error_metric(smiles, true_pKa, pred))
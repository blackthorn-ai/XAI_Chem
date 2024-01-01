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
sys.path.insert(0, os.path.dirname(__file__))

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
import wandb


def init_wandb():
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="enamine-pKa-amine(basic)",
        
        # track hyperparameters and run metadata
        config={
            "learning_rate": 0.001,
            "batch_size": 32,
            "architecture": "GNN + head",
            "dataset": "pKa_amine_data.csv",
            "epochs": 200,
            "is GNN freezed": True,
            "is head freezed": False,
            "info": "new model freezing(gnn.eval, predict.train(True)) + with ReduceLROnPlateau"
        }
    )

    return run


def load_pKa_acidic_model(args):
    pka1_model = Pka_acidic_view(node_feat_size = 74,
                                 edge_feat_size = 12,
                                 output_size = 1,
                                 num_layers= 6,
                                 graph_feat_size=200,
                                 dropout=0).to(args['device'])
    pka1_model.load_state_dict(torch.load(r'C:\work\DrugDiscovery\RT_LogP_with_pKa_model\RTlogD\Trained_model/site_acidic_best_loss_distinctive-butterfly-11.pkl',map_location='cpu'))
    return pka1_model


def load_pKa_basic_model(args):
    pka2_model = Pka_basic_view(node_feat_size = 74,
                                 edge_feat_size = 12,
                                 output_size = 1,
                                 num_layers= 6,
                                 graph_feat_size=200,
                                 dropout=0).to(args['device'])
    pka2_model.load_state_dict(torch.load(r'C:\work\DrugDiscovery\RT_LogP_with_pKa_model\RTlogD\Trained_model/site_amine_best_loss_sweet-capybara-11.pkl',map_location='cpu'))
    return pka2_model


def calculate_metrics(true_values, pred_values):
    mse = round(sqrt(mean_squared_error(true_values, pred_values)),3)
    mae = round(mean_absolute_error(true_values, pred_values),3)
    r_score = round(r2_score(true_values, pred_values),3)

    return {"mse": mse,
            "mae": mae,
            "r_score": r_score,}


def evaluate(acid_model, amine_model, acid_set, amine_test_set):

    exp_config['batch_size'] = 512
    acid_loader = DataLoader(dataset=acid_set, batch_size=exp_config['batch_size'],
                             collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    amine_loader = DataLoader(dataset=amine_test_set, batch_size=exp_config['batch_size'],
                             collate_fn=collate_molgraphs, num_workers=args['num_workers'])


    with torch.no_grad():
        for i, batch_data in enumerate(acid_loader):
            _, bg, labels, masks = batch_data
            model_prediction, _ = acid_model(bg,bg.ndata['h'], bg.edata['e'])

        true_acid_values = labels.view(-1).tolist()
        pred_acid_values = model_prediction.view(-1).tolist()

    with torch.no_grad():
        for i, batch_data in enumerate(amine_loader):
            _, bg, labels, masks = batch_data
            model_prediction, _ = amine_model(bg,bg.ndata['h'], bg.edata['e'])

        true_amine_values = labels.view(-1).tolist()
        pred_amine_values = model_prediction.view(-1).tolist()

    true_values = true_acid_values + true_amine_values
    pred_values = pred_acid_values + pred_amine_values
    
    metrics = calculate_metrics(true_values, pred_values)
    print(true_values)
    print(pred_values)
    print(metrics)


   
if __name__ == '__main__':
    with open(r'C:\work\DrugDiscovery\RT_LogP_with_pKa_model\RTlogD\final_model/RTlogD/args.pickle', 'rb') as file:
        args =pickle.load(file)
        args['task'] = ['pKa']
    with open(r'C:\work\DrugDiscovery\RT_LogP_with_pKa_model\RTlogD\final_model/RTlogD/configure.json', 'r') as f:
        exp_config = json.load(f)
    args['device'] = torch.device('cpu')
    args = setup(args)

    acidic_model = load_pKa_acidic_model(args)
    amine_model = load_pKa_basic_model(args)

    train_amine_set = pd.read_csv(r'C:\work\DrugDiscovery\RT_LogP_with_pKa_model\RTlogD\RD_dataset\train_pKa_amine_data.csv')
    test_amine_set = pd.read_csv(r'C:\work\DrugDiscovery\RT_LogP_with_pKa_model\RTlogD\RD_dataset\test_pKa_amine_data.csv')

    train_acid_set = pd.read_csv(r'C:\work\DrugDiscovery\RT_LogP_with_pKa_model\RTlogD\RD_dataset\train_pKa_acid_data.csv')
    test_acid_set = pd.read_csv(r'C:\work\DrugDiscovery\RT_LogP_with_pKa_model\RTlogD\RD_dataset\test_pKa_acid_data.csv')

    train_amine_set = load_dataset(args,train_amine_set,"test")
    test_amine_set = load_dataset(args,test_amine_set,"test")

    train_acid_set = load_dataset(args,train_acid_set,"test")
    test_acid_set = load_dataset(args,test_acid_set,"test")

    exp_config = get_configure(args['model'],"test")

    evaluate(acidic_model, amine_model, test_acid_set, test_amine_set)

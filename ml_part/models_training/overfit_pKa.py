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
            "is overfit": True,
            "info": "overfit"
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
    pka1_model.load_state_dict(torch.load(r'ml_part\weights\pKa\combined_dataset\acid_best_loss_daily-morning-84.pkl',map_location='cpu'))
    return pka1_model


def load_pKa_basic_model(args):
    pka2_model = Pka_basic_view(node_feat_size = 74,
                                 edge_feat_size = 12,
                                 output_size = 1,
                                 num_layers= 6,
                                 graph_feat_size=200,
                                 dropout=0).to(args['device'])
    pka2_model.load_state_dict(torch.load(r'ml_part\weights\pKa\site_basic.pkl',map_location='cpu'))
    return pka2_model


def calculate_metrics(true_values, pred_values):
    mse = round(sqrt(mean_squared_error(true_values, pred_values)),3)
    mae = round(mean_absolute_error(true_values, pred_values),3)
    r_score = round(r2_score(true_values, pred_values),3)

    return {"mse": mse,
            "mae": mae,
            "r_score": r_score,}


def train(model, _train_set, num_epochs=5000, use_wandb=True, save_best_model=True):
    run_name = ""
    if use_wandb is True:
        run = init_wandb()
        run_name = run.name
    exp_config['batch_size'] = 200
    train_loader = DataLoader(dataset=_train_set, batch_size=exp_config['batch_size'],
                              collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.0017657)
    # optimizer_sgd = SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=10, factor=0.2, min_lr=0.000000002)
    # cyclic_lr_scheduler = CyclicLR(optimizer=optimizer_sgd, base_lr=0.001, max_lr=0.01)

    best_loss = pow(10, 3)
    for epoch in tqdm(range(num_epochs)):
        # model.freeze_except_predict()
        running_loss = 0.0
        for i, batch_data in enumerate(train_loader):
            _, bg, labels, masks = batch_data
            optimizer.zero_grad()

            model_prediction, _ = model(bg,bg.ndata['h'], bg.edata['e'])
            loss = criterion(model_prediction, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
        avg_loss = running_loss / (i + 1)

        scheduler.step(avg_loss)

        if best_loss > avg_loss:
            best_loss = avg_loss
            if save_best_model is True:
                torch.save(model.state_dict(), rf'ml_part\weights\pKa\overfit_amine_best_loss_{run_name}.pkl')

        true_train_values = labels.view(-1).tolist()
        pred_train_values = model_prediction.view(-1).tolist()
        train_metrics = calculate_metrics(true_train_values, pred_train_values)

        lr = optimizer.param_groups[0]['lr']

        if use_wandb is True:
            wandb.log({"loss/train": avg_loss, 
                        "lr": lr,
                        "mse/train": train_metrics['mse'],
                        "mae/train": train_metrics['mae'],
                        "r^2/train": train_metrics['r_score']
            })
        print(f"LOSS train: {loss}, lr: {lr}, mse/train: {train_metrics['mse']}, mae/train: {train_metrics['mae']}, r^2/train: {train_metrics['r_score']}")

   
if __name__ == '__main__':
    with open(r'ml_part\configs\args.pickle', 'rb') as file:
        args =pickle.load(file)
        args['task'] = ['pKa']
    with open(r'ml_part\configs\configure.json', 'r') as f:
        exp_config = json.load(f)
    args['device'] = torch.device('cpu')
    args['smiles_column'] = 'Smiles'
    args = setup(args)

    acidic_model = load_pKa_acidic_model(args)

    train_set = pd.read_csv(r'data\pKa_basicity_data\gnn_cv_canon_smiles\train_basic.csv')
    test_set = pd.read_csv(r'data\pKa_basicity_data\gnn_cv_canon_smiles\test_basic.csv')

    train_set = pd.concat([train_set, test_set], axis=0)

    train_set = load_dataset(args,train_set,"test")

    exp_config = get_configure(args['model'],"test")

    train(acidic_model, train_set)

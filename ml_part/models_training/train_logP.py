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
from torch.nn import SmoothL1Loss, MSELoss
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR

from utils import setup, get_configure, collate_molgraphs, load_dataset, load_model, predict
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
        project="enamine-logP",
        
        # track hyperparameters and run metadata
        config={
            "learning_rate": 0.001,
            "batch_size": 32,
            "architecture": "AttentiveFPRegressor + default pKa basic, pKa acid",
            "dataset": "logP_data.csv",
            "epochs": 200,
            "is AttentiveFPRegressor freezed": False,
            "optimizer": "Adam(lr=0.001, weight_decay=0.9)",
            "loss": "MSELoss",
            "scheduler": "ReduceLROnPlateau(mode='min', patience=100, factor=0.5, min_lr=0.00002)",
            "info": "gnn.train(True), readout.train(True), shuffle=True"
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
    pka1_model.load_state_dict(torch.load(r'ml_part\weights\pKa\site_acidic.pkl',map_location='cpu'))
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


def train(model, _train_set, _test_set, num_epochs=200, use_wandb=True, save_best_model=True):
    run_name = ""
    if use_wandb is True:
        run = init_wandb()
        run_name = run.name
    exp_config['batch_size'] = 32
    train_loader = DataLoader(dataset=_train_set, batch_size=exp_config['batch_size'],
                             collate_fn=collate_molgraphs, num_workers=args['num_workers'], shuffle=True)
    test_loader = DataLoader(dataset=_test_set, batch_size=exp_config['batch_size'],
                             collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.9)
    criterion = MSELoss()
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=100, factor=0.5, min_lr=0.00002)

    best_vloss = pow(10, 3)
    for epoch in tqdm(range(num_epochs)):
        model.freeze_model_for_finetune()
        running_loss = 0.0
        running_vloss = 0.0
        train_pred, train_true = torch.Tensor([]), torch.Tensor([])
        val_pred, val_true = torch.Tensor([]), torch.Tensor([])
        for i, batch_data in enumerate(train_loader):
            _, bg, labels, masks = batch_data
            optimizer.zero_grad()

            prediction, pka_acidic_prediction, pka_basic_prediction = predict(args=args, model=model, bg=bg, default_weights=True)
            logP_pred = prediction[:, 1].view(-1, 1)
            loss = criterion(logP_pred, labels)
            loss.backward()

            optimizer.step()
            

            train_pred = torch.cat((train_pred, logP_pred), dim=0)
            train_true = torch.cat((train_true, labels))

            running_loss += loss.item()
        avg_loss = running_loss / i

        true_train_values = train_true.view(-1).tolist()
        pred_train_values = train_pred.view(-1).tolist()
        train_metrics = calculate_metrics(true_train_values, pred_train_values)

        model.eval()
        with torch.no_grad():
            for i, batch_data in enumerate(test_loader):
                _, bg, labels, masks = batch_data
                prediction, pka_acidic_prediction, pka_basic_prediction = predict(args, model, bg)
                logP_pred = prediction[:, 1].view(-1, 1)
                vloss = criterion(logP_pred, labels)

                val_pred = torch.cat((val_pred, logP_pred), dim=0)
                val_true = torch.cat((val_true, labels))
                
                running_vloss += vloss.item()

        scheduler.step(avg_vloss)
        avg_vloss = running_vloss / (i + 1)

        true_val_values = val_true.view(-1).tolist()
        pred_val_values = val_pred.view(-1).tolist()
        val_metrics = calculate_metrics(true_val_values, pred_val_values)

        lr = optimizer.param_groups[0]['lr']
        if avg_vloss < best_vloss and save_best_model is True:
            torch.save(model.state_dict(), rf'ml_part\weights\logP\logP_RTLogD_best_loss_{run_name}.pth')
            best_vloss = avg_vloss


        if use_wandb is True:
            wandb.log({"loss/train": avg_loss, 
                        "loss/val": avg_vloss, 
                        "lr": lr,
                        "mse/train": train_metrics['mse'],
                        "mae/train": train_metrics['mae'],
                        "r^2/train": train_metrics['r_score'],
                        "mse/val": val_metrics['mse'],
                        "mae/val": val_metrics['mae'],
                        "r^2/val": val_metrics['r_score']})
        print('LOSS train: {} valid: {}, lr: {}'.format(avg_loss, avg_vloss, lr))



def main(train_set, test_set, args, exp_config):
    exp_config.update({
        'model': args['model'],
        'mode':args['mode'],
        'n_tasks': args['n_tasks'],
        'atom_featurizer_type': args['atom_featurizer_type'],
        'bond_featurizer_type': args['bond_featurizer_type']
    })

    if args['atom_featurizer_type'] != 'pre_train':
        exp_config['in_node_feats'] = args['node_featurizer'].feat_size()+2
    if args['edge_featurizer'] is not None and args['bond_featurizer_type'] != 'pre_train':
        exp_config['in_edge_feats'] = args['edge_featurizer'].feat_size()

    model = load_model(exp_config).to(args['device'])
    checkpoint = torch.load(r"ml_part\weights\logP\model_pretrain_76_default_weights.pth",map_location=torch.device('cpu'))#my_model/CRT76-logD/model_pretrain_76.pth"
    model.load_state_dict(checkpoint)

    train(model, train_set, test_set)
   
if __name__ == '__main__':
    with open(r'ml_part\configs\args.pickle', 'rb') as file:
        args =pickle.load(file)
        args['task'] = ['logP']
    with open(r'ml_part\configs\configure.json', 'r') as f:
        exp_config = json.load(f)
    args['device'] = torch.device('cpu')
    args = setup(args)

    acidic_model = load_pKa_basic_model(args)

    train_set = pd.read_csv(r'data\logP_lipophilicity_data\gnn_cv\train.csv')
    test_set = pd.read_csv(r'data\logP_lipophilicity_data\gnn_cv\test.csv')

    train_set = load_dataset(args,train_set,"test")
    test_set = load_dataset(args,test_set,"test")
    exp_config = get_configure(args['model'],"test")

    main(train_set, test_set, args, exp_config)

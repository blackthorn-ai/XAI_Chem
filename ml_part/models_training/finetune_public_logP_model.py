# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import time
import json
import pickle
import pandas as pd
import numpy as np
from functools import partial
import torch
import sys
import os
sys.path.append("..") 
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR

from utils import setup, get_configure, collate_molgraphs, load_dataset, load_model, predict
from My_Pka_Model import Pka_acidic_view,Pka_basic_view

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt

from hyperopt import fmin, tpe, hp

from tqdm import tqdm
import wandb

crossvals_list = []

class EarlyStopper:
    def __init__(self, patience=20, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = pow(10,10)

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > self.min_validation_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class Trainer:

    def __init__(self) -> None:
        self.space = {
            'lr': hp.loguniform('lr', -10, -3),
            'weight_decay': hp.loguniform('weight_decay', -10, -1),
            'train_type': hp.choice('train_type', ['only_predictor', 'predictor_and_readout'])
        }

        with open(r'ml_part\configs\args.pickle', 'rb') as file:
            self.args =pickle.load(file)
            self.args['task'] = ['logP']
        with open(r'ml_part\configs\configure.json', 'r') as f:
            exp_config = json.load(f)
        self.args['device'] = torch.device('cpu')
        self.args = setup(self.args)

        self.args['smiles_column'] = "Smiles"

        exp_config = get_configure(self.args['model'],"test")

        self.train_set = pd.read_csv(r'data\logP_lipophilicity_data\gnn_cv\train.csv')
        self.test_set = pd.read_csv(r'data\logP_lipophilicity_data\gnn_cv\test.csv')

        # best_hyperparameters = self.find_best_params_with_hyperopt()
        best_hyperparameters = {'lr': 0.00014238586440724804, 'train_type': 'only_predictor', 'weight_decay': 0.004780988028140225}

        print(crossvals_list)
        cv_file = pd.DataFrame(crossvals_list)
        # cv_file.to_csv(r'ml_part\models_training\saved_cv_csv\logP_cv.csv')

        with open(r'ml_part\configs\configure.json', 'r') as f:
            exp_config = json.load(f)
        model = load_model(exp_config).to(self.args['device'])
        checkpoint = torch.load(r"ml_part\weights\logP\model_pretrain_76_default_weights.pth",map_location=torch.device('cpu'))#my_model/CRT76-logD/model_pretrain_76.pth"
        model.load_state_dict(checkpoint)

        train_set = load_dataset(self.args, self.train_set, "test")
        test_set = load_dataset(self.args, self.test_set, "test")
        
        loss_cv, metrics_oos, metrics_train, best_val_pred_values, best_train_pred_values, true_val_values, true_train_values = Trainer.train(args=self.args,
            model=model,  
            _train_set=train_set, 
            _test_set=test_set,
            lr=best_hyperparameters['lr'],
            weight_decay=best_hyperparameters['weight_decay'],
            train_type=best_hyperparameters['train_type'],
            save_best_model=True
        )
        
        print(f"OOS: {metrics_oos}")
        print(f"Train: {metrics_train}")
        print(f"True train values: {true_train_values}")
        print(f"Pred train values: {best_train_pred_values}")
        print(f"True val values: {true_val_values}")
        print(f"Pred val values: {best_val_pred_values}")

    @staticmethod
    def init_wandb(
        train_mode: str = None, batch_size: int = 32,
        lr: float = 10 ** (-2.5),
        weight_decay: float = 10 ** (-5.0),
        model_name: str = "",
        epochs: int = 200
    ):
        run = wandb.init(
            project="enamine-logP",

            config={
                "learning_rate": lr,
                "train_mode": train_mode,
                "batch_size": batch_size,
                "architecture": model_name,
                "dataset": "data\logP_lipophilicity_data",
                "epochs": epochs,
                "optimizer": f"Adam(lr={lr}, weight_decay={weight_decay})",
                "loss": "MSELoss",
                "scheduler": "ReduceLrOnPlateu(mode='min', patience=10, factor=0.5, min_lr=0.000002)",
                "EarlyStoppingPatience": 20
            }
        )

        return run.name

    def find_best_params_with_hyperopt(self):
        algo = tpe.suggest

        objective_partial = partial(Trainer.optimization_function, X_train=self.train_set)

        best_hyperparams = fmin(fn=objective_partial, space=self.space, algo=algo, max_evals=10, verbose=1)

        print("Найкращі гіперпараметри:", best_hyperparams)
        return best_hyperparams
    
    @staticmethod
    def optimization_function(params, X_train):
        global crossvals_list

        amount_of_cross_vals = 2

        cv_indices_dict = {0: [], 1: []}
        for index, row in X_train.iterrows():
            cv_indices_dict[row['fold_id']].append(index)

        cv_indices = [[cv_indices_dict[0], cv_indices_dict[1]], [cv_indices_dict[1], cv_indices_dict[0]]]

        cv_dict = {"params": params}

        total_loss = 0
        for cross_val_index in range(amount_of_cross_vals):
            # df.loc[df.index[index_list]]
            train_df_cv = X_train.loc[cv_indices[cross_val_index][0]]

            test_df_cv = X_train.loc[cv_indices[cross_val_index][1]]

            with open(r'ml_part\configs\args.pickle', 'rb') as file:
                args =pickle.load(file)
                args['task'] = ['logP']
            with open(r'ml_part\configs\configure.json', 'r') as f:
                exp_config = json.load(f)
            args['device'] = torch.device('cpu')
            args = setup(args)

            args['smiles_column'] = "Smiles"

            model = load_model(exp_config).to(args['device'])
            checkpoint = torch.load(r"ml_part\weights\logP\model_pretrain_76_default_weights.pth",map_location=torch.device('cpu'))#my_model/CRT76-logD/model_pretrain_76.pth"
            model.load_state_dict(checkpoint)
            
            train_set = load_dataset(args,train_df_cv,"test")
            test_set = load_dataset(args,test_df_cv,"test")

            loss_cv, metrics_cv, metrics_cv_train, best_val_pred_values, best_train_pred_values, true_val_values, true_train_values = Trainer.train(args=args,
                                                                  model=model,  
                                                                  _train_set=train_set, 
                                                                  _test_set=test_set,
                                                                  save_best_model=False,
                                                                  lr=params['lr'],
                                                                  weight_decay=params['weight_decay'],
                                                                  train_type=params['train_type']
                                                                  )

            cv_dict[f"cv_{cross_val_index}_loss"] = loss_cv
            cv_dict[f"cv_{cross_val_index}_r^2"] = metrics_cv["r_score"]
            cv_dict[f"cv_{cross_val_index}_mse"] = metrics_cv["mse"]
            cv_dict[f"cv_{cross_val_index}_mae"] = metrics_cv["mae"]
            cv_dict[f"cv_{cross_val_index}_true_values"] = true_val_values
            cv_dict[f"cv_{cross_val_index}_pred_values"] = best_val_pred_values
            cv_dict[f"cv_{cross_val_index}_train_r^2"] = metrics_cv_train["r_score"]
            cv_dict[f"cv_{cross_val_index}_train_mse"] = metrics_cv_train["mse"]
            cv_dict[f"cv_{cross_val_index}_train_mae"] = metrics_cv_train["mae"]
            cv_dict[f"cv_{cross_val_index}_train_true_values"] = true_train_values
            cv_dict[f"cv_{cross_val_index}_train_pred_values"] = best_train_pred_values

            total_loss += loss_cv

        crossvals_list.append(cv_dict)
        return total_loss


    @staticmethod
    def load_pKa_acidic_model(args):
        pka1_model = Pka_acidic_view(node_feat_size = 74,
                                    edge_feat_size = 12,
                                    output_size = 1,
                                    num_layers= 6,
                                    graph_feat_size=200,
                                    dropout=0).to(args['device'])
        pka1_model.load_state_dict(torch.load(r'ml_part\weights\pKa\site_acidic.pkl',map_location='cpu'))
        return pka1_model


    @staticmethod
    def load_pKa_basic_model(args):
        pka2_model = Pka_basic_view(node_feat_size = 74,
                                    edge_feat_size = 12,
                                    output_size = 1,
                                    num_layers= 6,
                                    graph_feat_size=200,
                                    dropout=0).to(args['device'])
        pka2_model.load_state_dict(torch.load(r'ml_part\weights\pKa\site_basic.pkl',map_location='cpu'))
        return pka2_model


    @staticmethod
    def calculate_metrics(true_values, pred_values):
        mse = round(mean_squared_error(true_values, pred_values),3)
        mae = round(mean_absolute_error(true_values, pred_values),3)
        r_score = round(r2_score(true_values, pred_values),3)

        return {"mse": mse,
                "mae": mae,
                "r_score": r_score,}


    @staticmethod
    def train(args, model, _train_set, _test_set, num_epochs=200, use_wandb=True, save_best_model=True,
              lr=10**(-2.5), weight_decay=0.0006897, batch_size=150, train_type: str = None,):
        run_name = ""
        if use_wandb is True:
            run_name = Trainer.init_wandb(train_mode=train_type,
                                        batch_size=batch_size,
                                        lr=lr,
                                        weight_decay=weight_decay,
                                        model_name="logP attentive fp canonical basic",
                                        epochs=num_epochs)
        batch_size = batch_size
        train_loader = DataLoader(dataset=_train_set, batch_size=batch_size,
                                collate_fn=collate_molgraphs, num_workers=args['num_workers'])
        test_loader = DataLoader(dataset=_test_set, batch_size=batch_size,
                                collate_fn=collate_molgraphs, num_workers=args['num_workers'])
        
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=10, factor=0.5, min_lr=0.000002)
        early_stopper = EarlyStopper(patience=20, min_delta=0.001)

        best_vloss = pow(10, 10)
        best_train_metrics, best_val_metrics = None, None
        best_train_pred_values, best_val_pred_values = None, None
        for epoch in tqdm(range(num_epochs)):
            model.train_mode(train_type)
            running_loss = 0.0
            running_vloss = 0.0
            for i, batch_data in enumerate(train_loader):
                _, bg, labels, masks = batch_data
                optimizer.zero_grad()

                prediction, pka_acidic_prediction, pka_basic_prediction = predict(args=args, model=model, bg=bg, default_weights=True)
                model_prediction = prediction[:, 1].view(-1, 1)
                loss = criterion(model_prediction, labels)
                loss.backward()

                optimizer.step()
                scheduler.step(loss)

                running_loss += loss.item()
            avg_loss = running_loss / (i + 1)

            true_train_values = labels.view(-1).tolist()
            pred_train_values = model_prediction.view(-1).tolist()
            train_metrics = Trainer.calculate_metrics(true_train_values, pred_train_values)

            model.eval()
            with torch.no_grad():
                for i, batch_data in enumerate(test_loader):
                    _, bg, labels, masks = batch_data
                    prediction, pka_acidic_prediction, pka_basic_prediction = predict(args=args, model=model, bg=bg, default_weights=True)
                    model_prediction = prediction[:, 1].view(-1, 1)
                    vloss = criterion(model_prediction, labels)
                    
                    running_vloss += vloss.item()
            avg_vloss = running_vloss / (i + 1)

            true_val_values = labels.view(-1).tolist()
            pred_val_values = model_prediction.view(-1).tolist()
            val_metrics = Trainer.calculate_metrics(true_val_values, pred_val_values)

            lr = optimizer.param_groups[0]['lr']
            if avg_vloss < best_vloss:
                if save_best_model is True:
                    torch.save(model.state_dict(), rf'ml_part\weights\logP\logP_best_loss_{run_name}.pkl')
                best_train_metrics = train_metrics
                best_val_metrics = val_metrics
                best_val_pred_values = pred_val_values.copy()
                best_train_pred_values = pred_train_values.copy()
                pass

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
                
            is_early_stop = early_stopper.early_stop(avg_vloss)
            if is_early_stop:
                break

            print('LOSS train: {} valid: {}, lr: {}'.format(loss, vloss, lr))

        if use_wandb is True:
            wandb.finish()

        return avg_vloss, best_val_metrics, best_train_metrics, best_val_pred_values, best_train_pred_values, true_val_values, true_train_values
   
if __name__ == '__main__':
    
    Trainer()

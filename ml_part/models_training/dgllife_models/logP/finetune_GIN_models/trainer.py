from functools import partial

import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from hyperopt import fmin, tpe, hp

from model import LipophilicityModelService
from utils import load_model, load_dataset, collate_molgraphs, calculate_metrics, predict, init_wandb

crossvals_list = []
train_list = []

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
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class Trainer:
    def __init__(self) -> None:
        global link_to_colab_directory
        self.space = {
            'lr': hp.loguniform('lr', np.log(0.0001), np.log(0.05)),
            'weight_decay': hp.loguniform('weight_decay', -10, -1),
            'train_type': hp.choice('train_type', ['only_predictor', 'predictor_and_readout', 'all_layers'])
        }
        train_csv = r'data\logP_lipophilicity_data\gnn_cv\train.csv'
        test_csv = r'data\logP_lipophilicity_data\gnn_cv\test.csv'

        self.train_df = pd.read_csv(train_csv, index_col=0)
        self.test_df = pd.read_csv(test_csv, index_col=0)

        # models_names = ['AttentiveFP_canonical_Lipophilicity', 'Weave_attentivefp_Lipophilicity', 'GCN_attentivefp_Lipophilicity', 'GAT_attentivefp_Lipophilicity', 'AttentiveFP_attentivefp_Lipophilicity', 'Weave_canonical_Lipophilicity', 'GCN_canonical_Lipophilicity', 'GAT_canonical_Lipophilicity',]
        models_names = ['gin_supervised_contextpred_Lipophilicity', 'gin_supervised_infomax_Lipophilicity', 'gin_supervised_edgepred_Lipophilicity', 'gin_supervised_masking_Lipophilicity']
        for model_name in tqdm(models_names, desc="Outer Loop"):
        # for model_name in models_names:

            best_hyperparams = self.find_best_params_with_hyperopt(model_name)

            model_service_cv = LipophilicityModelService(model_name=model_name)

            train_set_cv = load_dataset(self.train_df, features_type=model_name)
            test_set_cv = load_dataset(self.test_df, features_type=model_name)
            loss_cv, metrics_cv, best_tmetrics, best_epoch_num, run = Trainer.train(model_service_cv, train_set_cv, test_set_cv,
                                                                     lr=best_hyperparams['lr'],
                                                                     weight_decay=best_hyperparams['weight_decay'],
                                                                     train_type=best_hyperparams['train_type'],
                                                                     save_best_model=True, model_name=model_name)

            train_dict = {}
            train_dict["model_name"] = model_name
            train_dict[f"train_r^2"] = best_tmetrics["r^2"]
            train_dict[f"train_mse"] = best_tmetrics["mse"]
            train_dict[f"train_mae"] = best_tmetrics["mae"]
            train_dict[f"oos_r^2"] = metrics_cv["r^2"]
            train_dict[f"oos_mse"] = metrics_cv["mse"]
            train_dict[f"oos_mae"] = metrics_cv["mae"]
            global train_list
            train_list.append(train_dict)

            global crossvals_list
            df = pd.DataFrame(crossvals_list)
            df.to_csv(link_to_colab_directory + rf'/{model_name}_cv_result_23_04.csv')

        df = pd.DataFrame(train_list)
        df.to_csv(link_to_colab_directory + rf'/best_results_for_MPNN_models.csv')


    def find_best_params_with_hyperopt(self, model_name):
        algo = tpe.suggest

        objective_partial = partial(Trainer.optimization_function, X_train=self.train_df, model_name=model_name)

        best_hyperparams = fmin(fn=objective_partial, space=self.space, algo=algo, max_evals=20, verbose=1)

        print("Найкращі гіперпараметри:", best_hyperparams)
        return best_hyperparams


    @staticmethod
    def optimization_function(params, X_train, model_name):
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

            model_service_cv = LipophilicityModelService(model_name=model_name)

            train_set_cv = load_dataset(train_df_cv, features_type=model_name)
            test_set_cv = load_dataset(test_df_cv, features_type=model_name)
            model_service_cv.train_mode(train_type=params['train_type'])
            loss_cv, metrics_cv, best_tmetrics, best_epoch_num, run = Trainer.train(model_service_cv, train_set_cv, test_set_cv,
                                                                                    lr=params['lr'],
                                                                                    weight_decay=params['weight_decay'],
                                                                                    train_type=params['train_type'])

            cv_dict[f"cv_{cross_val_index}_loss"] = loss_cv
            cv_dict[f"cv_{cross_val_index}_best_epoch"] = best_epoch_num
            cv_dict[f"cv_{cross_val_index}_run_name"] = run

            cv_dict[f"cv_{cross_val_index}_r^2"] = metrics_cv["r^2"]
            cv_dict[f"cv_{cross_val_index}_mse"] = metrics_cv["mse"]
            cv_dict[f"cv_{cross_val_index}_mae"] = metrics_cv["mae"]

            cv_dict[f"cv_{cross_val_index}_train_r^2"] = best_tmetrics["r^2"]
            cv_dict[f"cv_{cross_val_index}_train_mse"] = best_tmetrics["mse"]
            cv_dict[f"cv_{cross_val_index}_train_mae"] = best_tmetrics["mae"]

            total_loss += loss_cv

        crossvals_list.append(cv_dict)
        return total_loss



    @staticmethod
    def train(model_service, train_set, test_set,
              lr=10**(-2.5), weight_decay=0.0006897, batch_size=32,
              train_type: str = None,
              num_epochs=2000, use_wandb=False, save_best_model=False, model_name: str = None):
        global link_to_colab_directory

        if use_wandb:
            run_name = init_wandb(train_mode=train_type,
                                  batch_size=batch_size,
                                  lr=lr,
                                  weight_decay=weight_decay,
                                  model_name=model_name,
                                  epochs=num_epochs)

        model = model_service.model

        train_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                                  collate_fn=collate_molgraphs, num_workers=1, shuffle=True)
        test_loader = DataLoader(dataset=test_set, batch_size=batch_size,
                                 collate_fn=collate_molgraphs, num_workers=1)

        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = MSELoss()
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=10, factor=0.5, min_lr=0.000002)
        early_stopper = EarlyStopper(patience=20, min_delta=0.001)

        run_name="abc"
        best_vloss = pow(10, 20)
        best_vmetrics = None
        best_tmetrics = None
        best_vepoch = -1
        # model_service.train_mode(train_type=train_type)
        for epoch in tqdm(range(num_epochs), desc=f"Train {run_name} with lr: {lr}, train mode: {train_type}, weight decay: {weight_decay}, batch size: {batch_size}"):
        # for epoch in tqdm(range(num_epochs), leave=True):
        # for epoch in range(num_epochs):
            running_loss = 0.0
            running_vloss = 0.0
            train_pred, train_true = torch.Tensor([]).cuda(), torch.Tensor([]).cuda()
            val_pred, val_true = torch.Tensor([]).cuda(), torch.Tensor([]).cuda()
            model = model_service.train_mode(model=model, train_type=train_type)
            for i, batch_data in enumerate(train_loader):
                _, bg, labels, masks = batch_data

                if torch.cuda.is_available():
                    bg = bg.to('cuda')
                    labels = labels.to('cuda')

                optimizer.zero_grad()

                logP_pred = predict(model=model,
                                    bg=bg)

                loss = criterion(logP_pred, labels)
                loss.backward()

                optimizer.step()

                train_pred = torch.cat((train_pred, logP_pred), dim=0)
                train_true = torch.cat((train_true, labels))

                running_loss += loss.item()
            avg_loss = running_loss / (i + 1)

            # scheduler.step(avg_loss)

            true_train_values = train_true.view(-1).tolist()
            pred_train_values = train_pred.view(-1).tolist()
            train_metrics = calculate_metrics(true_train_values, pred_train_values)

            model = model_service.eval_mode(model=model)
            with torch.no_grad():
                for i, batch_data in enumerate(test_loader):
                    _, bg, labels, masks = batch_data
                    if torch.cuda.is_available():
                        bg = bg.to('cuda')
                        labels = labels.to('cuda')

                    logP_pred = predict(model, bg)
                    vloss = criterion(logP_pred, labels)

                    val_pred = torch.cat((val_pred, logP_pred), dim=0)
                    val_true = torch.cat((val_true, labels))

                    running_vloss += vloss.item()
            avg_vloss = running_vloss / (i + 1)

            true_val_values = val_true.view(-1).tolist()
            pred_val_values = val_pred.view(-1).tolist()
            val_metrics = calculate_metrics(true_val_values, pred_val_values)

            lr = optimizer.param_groups[0]['lr']
            if avg_vloss < best_vloss:
                # torch.save(model.state_dict(), rf'ml_part\weights\logP_dgllife_lipophilicity\logP_best_loss_{run_name.name}.pth')
                best_vloss = avg_vloss
                best_vmetrics = val_metrics
                best_tmetrics = train_metrics
                best_vepoch = epoch
                if save_best_model:
                    # torch.save(model.state_dict(), link_to_colab_directory + rf'/{type(model).__name__}_logP_best_loss.pth')
                    torch.save(model.state_dict(), link_to_colab_directory + rf'/{model_name}_logP_best_loss.pth')

            if use_wandb is True:
                wandb.log({"loss/train": avg_loss,
                            "loss/val": avg_vloss,
                            "lr": lr,
                            "mse/train": train_metrics['mse'],
                            "mae/train": train_metrics['mae'],
                            "r^2/train": train_metrics['r^2'],
                            "mse/val": val_metrics['mse'],
                            "mae/val": val_metrics['mae'],
                            "r^2/val": val_metrics['r^2']})

            is_early_stop = early_stopper.early_stop(avg_vloss)
            if is_early_stop:
                break

            scheduler.step(avg_vloss)

        wandb.finish()
        return best_vloss, best_vmetrics, best_tmetrics, best_vepoch, run_name

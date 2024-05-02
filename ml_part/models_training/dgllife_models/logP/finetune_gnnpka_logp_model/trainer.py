from functools import partial

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
        self.space = {
            # 'lr': hp.loguniform('lr', -10, -3),
            'train_type': hp.choice('train_type', ['only_predictor', 'predictor_and_readout'])
        }

        train_csv = r'data\logP_lipophilicity_data\gnn_cv\train.csv'
        test_csv = r'data\logP_lipophilicity_data\gnn_cv\test.csv'
        
        self.train_df = pd.read_csv(train_csv, index_col=0)
        self.test_df = pd.read_csv(test_csv, index_col=0)

        models_names = ['MPNN_canonical_Lipophilicity', 'MPNN_attentivefp_Lipophilicity', ]
        # models_names = ['AttentiveFP_canonical_Lipophilicity', 'GAT_canonical_Lipophilicity', ]
        for model_name in models_names:

            self.find_best_params_with_hyperopt(model_name)


    def find_best_params_with_hyperopt(self, model_name):
        algo = tpe.suggest

        objective_partial = partial(Trainer.optimization_function, X_train=self.train_df, model_name=model_name)

        best_hyperparams = fmin(fn=objective_partial, space=self.space, algo=algo, max_evals=3, verbose=1)

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

            train_set_cv = load_dataset(train_df_cv)
            test_set_cv = load_dataset(test_df_cv)
            loss_cv, metrics_cv, best_epoch_num, run = Trainer.train(model_service_cv, train_set_cv, test_set_cv,
                                                                          train_type=params['train_type'])
            
            cv_dict[f"cv_{cross_val_index}_loss"] = loss_cv
            cv_dict[f"cv_{cross_val_index}_best_epoch"] = best_epoch_num
            cv_dict[f"cv_{cross_val_index}_run_name"] = run.name
            cv_dict[f"cv_{cross_val_index}_r^2"] = metrics_cv["r^2"]
            cv_dict[f"cv_{cross_val_index}_mse"] = metrics_cv["mse"]
            cv_dict[f"cv_{cross_val_index}_mae"] = metrics_cv["mae"]

            total_loss += loss_cv

        crossvals_list.append(cv_dict)
        return total_loss

        

    @staticmethod
    def train(model_service, train_set, test_set, 
              lr=10**(-2.5), weight_decay=0.0006897, batch_size=16,
              train_type: str = None,
              num_epochs=100, use_wandb=True, save_best_model=True):

        model = model_service.model

        batch_size = batch_size
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                                  collate_fn=collate_molgraphs, num_workers=1, shuffle=True)
        test_loader = DataLoader(dataset=test_set, batch_size=batch_size,
                                 collate_fn=collate_molgraphs, num_workers=1)
        
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = MSELoss()
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=5, factor=0.5, min_lr=0.000002)
        early_stopper = EarlyStopper(patience=20, min_delta=0.001)

        run_name = ""
        # run_name = init_wandb(train_type)
        best_vloss = pow(10, 3)
        best_vmetrics = None
        best_vepoch = -1
        
        for epoch in tqdm(range(num_epochs)):

            running_loss = 0.0
            running_vloss = 0.0
            train_pred, train_true = torch.Tensor([]), torch.Tensor([])
            val_pred, val_true = torch.Tensor([]), torch.Tensor([])
            model_service.train_mode(train_type=train_type)
            for i, batch_data in enumerate(train_loader):
                _, bg, labels, masks = batch_data
                
                # if torch.cuda.is_available():
                #     bg = bg.cuda()
                #     labels = labels.cuda()

                optimizer.zero_grad()

                logP_pred = predict(model=model, 
                                    bg=bg)
                
                loss = criterion(logP_pred, labels)
                loss.backward()

                optimizer.step()
                

                train_pred = torch.cat((train_pred, logP_pred.cpu()), dim=0)
                train_true = torch.cat((train_true, labels.cpu()))

                running_loss += loss.item()
            avg_loss = running_loss / i

            # scheduler.step(avg_loss)

            true_train_values = train_true.view(-1).tolist()
            pred_train_values = train_pred.view(-1).tolist()
            train_metrics = calculate_metrics(true_train_values, pred_train_values)

            model_service.eval_mode()
            with torch.no_grad():
                for i, batch_data in enumerate(test_loader):
                    _, bg, labels, masks = batch_data
                    # if torch.cuda.is_available():
                    #     bg = bg.cuda()
                    #     labels = labels.cuda()

                    logP_pred = predict(model, bg)
                    vloss = criterion(logP_pred, labels)

                    val_pred = torch.cat((val_pred, logP_pred.cpu()), dim=0)
                    val_true = torch.cat((val_true, labels.cpu()))
                    
                    running_vloss += vloss.item()
            avg_vloss = running_vloss / (i + 1)

            true_val_values = val_true.view(-1).tolist()
            pred_val_values = val_pred.view(-1).tolist()
            val_metrics = calculate_metrics(true_val_values, pred_val_values)

            lr = optimizer.param_groups[0]['lr']
            if avg_vloss < best_vloss and save_best_model is True:
                # torch.save(model.state_dict(), rf'ml_part\weights\logP_dgllife_lipophilicity\logP_best_loss_{run_name.name}.pth')
                best_vloss = avg_vloss
                best_vmetrics = val_metrics
                best_vepoch = epoch

            # if use_wandb is True:
            #     wandb.log({"loss/train": avg_loss, 
            #                 "loss/val": avg_vloss, 
            #                 "lr": lr,
            #                 "mse/train": train_metrics['mse'],
            #                 "mae/train": train_metrics['mae'],
            #                 "r^2/train": train_metrics['r^2'],
            #                 "mse/val": val_metrics['mse'],
            #                 "mae/val": val_metrics['mae'],
            #                 "r^2/val": val_metrics['r^2']})
                
            if early_stopper.early_stop(avg_vloss):
                break

            scheduler.step(avg_vloss)

            # print("TRAIN")
            # print(train_metrics)
            # print("VAL")
            # print(val_metrics)
            # print('LOSS train: {} valid: {}, lr: {}'.format(avg_loss, avg_vloss, lr))

        # wandb.finish()
        return best_vloss, best_vmetrics, best_vepoch, run_name

import pandas as pd
import wandb
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import LipophilicityModelService
from utils import load_model, load_dataset, collate_molgraphs, calculate_metrics, predict

def init_wandb():
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="enamine-logP",
        
        # track hyperparameters and run metadata
        config={
            "learning_rate": 0.011064509215476023,
            "batch_size": 16,
            "architecture": "AttentiveFP_canonical_Lipophilicity",
            "dataset": "data\logP_lipophilicity_data",
            "epochs": 200,
            "optimizer": "Adam(lr=0.011064509215476023, weight_decay=0.0006897008354482659)",
            "loss": "MSELoss",
            "scheduler": "None",
            "info": "gnn.train(False), readout.train(False), shuffle=True"
        }
    )

    return run

def train(model_service, train_set, test_set, num_epochs=200, use_wandb=True, save_best_model=True):

    lr = 0.011064509215476023
    weight_decay = 0.0006897008354482659
    batch_size = 32

    model = model_service.model

    # batch_size = 32
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                             collate_fn=collate_molgraphs, num_workers=1, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size,
                             collate_fn=collate_molgraphs, num_workers=1)
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = MSELoss()
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=5, factor=0.5, min_lr=0.000002)

    run_name = init_wandb()
    best_vloss = pow(10, 3)
    model_service.train_mode()
    for epoch in tqdm(range(num_epochs)):

        running_loss = 0.0
        running_vloss = 0.0
        train_pred, train_true = torch.Tensor([]), torch.Tensor([])
        val_pred, val_true = torch.Tensor([]), torch.Tensor([])
        for i, batch_data in enumerate(train_loader):
            _, bg, labels, masks = batch_data
            
            # if torch.cuda.is_available():
            #     bg = bg.to("cuda")
            #     labels = labels.to("cuda")
            #     masks = masks.to("cuda")

            optimizer.zero_grad()

            logP_pred = predict(model=model, 
                                bg=bg)
            
            loss = criterion(logP_pred, labels)
            loss.backward()

            optimizer.step()
            

            train_pred = torch.cat((train_pred, logP_pred), dim=0)
            train_true = torch.cat((train_true, labels))

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
        if avg_vloss < best_vloss and save_best_model is True:
            torch.save(model.state_dict(), rf'ml_part\weights\logP_dgllife_lipophilicity\logP_best_loss_{run_name.name}.pth')
            best_vloss = avg_vloss

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

        print('LOSS train: {} valid: {}, lr: {}'.format(avg_loss, avg_vloss, lr))

if __name__ == "__main__":

    model_service = LipophilicityModelService()

    train_csv = r'data\logP_lipophilicity_data\gnn_cv\train.csv'
    test_csv = r'data\logP_lipophilicity_data\gnn_cv\test.csv'
    
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    train_set = load_dataset(train_df)
    test_set = load_dataset(test_df)

    train(model_service=model_service,
          train_set=train_set,
          test_set=test_set)

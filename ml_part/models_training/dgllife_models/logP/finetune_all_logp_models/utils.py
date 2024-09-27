import os
from math import sqrt
from functools import partial

import wandb
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import dgl
from dgllife.model import load_pretrained
from dgllife.data import MoleculeCSVDataset
from dgllife.utils import smiles_to_bigraph
from dgllife.utils import AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from dgllife.model.model_zoo.gat_predictor import GATPredictor

def init_wandb(train_mode: str = None):
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="enamine-logP",
        
        # track hyperparameters and run metadata
        config={
            "learning_rate": 10 ** (-2.5),
            "train_mode": train_mode,
            "batch_size": 32,
            "architecture": "AttentiveFP_canonical_Lipophilicity",
            "dataset": "data\logP_lipophilicity_data",
            "epochs": 100,
            "is AttentiveFPGNN freezed": True,
            "is AttentiveFPReadout freezed": True,
            "optimizer": "Adam(lr=10 ** (-2.5), weight_decay=10 ** (-5.0))",
            "loss": "MSELoss",
            "scheduler": "None",
            "info": "gnn.train(False), readout.train(False), shuffle=True"
        }
    )

    return run

def calculate_metrics(true_values, pred_values):
    mse = round(mean_squared_error(true_values, pred_values),3)
    mae = round(mean_absolute_error(true_values, pred_values),3)
    r_score = round(r2_score(true_values, pred_values),3)

    return {"mse": mse,
            "mae": mae,
            "r^2": r_score,}

def load_model(model_name: str = None):
    if model_name is None:
        model_name = model_name

    model = load_pretrained("AttentiveFP_canonical_Lipophilicity")

    model = model.to('cpu')
    # if torch.cuda.is_available():
    #     model = model.to('cuda')

    return model

def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader.

    Parameters
    ----------
    data : list of 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and a binary
        mask indicating the existence of labels.

    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels.
    """
    smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)

    return smiles, bg, labels, masks

def load_dataset(df):
    dataset = MoleculeCSVDataset(df=df,
                                 smiles_to_graph=partial(smiles_to_bigraph, num_virtual_nodes=0, add_self_loop=True),
                                 node_featurizer=CanonicalAtomFeaturizer(),
                                 edge_featurizer=CanonicalBondFeaturizer(self_loop=True),
                                 smiles_column='Smiles',
                                 cache_file_path=os.path.join(r'logp_attentive_lipophilicity_model', 'graph.bin'),
                                 task_names=['logP'],
                                 n_jobs=1,
                                 load=False
                                )

    return dataset

def predict(model, bg, device='cuda'):
   
    # bg = bg.to(device)

    node_feats = bg.ndata['h']
    edge_feats = bg.edata['e']

    if type(model).__name__ == "GATPredictor":
        return model(bg, node_feats)

    return model(bg, node_feats, edge_feats)

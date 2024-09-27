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
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from dgllife.utils import AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer
from dgllife.utils import PretrainAtomFeaturizer, PretrainBondFeaturizer
from dgllife.model.model_zoo.gat_predictor import GATPredictor

def calculate_metrics(true_values, pred_values):
    mse = round(mean_squared_error(true_values, pred_values),3)
    mae = round(mean_absolute_error(true_values, pred_values),3)
    r_score = round(r2_score(true_values, pred_values),3)

    return {"mse": mse,
            "mae": mae,
            "r^2": r_score,}

def load_model(model_name: str = None):
    if model_name is None:
        model_name = "AttentiveFP_canonical_Lipophilicity"

    model = load_pretrained(model_name)

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

def load_dataset(df, features_type: bool = False):
    atom_features = CanonicalAtomFeaturizer()
    bond_features = CanonicalBondFeaturizer(self_loop=True)
    if "attentive" in features_type.lower():
        atom_features = AttentiveFPAtomFeaturizer()
        bond_features = AttentiveFPBondFeaturizer(self_loop=True)
    elif "gin" in features_type.lower():
        atom_features = PretrainAtomFeaturizer()
        bond_features = PretrainBondFeaturizer(self_loop=True)
    dataset = MoleculeCSVDataset(df=df,
                                 smiles_to_graph=partial(smiles_to_bigraph, num_virtual_nodes=0, add_self_loop=True),
                                 node_featurizer=atom_features,
                                 edge_featurizer=bond_features,
                                 smiles_column='Smiles',
                                 cache_file_path=os.path.join(r'logp_lipophilicity_model', 'graph.bin'),
                                 task_names=['logP'],
                                 n_jobs=1,
                                 load=False
                                )

    return dataset

def predict(model, bg, device='cuda'):
 
    if "gin" in type(model).__name__.lower():
        node_feats = [
            bg.ndata.pop('atomic_number'),
            bg.ndata.pop('chirality_type')
        ]
        edge_feats = [
            bg.edata.pop('bond_type'),
            bg.edata.pop('bond_direction_type')
        ]
        node_feats = [n.to(device) for n in node_feats]
        edge_feats = [e.to(device) for e in edge_feats]

    else:
        node_feats = bg.ndata['h']
        edge_feats = bg.edata['e']
        
        if "GAT" in type(model).__name__ or "GCN" in type(model).__name__:
            return model(bg, node_feats)

    return model(bg, node_feats, edge_feats)

def init_wandb(train_mode: str = None, batch_size: int = 32,
               lr: float = 10 ** (-2.5),
               weight_decay: float = 10 ** (-5.0),
               model_name: str = "",
               epochs: int = 200):
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

    return run

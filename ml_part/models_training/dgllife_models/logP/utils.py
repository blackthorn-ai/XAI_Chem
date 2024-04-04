import os
from math import sqrt
from functools import partial

import torch
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import dgl
from dgllife.model import load_pretrained
from dgllife.data import MoleculeCSVDataset
from dgllife.utils import smiles_to_bigraph
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer

def calculate_metrics(true_values, pred_values):
    mse = round(mean_squared_error(true_values, pred_values),3)
    mae = round(mean_absolute_error(true_values, pred_values),3)
    r_score = round(r2_score(true_values, pred_values),3)

    return {"mse": mse,
            "mae": mae,
            "r^2": r_score,}

def load_model(model_name: str = None):
    if model_name is None:
        model_name = 'AttentiveFP_canonical_Lipophilicity'

    model = load_pretrained(model_name)

    model = model.to('cpu')

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

def predict(model, bg, device='cpu'):
    bg = bg.to(device)

    node_feats = bg.ndata['h'].to(device)
    edge_feats = bg.edata['e'].to(device)

    return model(bg, node_feats, edge_feats)

import os
from functools import partial

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dgl
from dgllife.utils import smiles_to_bigraph, smiles_to_complete_graph
from dgllife.data import TencentAlchemyDataset, MoleculeCSVDataset
from dgllife.model import load_pretrained
from dgllife.data.alchemy import alchemy_nodes, alchemy_edges

device = 'cpu'

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
    smiles, graphs, labels = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    return smiles, bg, labels

def regress(model, bg):
    bg = bg.to(device)
    
    h = bg.ndata.pop('n_feat')
    e = bg.edata.pop('e_feat')
    h, e = h.to(device), e.to(device)
    return model(bg, h, e)

def run_an_eval_epoch(model, data_loader):
    model.eval()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels = batch_data
            labels = labels.to(device)
            prediction = regress(model, bg)
            
            print(prediction)

    return prediction

def main():

    csv_logP_test_path = r'data\logP_lipophilicity_data\gnn_cv\test.csv'
    df_test = pd.read_csv(csv_logP_test_path, index_col=0)

    lipophilicity_model = load_pretrained("MPNN_canonical_Lipophilicity")

    val_set = TencentAlchemyDataset(mode='valid')
    # val_set = MoleculeCSVDataset(df=df_test,
    #                              smiles_to_graph=smiles_to_bigraph,
    #                              node_featurizer=alchemy_nodes,
    #                              edge_featurizer=alchemy_edges,
    #                              smiles_column='Smiles',
    #                              cache_file_path=os.path.join(r'logp_attentive_lipophilicity_model', 'graph.bin'),
    #                              task_names=['logP'],
    #                              n_jobs=1,
    #                              load=False
    #                             )

    eval_loader = DataLoader(dataset=val_set,
                            batch_size=32,
                            collate_fn=collate_molgraphs)
    
    val_score = run_an_eval_epoch(lipophilicity_model, eval_loader)

if __name__ == "__main__":
    main()
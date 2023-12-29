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
from model_load import load_logP_model, load_pKa_acidic_model, load_pKa_basic_model

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt

from rdkit import Chem
from rdkit.Chem import rdmolops
import dgl
from dgllife.utils import smiles_to_bigraph
import networkx as nx
import matplotlib.pyplot as plt

from tqdm import tqdm


def get_atoms_name_by_idx(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)

    atoms_idxes = {}

    for atom in mol.GetAtoms():
        atom_name = atom.GetSymbol()
        atoms_idxes[atom.GetIdx()] = atom_name

    return atoms_idxes



def relevance_propagation(model, smiles, args, exp_config):
    bg = convert_smiles_to_dglgraph(smiles, args)
    
    node_features = bg.ndata['h']
    edge_features = bg.edata['e']

    model_prediction, _ = model(bg, node_features, edge_features)
    
    node_relevances, edge_relevances = model.lrp(bg, node_features, edge_features)

    predicted_value = model_prediction.item()
    print(f"SMILES: {smiles}, pred: {predicted_value}")

    atoms_name_by_idx = get_atoms_name_by_idx(smiles)
    display_graph(bg, node_relevances, edge_relevances, atoms_name_by_idx)

    return node_relevances, edge_relevances


def display_graph(bg, node_relevances, edge_relevances, atoms_name_by_idx):
    node_relevances_array = [node_relevance for idx, node_relevance in node_relevances.items()]
    edge_relevances_array = [edge_relevance for idx, edge_relevance in edge_relevances.items()]
    atoms_name_array = [atom_name_by_idx for idx, atom_name_by_idx in atoms_name_by_idx.items()]
    print(atoms_name_array)
    nx_graph = bg.to_networkx()
    
    pos = nx.spring_layout(nx_graph) 
    cmap = plt.cm.Reds
    arc_rad = 0.1
    options = {
        "font_weight": "bold",
        "node_color": node_relevances_array,
        "edge_color": edge_relevances_array,
        "cmap": cmap,
        "edge_cmap": cmap,
        "width": 2,
        "with_labels": True,
        "labels": atoms_name_by_idx,
        "connectionstyle": f'arc3, rad = {arc_rad}'
    }
    nx.draw(nx_graph, **options)

    cbar_node = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=plt.gca())
    cbar_node.set_label('Node Relevance')

    cbar_edge = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=plt.gca())
    cbar_edge.set_label('Edge Relevance')

    plt.show()


def convert_smiles_to_dglgraph(smiles: str, args):
    dglgraph = smiles_to_bigraph(smiles=smiles,
                                 node_featurizer=args['node_featurizer'],
                                 edge_featurizer=args['edge_featurizer'])

    return dglgraph
    

SMILES = "NCCC(F)(F)F​"
   
if __name__ == '__main__':
    with open(r'C:\work\DrugDiscovery\RT_LogP_with_pKa_model\RTlogD\final_model/RTlogD/args.pickle', 'rb') as file:
        args =pickle.load(file)
        args['task'] = ['pKa']
    with open(r'C:\work\DrugDiscovery\RT_LogP_with_pKa_model\RTlogD\final_model/RTlogD/configure.json', 'r') as f:
        exp_config = json.load(f)
    args['device'] = torch.device('cpu')
    args = setup(args)

    amine_model = load_pKa_basic_model(args)
    acid_model = load_pKa_acidic_model(args)

    node_relevances, edge_relevances = relevance_propagation(acid_model, SMILES, args, exp_config)
    print(node_relevances)
    print(edge_relevances)

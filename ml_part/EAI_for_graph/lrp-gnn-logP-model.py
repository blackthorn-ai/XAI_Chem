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

from utils import setup, load_model
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


def get_atoms_name_by_idx(node_features) -> dict:
    allowable_set = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                     'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn',
                     'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au',
                     'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb']
    
    atoms_idxes = {}
    for node_index in range(len(node_features)):
        atom_features = node_features[node_index].tolist()
        for atom_feature_index in range(len(allowable_set)):
            if atom_features[atom_feature_index] == 1:
                atoms_idxes[node_index] = allowable_set[atom_feature_index]
                break

    print(atoms_idxes)
    return atoms_idxes


def dglgraph_info(bg):
    print("EDGEs:")
    for i in range(8):
        print(i, bg.find_edges(i))
    return bg


def prepare_node_feats(node_feats, pka1_atom_list, pka2_atom_list):
    pka1_atom_list=np.array(pka1_atom_list)
    pka1_atom_list[np.isinf(pka1_atom_list)]=15
    pka2_atom_list=np.array(pka2_atom_list)
    pka2_atom_list[np.isinf(pka2_atom_list)]=0

    pka1_feature = torch.Tensor(pka1_atom_list/11).to(args['device'])
    pka2_feature = torch.Tensor(pka2_atom_list/11).to(args['device'])

    pka1_feature=pka1_feature.unsqueeze(-1)
    pka2_feature=pka2_feature.unsqueeze(-1)

    node_feats = torch.cat([node_feats,pka1_feature,pka2_feature],dim = 1)
    return node_feats


def predict_logP(bg, node_features, edge_features, logP_model, pKa_acid_model, pKa_amine_model):
    node_features_for_acid = node_features.clone()
    node_features_for_amine = node_features.clone()
    
    bg_for_acid = bg.clone()
    bg_for_amine = bg.clone()

    pKa_acid_model.eval()
    pKa_amine_model.eval()

    pKa_acid, pKa_acid_atom_list = pKa_acid_model(bg_for_acid, node_features_for_acid, edge_features)
    pKa_amine, pKa_amine_atom_list = pKa_amine_model(bg_for_amine, node_features_for_amine, edge_features)

    node_feats = prepare_node_feats(node_features, pKa_acid_atom_list, pKa_amine_atom_list)
    edge_feats = edge_features.clone()

    logP_model.eval()
    logP_prediction = logP_model(bg, node_feats, edge_feats)
    predicted_value = logP_prediction[0][-1].item()
    print(f"Pred value: {predicted_value}")

    return bg, node_feats, edge_feats



def relevance_propagation(logP_model, acid_model, amine_model, smiles, args, exp_config):
    bg = convert_smiles_to_dglgraph(smiles, args)
    dglgraph_info(bg)
    
    print(bg.ndata.keys(), bg.edata.keys())
    node_features = bg.ndata['h']
    edge_features = bg.edata['e']
    # for node_feature in node_features:
    #     print(node_feature)

    for edge_feature in edge_features:
        print(edge_feature)

    bg, node_feats, edge_feats = predict_logP(bg, node_features, edge_features, logP_model, acid_model, amine_model)

    logP_model.eval()
    node_relevances, edge_relevances = logP_model.lrp(bg, node_feats, edge_feats)

    # model_prediction, _ = model(bg, node_features, edge_features)
    
    # node_relevances, edge_relevances = model.lrp(bg, node_features, edge_features)

    # print(f"SMILES: {smiles}, pred: {model_prediction}")
    # predicted_value = model_prediction.item()

    # old networkx graph(problems with edges)
    # atoms_name_by_idx = get_atoms_name_by_idx(node_features)
    # display_graph(bg, node_relevances, edge_relevances, atoms_name_by_idx)

    # new networkx graph
    convert_bigraph_to_networkx_graph(bg, node_relevances, edge_relevances)

    return node_relevances, edge_relevances


def convert_bigraph_to_networkx_graph(bg, node_relevances, edge_relevances):
    node_relevances_array = [node_relevance for idx, node_relevance in node_relevances.items()]
    edge_relevances_array = [edge_relevance for idx, edge_relevance in edge_relevances.items()]

    node_features = bg.ndata['h'].clone()
    atoms_name_by_idx = get_atoms_name_by_idx(node_features)

    nodes_names = [idx for idx, atom_name_by_idx in atoms_name_by_idx.items()]
    
    G = nx.DiGraph()
    for v, w in zip(nodes_names, node_relevances):
        G.add_node(v, weight=w)
    
    edge_relevances_array_temp = []
    for edge_index in range(len(edge_relevances)):
        from_node_tensor, dest_node_tensor = bg.find_edges(edge_index)
        from_node, dest_node = from_node_tensor.item(), dest_node_tensor.item()
        
        edge = (from_node, dest_node)
        edge_weight = edge_relevances[edge_index]
        
        # if from_node < dest_node:
        edge_weight = round(edge_weight, 3)
        G.add_edge(*edge, weight=edge_weight)
        print(edge, edge_relevances[edge_index])
        edge_relevances_array_temp.append(edge_weight)

    print(edge_relevances_array_temp)
    pos = nx.spring_layout(G) 
    labels = nx.get_edge_attributes(G, 'weight')
    cmap = plt.cm.coolwarm
    arc_rad = 0.05

    edge_weights = [data['weight'] for _, _, data in G.edges(data=True)]
    min_weight = min(edge_weights)
    max_weight = max(edge_weights)
    normalized_weights = [(weight - min_weight) / (max_weight - min_weight) for weight in edge_weights]
    edge_colors = plt.cm.coolwarm(normalized_weights)

    options = {
        # "pos": pos,
        "font_weight": "bold",
        "node_color": node_relevances_array,
        "edge_color": edge_colors,
        "cmap": cmap,
        "edge_cmap": cmap,
        "width": 2,
        "with_labels": True,
        "labels": atoms_name_by_idx,
        "connectionstyle": f'arc3, rad = {arc_rad}',
    }
    nx.draw(G, **options)
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    cbar_node = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=plt.gca())
    cbar_node.set_label('Node Relevance')

    cbar_edge = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=plt.gca())
    cbar_edge.set_label('Edge Relevance')

    plt.show()
    


def display_graph(bg, node_relevances, edge_relevances, atoms_name_by_idx):
    node_relevances_array = [node_relevance for idx, node_relevance in node_relevances.items()]
    edge_relevances_array = [edge_relevance for idx, edge_relevance in edge_relevances.items()]
    atoms_name_array = [atom_name_by_idx for idx, atom_name_by_idx in atoms_name_by_idx.items()]
    print(atoms_name_array)
    nx_graph = bg.to_networkx()

    print("node relevance:", node_relevances_array)
    print("edge relevance:", edge_relevances_array)
    
    pos = nx.spring_layout(nx_graph) 
    cmap = plt.cm.coolwarm
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
    

SMILES = "FC([H])([C@H]1C[C@H]1C(O)=O)[H]​​"
   
if __name__ == '__main__':
    with open(r'C:\work\DrugDiscovery\RT_LogP_with_pKa_model\RTlogD\final_model/RTlogD/args.pickle', 'rb') as file:
        args =pickle.load(file)
        args['task'] = ['pKa']
    with open(r'C:\work\DrugDiscovery\RT_LogP_with_pKa_model\RTlogD\final_model/RTlogD/configure.json', 'r') as f:
        exp_config = json.load(f)
    args['device'] = torch.device('cpu')
    args = setup(args)

    logP_model = load_logP_model(exp_config, args)
    amine_model = load_pKa_basic_model(args)
    acid_model = load_pKa_acidic_model(args)

    node_relevances, edge_relevances = relevance_propagation(logP_model, acid_model, amine_model, SMILES, args, exp_config)
    print(node_relevances)
    print(edge_relevances)

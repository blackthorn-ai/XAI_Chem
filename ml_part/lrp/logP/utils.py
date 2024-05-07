from collections import deque

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import torch

from rdkit import Chem
from rdkit.Chem import rdchem

def prepare_data():
    df_main = pd.read_csv(r'C:\work\DrugDiscovery\main_git\XAI_Chem\data\init_data\pKa_Prediction_Starting data_2024.01.25.csv', index_col=0)

    SMILES_to_fgroup = {}
    SMILES_to_identificator = {}
    SMILES_to_cycle_type = {}
    for index, row in df_main.iterrows():
        SMILES = row['Amides for LogP']
        if pd.isnull(SMILES):
            continue
        
        SMILES_to_fgroup[SMILES] = row['F group']
        SMILES_to_identificator[SMILES] = row['identificator']
        SMILES_to_cycle_type[SMILES] = row['Framework']
    
    return SMILES_to_fgroup, SMILES_to_identificator, SMILES_to_cycle_type

def get_color(value, cmap_type='coolwarm', vmin=-1, vmax=1):
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_type)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    rgba_color = sm.to_rgba(value)
    return rgba_color

def subgroup_relevance(atoms_subgroup, node_relevances):
        node_relevance = 0
        for atom in atoms_subgroup:
            # node_relevance += abs(node_relevances[atom])
            node_relevance += node_relevances[atom]

        return node_relevance / len(atoms_subgroup)

def check_if_edge_in_one_group(groups, start_atom_idx, 
                               end_atom_idx):
    is_in_one_group = False
    group_with_edge_index = None
    
    for group_index in range(len(groups)):
         if start_atom_idx in groups[group_index] and end_atom_idx in groups[group_index]:
              is_in_one_group = True
              group_with_edge_index = group_index

    return is_in_one_group, group_with_edge_index

def normalize_to_minus_one_to_one(data: dict):
    min_val = min(data.values())
    max_val = max(data.values())
    difference = max_val - min_val

    for key, value in data.items():
        data[key] = ((value - min_val) / difference) * 2 - 1
    
    return data 

def find_the_furthest_atom(mol: rdchem.Mol, 
                           atom_id: int, 
                           atoms_not_to_visit: list = []):
    queue = deque([(atom_id, 0)])

    visited = set()
    
    while queue:
        current_atom, distance = queue.popleft()
        
        visited.add(current_atom)
        
        neighbors = []
        for atom in mol.GetAtomWithIdx(current_atom).GetNeighbors():
            if atom.GetSymbol().lower() == 'h':
                continue
            if atom.GetIdx() in atoms_not_to_visit:
                continue
            if atom.IsInRing():
                continue
            
            neighbors.append(atom.GetIdx())
        
        for neighbor in neighbors:
            if neighbor not in visited:
                queue.append((neighbor, distance + 1))
    
    return current_atom, distance

import pandas as pd
import numpy as np
import re
import sys
import os
from itertools import chain
sys.path.append("..") 
sys.path.insert(0, os.path.dirname(__file__))

from rdkit import Chem
from rdkit.Chem import rdmolops, rdmolfiles, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import BRICS
import dgl
from dgllife.utils import smiles_to_bigraph
import networkx as nx

from utils_lrp import check_if_node_in_subgroup, check_if_edge_in_subgroup, get_color, prepare_node_feats
from constants import ModelType

class Lrp:
    def __init__(self, logP_model, amine_pKa_model, acid_pKa_model, smiles, exp_config, args, model_type=ModelType):
        super(Lrp, self).__init__()

        self.args = args
        self.logP_model = logP_model
        self.amine_pKa_model = amine_pKa_model
        self.acid_pKa_model = acid_pKa_model
        self.smiles = smiles
        # self.subgroup_smiles = self.convert_functional_group_to_smile(functional_group)

        self.bg = self.convert_smiles_to_dglgraph(smiles, args)

        self.pKa_node_features, self.pKa_edge_features = self.bg.ndata['h'], self.bg.edata['e']
        self.logP_node_features, self.logP_edge_features = self.prepare_features_for_logP_model()

        if model_type == ModelType.logP:
            self.node_relevances, self.edge_relevances = self.calculate_mol_relevances_for_logP()
        elif model_type == ModelType.pKa_acid:
            self.node_relevances, self.edge_relevances = self.calculate_mol_relevances_for_pKa(is_acid=True)
        elif model_type == ModelType.pKa_amine:
            self.node_relevances, self.edge_relevances = self.calculate_mol_relevances_for_pKa(is_amine=True)
        self.node_relevances, self.edge_relevances = self.normilize_relevances(self.node_relevances, self.edge_relevances)

    @staticmethod
    def convert_functional_group_to_smile(functional_group):
        functional_group_to_smiles = {
            "CF3": "C(F)(F)F", 
            "CH2F": "CF", 
            "gem-CF2": "C(F)(F)", 
            "CHF2": "C(F)(F)",
            "CHF": "CF",
            "non-F": ""
        }
        return functional_group_to_smiles[functional_group]

    @staticmethod
    def convert_smiles_to_dglgraph(smiles: str, args):
        dglgraph = smiles_to_bigraph(smiles=smiles,
                                     node_featurizer=args['node_featurizer'],
                                     edge_featurizer=args['edge_featurizer'])

        return dglgraph
    
    def prepare_features_for_logP_model(self):
        node_features = self.bg.ndata['h']
        node_features_for_acid = node_features.clone()
        node_features_for_amine = node_features.clone()

        edge_features = self.bg.edata['e'].clone()
        
        bg_for_acid = self.bg.clone()
        bg_for_amine = self.bg.clone()

        self.acid_pKa_model.eval()
        self.amine_pKa_model.eval()

        pKa_acid, pKa_acid_atom_list = self.acid_pKa_model(bg_for_acid, node_features_for_acid, edge_features)
        pKa_amine, pKa_amine_atom_list = self.amine_pKa_model(bg_for_amine, node_features_for_amine, edge_features)

        node_feats = prepare_node_feats(node_features, pKa_acid_atom_list, pKa_amine_atom_list, self.args)
        edge_feats = edge_features.clone()

        return node_feats, edge_feats

    def calculate_mol_relevances_for_pKa(self, is_amine=False, is_acid=False):
        if is_amine is True:
            self.amine_pKa_model.eval()
            node_relevances, edge_relevances = self.amine_pKa_model.lrp(self.bg, self.pKa_node_features, self.pKa_edge_features)
        else:
            self.acid_pKa_model.eval()
            node_relevances, edge_relevances = self.acid_pKa_model.lrp(self.bg, self.pKa_node_features, self.pKa_edge_features)
        
        return (node_relevances, edge_relevances)
    
    def calculate_mol_relevances_for_logP(self):
        self.logP_model.eval()
        node_relevances, edge_relevances = self.logP_model.lrp(self.bg, self.logP_node_features, self.logP_edge_features)

        return (node_relevances, edge_relevances)
    
    @staticmethod
    def normilize_relevances(node_relevance_dict, edge_relevance_dict):

        max_relevance = max(max(node_relevance_dict.values()), max(edge_relevance_dict.values()))
        min_relevance = min(min(node_relevance_dict.values()), min(edge_relevance_dict.values()))

        for atom_id, atom_relevance in node_relevance_dict.items():
            node_relevance_dict[atom_id] = (((atom_relevance - min_relevance) / (max_relevance - min_relevance)) - 0.5) * 2

        for atom_id, atom_relevance in edge_relevance_dict.items():
            edge_relevance_dict[atom_id] = (((atom_relevance - min_relevance) / (max_relevance - min_relevance)) - 0.5) * 2

        return node_relevance_dict, edge_relevance_dict

    def create_mol_as_bigraph(self):
        mol = Chem.MolFromSmiles(self.smiles)
        mol_subgroups = []

        fragments = BRICS.BRICSDecompose(mol)

        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)

        for fragment in fragments:
            regex_pattern = r'\[[^\]]+\]'
            smiles_without_brackets = re.sub(regex_pattern, '', fragment)

            regex_pattern = r'\(\)'

            string_without_empty_parentheses = re.sub(regex_pattern, '', smiles_without_brackets)
            mol_subgroups.append(Chem.MolFromSmiles(string_without_empty_parentheses))

        return mol, mol_subgroups
    
    def subgroup_relevance(self, atoms_subgroup):
        node_relevance = None
        for atom in atoms_subgroup:
            if node_relevance is None:
                node_relevance = self.node_relevances[atom]
            else:
                node_relevance = (node_relevance + self.node_relevances[atom]) / 2.

        edge_relevance = None
        for edge_index in range(len(self.edge_relevances)):
            from_node_tensor, dest_node_tensor = self.bg.find_edges(edge_index)
            from_node, dest_node = from_node_tensor.item(), dest_node_tensor.item()
            if from_node in atoms_subgroup and dest_node in atoms_subgroup:
                if edge_relevance is None: 
                    edge_relevance = self.edge_relevances[edge_index]
                else: 
                    edge_relevance = (edge_relevance + self.edge_relevances[atom]) / 2.

        # TODO: proper relevance calculation
        return node_relevance

    def convert_bigraph_edges_relevance_to_mol_bonds(self, mol):
        edge_relevances_for_mol = {}
        edge_relevances_for_mol_all_bonds = {}
        for mol_edge_idx in range(mol.GetNumBonds()):
            mol_begin_node = mol.GetBonds()[mol_edge_idx].GetBeginAtomIdx()
            mol_end_node = mol.GetBonds()[mol_edge_idx].GetEndAtomIdx()

            for bg_edge_idx, bg_edge_relevance in self.edge_relevances.items():
                from_node_tensor, dest_node_tensor = self.bg.find_edges(bg_edge_idx)
                from_node, dest_node = from_node_tensor.item(), dest_node_tensor.item()

                if (from_node == mol_begin_node and dest_node == mol_end_node) or \
                (from_node == mol_end_node and dest_node == mol_begin_node):
                    if mol_edge_idx not in edge_relevances_for_mol:
                        edge_relevances_for_mol[mol_edge_idx] = bg_edge_relevance
                        edge_relevances_for_mol_all_bonds[mol_edge_idx] = {bg_edge_idx: edge_relevances_for_mol[mol_edge_idx]}
                    else:
                        if abs(bg_edge_relevance) > abs(edge_relevances_for_mol[mol_edge_idx]):
                            edge_relevances_for_mol[mol_edge_idx] = bg_edge_relevance
                        edge_relevances_for_mol_all_bonds[mol_edge_idx][bg_edge_idx] = edge_relevances_for_mol[mol_edge_idx]

        return edge_relevances_for_mol, edge_relevances_for_mol_all_bonds

    def extract_atoms_colors_from_relevances(self, mol, atoms_subgroups):
        atoms_colors = {}
        atoms = []
        max_node_relevance = max(self.node_relevances.values())
        min_node_relevance = min(self.node_relevances.values())
        for atom_index, node_relevance in self.node_relevances.items():
            if not check_if_node_in_subgroup(atoms_subgroups, atom_index):
                
                mol.GetAtoms()[atom_index].SetProp("atomNote", str(round(self.node_relevances[atom_index], 2)))
                
                color_by_relevance = get_color(node_relevance, vmin=min_node_relevance, vmax=max_node_relevance)
                atoms_colors[atom_index] = color_by_relevance
                atoms.append(atom_index)

        for atoms_subgroup in atoms_subgroups:
            calculated_subgroup_relevance = self.subgroup_relevance(atoms_subgroup)
            
            mol.GetAtoms()[atoms_subgroup[0]].SetProp("atomNote", str(round(calculated_subgroup_relevance, 2)))

            color_by_relevance = get_color(calculated_subgroup_relevance, vmin=min_node_relevance, vmax=max_node_relevance)
            highlight_colors = {atom_idx: color_by_relevance for atom_idx in atoms_subgroup}
            atoms_colors.update(highlight_colors)
            atoms = atoms + list(atoms_subgroup)

        return atoms, atoms_colors
    
    def edge_description(self, mol, edge_dict):
        edge_description = None
        for bg_edge_idx, bg_edge_relevance in edge_dict.items():
            from_node_tensor, dest_node_tensor = self.bg.find_edges(bg_edge_idx)
            from_node, dest_node = from_node_tensor.item(), dest_node_tensor.item()

            from_node_name = mol.GetAtoms()[from_node].GetSymbol()
            dest_node_name = mol.GetAtoms()[dest_node].GetSymbol()

            rounded_relevance = round(bg_edge_relevance, 3) if bg_edge_relevance < 0 else round(bg_edge_relevance, 2)
            if edge_description is None:
                edge_description = f"{from_node_name}{dest_node_name}={rounded_relevance}"
            else:
                edge_description = f"{edge_description}|{from_node_name}{dest_node_name}={rounded_relevance}"
        return edge_description

    def extract_bonds_colors_from_relevances(self, mol, atoms_subgroups):
        bonds_colors = {}
        bonds = []
        # draw edge relevance
        mol_edge_relevances, mol_edge_relevances_all_bonds = self.convert_bigraph_edges_relevance_to_mol_bonds(mol)
        max_edge_relevance = max(mol_edge_relevances.values())
        min_edge_relevance = min(mol_edge_relevances.values())
        for edge_index, edge_relevance in mol_edge_relevances.items():
            from_node, dest_node = mol.GetBonds()[edge_index].GetBeginAtomIdx(), mol.GetBonds()[edge_index].GetEndAtomIdx()

            if not check_if_edge_in_subgroup(atoms_subgroups, from_node, dest_node):
                edge_description = self.edge_description(mol, mol_edge_relevances_all_bonds[edge_index])
                mol.GetBondWithIdx(edge_index).SetProp("bondNote", edge_description)

                color_by_relevance = get_color(edge_relevance, vmin=min_edge_relevance, vmax=max_edge_relevance)
                bonds_colors[edge_index] = color_by_relevance
                bonds.append(edge_index)

        max_node_relevance = max(self.node_relevances.values())
        min_node_relevance = min(self.node_relevances.values())
        for atoms_subgroup in atoms_subgroups:
            calculated_subgroup_relevance = self.subgroup_relevance(atoms_subgroup)
            color_by_relevance = get_color(calculated_subgroup_relevance, vmin=min_node_relevance, vmax=max_node_relevance)

            # bonds color calculation
            for sub_edge_idx in range(mol.GetNumBonds()):
                begin_node = mol.GetBonds()[sub_edge_idx].GetBeginAtomIdx()
                end_node = mol.GetBonds()[sub_edge_idx].GetEndAtomIdx()

                if begin_node in atoms_subgroup and end_node in atoms_subgroup:
                    bonds.append(sub_edge_idx)
                    bonds_colors[sub_edge_idx] = color_by_relevance
        
        return bonds, bonds_colors


    def save_molecule_with_relevances(self, output_svg_path): 

        mol, mol_subgroups = self.create_mol_as_bigraph()

        drawer = rdMolDraw2D.MolDraw2DSVG(1200, 1200)
        atoms_subgroups = []
        index = 0
        for mol_subgroup in mol_subgroups:
            atoms_subgroup = mol.GetSubstructMatches(mol_subgroup)

            subgroup_check, amount_check = False, 0
            for atom_name in atoms_subgroup[0]:
                atoms_subgroups_temp = tuple(chain(*atoms_subgroups))
                if atom_name not in atoms_subgroups_temp:
                    subgroup_check = True
                    amount_check += 1
                    break
            
            if index == 0: atoms_subgroups = atoms_subgroup
            if subgroup_check is True:
                if index > 0: atoms_subgroups = atoms_subgroups + atoms_subgroup

            # atoms_subgroups = []
            index += 1

        atoms, atoms_colors = self.extract_atoms_colors_from_relevances(mol, atoms_subgroups)
        bonds, bonds_colors = self.extract_bonds_colors_from_relevances(mol, atoms_subgroups)

        drawer.DrawMolecule(mol, highlightAtoms=atoms, highlightAtomColors=atoms_colors, highlightBonds=bonds, highlightBondColors=bonds_colors)

        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        
        output_svg = output_svg_path
        with open(output_svg, "w") as svg_file:
            svg_file.write(svg) 
  
import io
from PIL import Image

from rdkit import Chem
from rdkit.Chem import rdmolops, rdmolfiles
from rdkit.Chem.Draw import rdMolDraw2D

import torch
from torch.autograd import Variable

from dgllife.utils.mol_to_graph import SMILESToBigraph
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer

from pKa_model_service import PkaModelService
from utils import get_color, subgroup_relevance, check_if_edge_in_one_group, normalize_to_minus_one_to_one
from constants import functional_group_to_smiles
from constants import Identificator, RelevanceMode

class PkaLRP:
    def __init__(self,
                 model_service: PkaModelService = None,
                 SMILES: str = None,
                 identificator: Identificator = None,
                 fluorine_group: str = None,
                 is_centrize_relevances: bool = True
                 ) -> None:
        self.device = "cpu"
        self.SMILES = SMILES
        self.identificator = identificator.name
        self.fluorine_group = fluorine_group

        if model_service is None:
            
            model_service = PkaModelService(
                identificator=identificator,
            )
            
        self.model = model_service.pKa_model

        self.bg = self.prepare_pKa_graph(SMILES=SMILES)
        self.node_features = self.bg.ndata['h']
        self.edge_features = self.bg.edata['e']

        self.node_relevances, self.edge_relevances = self.lrp(self.bg, self.node_features, self.edge_features)
        if is_centrize_relevances:
            min_relevance = min(list(self.node_relevances.values()) + list(self.edge_relevances.values()))
            max_relevance = max(list(self.node_relevances.values()) + list(self.edge_relevances.values()))
            self.node_relevances = normalize_to_minus_one_to_one(data=self.node_relevances, min_val=min_relevance, max_val=max_relevance)
            self.edge_relevances = normalize_to_minus_one_to_one(data=self.edge_relevances, min_val=min_relevance, max_val=max_relevance)

        self.mol = PkaLRP.create_mol_as_bigraph(self.SMILES)
        self.atoms_groups, self.derivatives_atoms = self.obtain_subgroups_from_mol(molecule=self.mol)
        
        self.relevance_entire_derivatives = PkaLRP.calculate_derivative_relevance_in_molecule(
            mol=self.mol, node_relevances=self.node_relevances,
            derivatives_atoms=self.derivatives_atoms, 
            relevance_mode=RelevanceMode.entire_derivatives_fluor_only_nodes
        )

        self.relevance_only_fluorine = PkaLRP.calculate_derivative_relevance_in_molecule(
            mol=self.mol, node_relevances=self.node_relevances,
            derivatives_atoms=self.derivatives_atoms, 
            relevance_mode=RelevanceMode.entire_derivatives_fluor_only_nodes
        )

        self.relevance_fluorine_derivative_atom_in_cycle_and_edge_to_fluor = PkaLRP.calculate_relevance_for_one_atom_and_edge_in_molecule(
            mol=self.mol, fluorine_group=self.fluorine_group, 
            node_relevances=self.node_relevances, edge_relevances=self.edge_relevances
        )

    @staticmethod
    def prepare_pKa_graph(SMILES):
        """
        Prepares a graph representation of the molecule for pKa prediction.

        Args:
            SMILES (str): The SMILES string representing the molecule.

        Returns:
            graph (DGLGraph): The graph representation of the molecule.
        """
        smiles_to_graph = SMILESToBigraph(node_featurizer=CanonicalAtomFeaturizer(),
                                          edge_featurizer=CanonicalBondFeaturizer())

        return smiles_to_graph(SMILES)

    def lrp(self, bg, node_feats, edge_feats):
        self.model.eval()
        
        node_feats_original = node_feats.clone()
        edge_feats_original = edge_feats.clone()

        all_node_relevances = {}
        all_edge_relevances = {}
        # node relevance
        for node_feat_index in range(len(node_feats)):
            
            self.model.zero_grad()

            x_node = node_feats_original.clone()
            x_node = Variable(x_node.data, requires_grad=True)

            h0 = x_node

            mask = torch.zeros(x_node.shape).to(self.device)
            mask[node_feat_index] = 1

            x_node = x_node * mask + (1 - mask) * x_node.data

            # TODO
            # AGGREGATE STEP FOR WALKS(currently only nodes)
            
            # forward
            node_feats = self.model.init_context(bg, x_node, edge_feats)
            for gnn in self.model.gnn_layers:
                node_feats = gnn(bg, node_feats)
            atom_pka = self.model.predict(node_feats)

            g_feats = torch.log10(torch.sum(atom_pka))

            # backward
            g_feats.backward(retain_graph=True)

            all_node_relevances[node_feat_index] = h0.data * h0.grad
            h0.grad.data.zero_()

        for edge_feat_index in range(len(edge_feats)):
            
            self.model.zero_grad()

            x_edge = edge_feats_original.clone()
            x_edge = Variable(x_edge.data, requires_grad=True)

            e0 = x_edge

            mask = torch.zeros(x_edge.shape).to(self.device)
            mask[edge_feat_index] = 1

            x_edge = x_edge * mask + (1 - mask) * x_edge.data
            
            # forward
            node_feats = self.model.init_context(bg, node_feats_original, x_edge)
            for gnn in self.model.gnn_layers:
                node_feats = gnn(bg, node_feats)
            atom_pka = self.model.predict(node_feats)

            g_feats = torch.log10(torch.sum(atom_pka))

            # backward
            g_feats.backward(retain_graph=True)

            all_edge_relevances[edge_feat_index] = e0.data * e0.grad
            e0.grad.data.zero_()

        # nodes relevance preprocessing
        for idx, rel in all_node_relevances.items():
            relevance_score = rel.data.sum().item()
            all_node_relevances[idx] = relevance_score

        # edges relevance preprocessing
        for idx, rel in all_edge_relevances.items():
            relevance_score = rel.data.sum().item()
            all_edge_relevances[idx] = relevance_score

        return all_node_relevances, all_edge_relevances

    def obtain_subgroups_from_mol(self, molecule):

        main_derivatives_molecule_atoms = []
        for atom in molecule.GetAtoms():
            atom_index = atom.GetIdx()
            main_derivatives_molecule_atoms.append(atom_index)

        # FLUORIC GROUP MATCHES
        f_group_smiles = functional_group_to_smiles[self.fluorine_group]
        f_group_mol = Chem.MolFromSmiles(f_group_smiles)
        f_group_matches = molecule.GetSubstructMatches(f_group_mol)

        # combining matches
        matches = [list(match) for match in f_group_matches]
        all_indices = set()
        for indices in matches:
            all_indices.update(indices)
        
        main_molecule_atoms = []
        for atom in molecule.GetAtoms():
            atom_index = atom.GetIdx()
            if atom_index not in all_indices:
                main_molecule_atoms.append(atom_index)
        matches += [main_molecule_atoms]

        return matches, main_derivatives_molecule_atoms

    @staticmethod
    def create_mol_as_bigraph(smiles):
        mol = Chem.MolFromSmiles(smiles)

        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)

        return mol 
    
    @staticmethod
    def calculate_derivative_relevance_in_molecule(mol, node_relevances: dict, 
                                                   derivatives_atoms: list = [],
                                                   relevance_mode: RelevanceMode = RelevanceMode.only_fluor):
        relevance_for_subgroup = 0
        for atom_index, node_relevance in node_relevances.items():
            if atom_index not in derivatives_atoms:
                continue
            
            atom_symbol = mol.GetAtoms()[atom_index].GetSymbol()
            if relevance_mode == RelevanceMode.entire_derivatives_fluor_only_nodes:
                relevance_for_subgroup += node_relevance
            elif relevance_mode == RelevanceMode.only_fluor:
                if atom_symbol.lower() == 'f':
                    relevance_for_subgroup += node_relevance
                
        return round(relevance_for_subgroup, 2)

    @staticmethod
    def calculate_relevance_for_one_atom_and_edge_in_molecule(mol, fluorine_group,
                                                              node_relevances,
                                                              edge_relevances):
        relevance = 0
        
        if "non" in fluorine_group:
            # add relevance for molecules without fluorine
            return relevance
        
        fluorine_group_smiles = functional_group_to_smiles[fluorine_group]
        if "non" not in fluorine_group:
            fluorine_group_smiles = "C" + fluorine_group_smiles
        
        fluorine_group_mol = Chem.MolFromSmiles(fluorine_group_smiles)
        f_group_matches = mol.GetSubstructMatches(fluorine_group_mol)

        node_relevance = node_relevances[f_group_matches[0][0]]
        edge_relevance = 0
        for f_group_match in f_group_matches:
            atom_from = f_group_match[0]
            atom_to = f_group_match[1]

            for mol_edge_idx in range(mol.GetNumBonds()):
                mol_begin_node = mol.GetBonds()[mol_edge_idx].GetBeginAtomIdx()
                mol_end_node = mol.GetBonds()[mol_edge_idx].GetEndAtomIdx()

                if (mol_begin_node == atom_from and mol_end_node == atom_to) or mol_begin_node == atom_to and mol_end_node == atom_from:
                    edge_relevance += edge_relevances[mol_edge_idx]

        # print(node_relevance)
        # print(edge_relevance)
        relevance = node_relevance + edge_relevance / len(f_group_matches)
        return round(relevance, 2)

        

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

    def extract_atoms_colors_from_relevances(self, mol, atoms_groups):
        atoms_colors = {}
        atoms = []
        bonds_colors = {}
        bonds = []
        # max_node_relevance = max(self.node_relevances.values())
        # min_node_relevance = min(self.node_relevances.values())
        max_node_relevance = 1
        min_node_relevance = -1
        for atom_index, node_relevance in self.node_relevances.items():
            mol.GetAtoms()[atom_index].SetProp("atomNote", str(round(self.node_relevances[atom_index], 4)))
                
            color_by_relevance = get_color(node_relevance, vmin=min_node_relevance, vmax=max_node_relevance)
            atoms_colors[atom_index] = color_by_relevance
            atoms.append(atom_index)

        group_to_relevance = {}
        for group_index in range(len(atoms_groups)):
            group = atoms_groups[group_index]
            group_to_relevance[group_index] = subgroup_relevance(group, self.node_relevances)
        
        # min_node_relevance = min(group_to_relevance.values()) * 2
        # max_node_relevance = max(group_to_relevance.values()) * 2

        # atoms = []
        # for group_index in range(len(atoms_groups)):
        #     group = atoms_groups[group_index]
        #     group_relevance = group_to_relevance[group_index]
            
        #     is_group_subscribed = False
        #     for atom_index in group:
        #         if not is_group_subscribed:
        #             mol.GetAtoms()[atom_index].SetProp("atomNote", str(round(group_relevance, 2)))
        #             is_group_subscribed = True

        #         color_by_relevance = get_color(group_relevance, vmin=min_node_relevance, vmax=max_node_relevance)
        #         atoms_colors[atom_index] = color_by_relevance
        #         atoms.append(atom_index)

        # edge relevance
        mol_edge_relevances, mol_edge_relevances_all_bonds = self.convert_bigraph_edges_relevance_to_mol_bonds(mol)
        max_edge_relevance = 1
        min_edge_relevance = -1
        for edge_index, edge_relevance in mol_edge_relevances.items():
            from_node, dest_node = mol.GetBonds()[edge_index].GetBeginAtomIdx(), mol.GetBonds()[edge_index].GetEndAtomIdx()

            color_by_relevance = get_color(edge_relevance, vmin=min_edge_relevance, vmax=max_edge_relevance)
            mol.GetBondWithIdx(edge_index).SetProp("bondNote", str(round(edge_relevance, 4)))

            bonds_colors[edge_index] = color_by_relevance
            bonds.append(edge_index)

        # edge relevance for connection between groups in molecule
        # for edge_idx in range(mol.GetNumBonds()):
        #     begin_node = mol.GetBonds()[edge_idx].GetBeginAtomIdx()
        #     end_node = mol.GetBonds()[edge_idx].GetEndAtomIdx()

        #     is_same_group, group_index = check_if_edge_in_one_group(
        #         groups=atoms_groups,
        #         start_atom_idx=begin_node,
        #         end_atom_idx=end_node
        #     )
            
        #     bond_relevance = (max_node_relevance + min_node_relevance) / 2
        #     if is_same_group:
        #         bond_relevance = group_to_relevance[group_index]
            
        #     bond_relevance = mol_edge_relevances[edge_idx]
            
                
        #     bond_color = get_color(bond_relevance, vmin=min_node_relevance, vmax=max_node_relevance)

        #     bonds.append(edge_idx)
        #     bonds_colors[edge_idx] = bond_color

        # print("Atoms:", atoms)
        # print("Atoms colors:", atoms_colors)
        # print("Bonds:", bonds)
        # print("Bonds colors:", bonds_colors)
        return atoms, atoms_colors, bonds, bonds_colors

    def save_molecule_with_relevances(self, output_svg_path=None, output_png_path=None): 

        drawer = rdMolDraw2D.MolDraw2DCairo(1200, 700)

        atoms, atoms_colors, bonds, bonds_colors = self.extract_atoms_colors_from_relevances(self.mol, self.atoms_groups)

        drawer.DrawMolecule(self.mol, highlightAtoms=atoms, highlightAtomColors=atoms_colors, 
                            highlightBonds=bonds, highlightBondColors=bonds_colors)

        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        
        # if output_svg_path is not None:
        #     output_svg = output_svg_path
        #     with open(output_svg, "w") as svg_file:
        #         svg_file.write(svg) 

        if output_png_path is not None:
            with open(output_png_path, "wb") as png_file:
                png_file.write(drawer.GetDrawingText())

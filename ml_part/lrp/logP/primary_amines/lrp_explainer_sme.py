import io
from PIL import Image

from rdkit import Chem
from rdkit.Chem import rdmolops, rdmolfiles
from rdkit.Chem.Draw import rdMolDraw2D

import torch
from torch.autograd import Variable

from dgllife.utils.mol_to_graph import SMILESToBigraph
from dgllife.utils import AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer

from logP_model_service import LogPModelService
from utils import get_color, subgroup_relevance, check_if_edge_in_one_group, normalize_to_minus_one_to_one, find_the_furthest_atom
from constants import functional_group_to_smiles

class LogpLRP:
    def __init__(self,
                 model_service: LogPModelService = None,
                 SMILES: str = None,
                 identificator: str = None,
                 fluorine_group: str = None,
                 is_centrize_relevances: bool = True
                 ) -> None:
        self.device = "cpu"
        self.SMILES = SMILES
        self.identificator = identificator
        self.fluorine_group = fluorine_group

        if model_service is None:
            model_name = 'GCN_attentivefp_lipophilicity'
            model_weights = r'ml_part\weights\logP_dgllife_lipophilicity\best_models\GNN_attentivefp_model_logP_best_loss.pth'
            # model_weights = r'ml_part\weights\logP_dgllife_lipophilicity\with_explicit_hydrogen\GCN_attentivefp_Lipophilicity_logP_best_loss.pth'
            
            model_service = LogPModelService(
                model_name=model_name,
                model_weights_path=model_weights
            )
            
        self.model = model_service.logP_model

        self.mol = LogpLRP.create_mol_as_bigraph(self.SMILES)
        self.atoms_groups, self.derivatives_atoms = self.obtain_subgroups_from_mol(molecule=self.mol)

        self.bg = self.prepare_logP_graph(SMILES=SMILES)
        self.node_features = self.bg.ndata['h']

        self.node_relevances = self.lrp(self.bg, self.node_features, self.atoms_groups)
        if is_centrize_relevances:
            self.node_relevances = normalize_to_minus_one_to_one(data=self.node_relevances)

        self.relevance_fluorine_group = LogpLRP.calculate_relevance_for_f_group_in_molecule(
            mol=self.mol, fluorine_group=self.fluorine_group, 
            node_relevances=self.node_relevances
        )

    @staticmethod
    def prepare_logP_graph(SMILES):
        """
        Prepares a graph representation of the molecule for logP prediction.

        Args:
            SMILES (str): The SMILES string representing the molecule.

        Returns:
            graph (DGLGraph): The graph representation of the molecule.
        """
        smiles_to_graph = SMILESToBigraph(add_self_loop=True,
                                          node_featurizer=AttentiveFPAtomFeaturizer(),
                                          edge_featurizer=AttentiveFPBondFeaturizer(self_loop=True),
                                          explicit_hydrogens=False)
    
        return smiles_to_graph(SMILES)

    def lrp(self, bg, node_feats, atoms_subgroups):
        self.model.eval()
        
        node_feats_original = node_feats.clone()

        all_node_relevances = {}
        # node relevance
        for atoms_subgroup in atoms_subgroups:
            self.model.zero_grad()

            x_node = node_feats_original.clone()
            x_node = Variable(x_node.data, requires_grad=True)

            h0 = x_node

            mask = torch.zeros(x_node.shape).to(self.device)
            for atom_index in atoms_subgroup:
                mask[atom_index] = 1
            
            x_node = x_node * mask + (1 - mask) * x_node.data

            # forward
            node_feats = self.model.gnn(bg, x_node)
            graph_feats = self.model.readout(bg, node_feats)

            logP_prediction = self.model.predict(graph_feats)

            # backward
            logP_prediction.backward(retain_graph=True)

            for atom_index in atoms_subgroup:
                all_node_relevances[atom_index] = h0.data * h0.grad
            h0.grad.data.zero_()

        # nodes relevance preprocessing
        for idx, rel in all_node_relevances.items():
            relevance_score = rel.data.sum().item()
            all_node_relevances[idx] = relevance_score

        return all_node_relevances

    def obtain_subgroups_from_mol(self, molecule):

        # LogP ring match
        logP_ring = 'C1=CC=CC=C1'
        mol_ring = Chem.MolFromSmiles(logP_ring)
        mol_ring_matches = molecule.GetSubstructMatches(mol_ring)

        # NCOOH match
        NCOOH_SMILE = 'C(=O)'
        if "secondary" in self.identificator.lower():
            NCOOH_SMILE = 'C(=O)'
        ncooh_mol = Chem.MolFromSmiles(NCOOH_SMILE)
        ncooh_matches = molecule.GetSubstructMatches(ncooh_mol)

        atoms_cooh_and_logp_ring = [list(match) for match in mol_ring_matches] \
                + [list(match) for match in ncooh_matches]
        main_derivatives_molecule_atoms = []
        for atom in molecule.GetAtoms():
            atom_index = atom.GetIdx()
            if atom_index not in atoms_cooh_and_logp_ring:
                main_derivatives_molecule_atoms.append(atom_index)

        # FLUORIC GROUP MATCHES
        f_group_smiles = functional_group_to_smiles[self.fluorine_group]
        f_group_mol = Chem.MolFromSmiles(f_group_smiles)
        f_group_matches = molecule.GetSubstructMatches(f_group_mol)

        # combining matches
        matches = [list(match) for match in mol_ring_matches] \
                + [list(match) for match in ncooh_matches] \
                + [list(match) for match in f_group_matches]
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
        # mol = Chem.AddHs(mol)

        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)

        return mol 

    @staticmethod
    def get_atom_id_by_symbol(mol, atom_symbol):
        for atom in mol.GetAtoms():
            if atom.GetSymbol().lower() == atom_symbol.lower():
                return atom.GetIdx() 

    @staticmethod
    def calculate_relevance_for_f_group_in_molecule(mol, fluorine_group,
                                                              node_relevances):
        relevance, number_of_relevances = 0, 0
        
        if "non" in fluorine_group:
            # add relevance for molecules without fluorine
            nitrogen_atom_id = LogpLRP.get_atom_id_by_symbol(mol=mol, atom_symbol='N')
            # find the furthest atom from nitrogen
            atom_id, distance = find_the_furthest_atom(mol=mol,
                                                       atom_id=nitrogen_atom_id)

            return node_relevances[atom_id]
        
        fluorine_group_smiles = "C" + functional_group_to_smiles[fluorine_group]
        
        fluorine_group_mol = Chem.MolFromSmiles(fluorine_group_smiles)
        f_group_matches = mol.GetSubstructMatches(fluorine_group_mol)
        
        unique_atoms = set()
        for f_group_match in f_group_matches:
            for atom_index in f_group_match:
                unique_atoms.add(atom_index)
        for unique_atom_index in unique_atoms:
            relevance += node_relevances[unique_atom_index]
            number_of_relevances += 1

        return round(relevance / number_of_relevances, 3)

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
            mol.GetAtoms()[atom_index].SetProp("atomNote", str(round(self.node_relevances[atom_index], 2)))
                
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
        for edge_idx in range(mol.GetNumBonds()):
            begin_node = mol.GetBonds()[edge_idx].GetBeginAtomIdx()
            end_node = mol.GetBonds()[edge_idx].GetEndAtomIdx()

            is_same_group, group_index = check_if_edge_in_one_group(
                groups=atoms_groups,
                start_atom_idx=begin_node,
                end_atom_idx=end_node
            )
            
            bond_relevance = (max_node_relevance + min_node_relevance) / 2
            if is_same_group:
                bond_relevance = group_to_relevance[group_index]
                
            bond_color = get_color(bond_relevance, vmin=min_node_relevance, vmax=max_node_relevance)

            bonds.append(edge_idx)
            bonds_colors[edge_idx] = bond_color

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

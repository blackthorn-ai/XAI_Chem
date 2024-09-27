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
from utils import get_color, subgroup_relevance, check_if_edge_in_one_group, normalize_to_minus_one_to_one, find_the_furthest_atom
from constants import functional_group_to_smiles
from constants import Identificator, RelevanceMode

class PkaLRP:
    def __init__(self,
                 model_service: PkaModelService = None,
                 SMILES: str = None,
                 identificator: Identificator = None,
                 fluorine_group: str = None,
                 is_centrize_relevances: bool = True,
                 ) -> None:
        self.device = "cpu"
        self.SMILES = SMILES
        self.identificator = identificator.name
        self.fluorine_group = fluorine_group

        if model_service is None:
            
            model_service = PkaModelService(
                identificator=identificator,
                is_combined_model=True
            )
            
        self.model = model_service.pKa_model

        self.importance_fluorine_group = self.calculate_relevance_for_f_group_in_molecule(
            smiles=SMILES, fluorine_group=self.fluorine_group, 
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

    @staticmethod
    def create_mol_as_bigraph(smiles):
        mol = Chem.MolFromSmiles(smiles)

        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        if len(new_order) != 0: 
            mol = rdmolops.RenumberAtoms(mol, new_order)

        return mol 
    
    @staticmethod
    def remove_substructure(mol, mol_to_remove):
        if mol.HasSubstructMatch(mol_to_remove):
            mol = Chem.DeleteSubstructs(mol, mol_to_remove)

        return mol
    
    def calculate_relevance_for_f_group_in_molecule(self, smiles, fluorine_group):
        mol_with_F = PkaLRP.create_mol_as_bigraph(smiles=smiles)

        f_group_smiles = functional_group_to_smiles[fluorine_group]
        fluorine_mol = PkaLRP.create_mol_as_bigraph(smiles=f_group_smiles)
        
        mol_without_F = PkaLRP.remove_substructure(mol=mol_with_F, mol_to_remove=fluorine_mol)
        smiles_without_F = Chem.MolToSmiles(mol_without_F)

        pka_with_F = self.predict(smiles).item()
        pka_without_F = self.predict(smiles_without_F).item()

        print(pka_with_F, pka_without_F)

        return round(pka_with_F - pka_without_F, 3)
        

    def predict(self, smiles):
        self.model.eval()

        bg = self.prepare_pKa_graph(SMILES=smiles)
        
        prediction, _ = self.model(bg, bg.ndata['h'], bg.edata['e'])

        return prediction

    def extract_atoms_colors_from_relevances(self):
        atoms_colors = {}
        atoms = []
        bonds_colors = {}
        bonds = []

        mol_with_F = PkaLRP.create_mol_as_bigraph(smiles=self.SMILES)
        f_group_smiles = functional_group_to_smiles[self.fluorine_group]
        fluorine_mol = PkaLRP.create_mol_as_bigraph(smiles=f_group_smiles)

        f_group_matches = mol_with_F.GetSubstructMatches(fluorine_mol)

        SME = get_color(self.importance_fluorine_group, vmin=-4.695, vmax=0)
        if len(f_group_matches) != 0:
            mol_with_F.GetAtoms()[f_group_matches[0][0]].SetProp("atomNote", str(round(self.importance_fluorine_group, 3)))

        for match in f_group_matches:
            for atom in match:
                atoms.append(atom)
                atoms_colors[atom] = SME
        
        for mol_edge_idx in range(mol_with_F.GetNumBonds()):
            mol_begin_node = mol_with_F.GetBonds()[mol_edge_idx].GetBeginAtomIdx()
            mol_end_node = mol_with_F.GetBonds()[mol_edge_idx].GetEndAtomIdx()

            if mol_begin_node in atoms and mol_end_node in atoms:
                bonds_colors[mol_edge_idx] = SME
                bonds.append(mol_edge_idx)
        
        return mol_with_F, atoms, atoms_colors, bonds, bonds_colors


    def save_molecule_with_relevances(self, output_png_path=None): 

        drawer = rdMolDraw2D.MolDraw2DCairo(800, 470)

        mol_with_F, atoms, atoms_colors, bonds, bonds_colors = self.extract_atoms_colors_from_relevances()

        drawer.DrawMolecule(mol_with_F, highlightAtoms=atoms, highlightAtomColors=atoms_colors, 
                            highlightBonds=bonds, highlightBondColors=bonds_colors)

        drawer.FinishDrawing()

        if output_png_path is not None:
            with open(output_png_path, "wb") as png_file:
                png_file.write(drawer.GetDrawingText())

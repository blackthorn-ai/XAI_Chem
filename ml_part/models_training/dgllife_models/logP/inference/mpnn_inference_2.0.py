import dgl
from rdkit import Chem
from rdkit.Chem import AllChem
from dgllife.data.alchemy import alchemy_nodes, alchemy_edges
from dgllife.utils import mol_to_bigraph, mol_to_complete_graph, smiles_to_complete_graph

SMILES = 'FC1(F)CN(C1)C(=O)C1=CC=CC=C1'

# Створіть об'єкт RDKit з SMILES
mol = Chem.MolFromSmiles(SMILES)
mol = AllChem.EmbedMolecule(mol)

# Використайте alchemy_nodes і alchemy_edges для створення фічей атомів і зв'язків
atom_feats_dict = alchemy_nodes(mol)
bond_feats_dict = alchemy_edges(mol)

print(mol)

print(atom_feats_dict.keys())

print(atom_feats_dict['n_feat'])
print(bond_feats_dict['e_feat'])
print(atom_feats_dict['node_type'])

# Створіть DGLGraph з фічами атомів і зв'язків
g = mol_to_complete_graph(mol)

# Тепер ви можете використовувати g з моделлю MPNN

import os
import numpy as np
from rdkit import Chem
from openbabel import pybel
from rdkit.Chem import rdPartialCharges, rdFreeSASA, GetPeriodicTable, PeriodicTable, rdMolTransforms
from rdkit import RDConfig
from rdkit.Chem import FragmentCatalog, AllChem
import math
from scipy import stats
import pandas as pd

import numpy as np
np.random.seed(0)

from scipy import stats
from constants import mandatory_features, functional_group_to_smiles


def calculate_distance(point1, point2):
    return ((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2 + (point2[2] - point1[2])**2)**0.5


def calculate_dipole_moment(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)

    AllChem.MMFFSanitizeMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)

    rdPartialCharges.ComputeGasteigerCharges(mol)

    charges = []
    coordinates = []
    for atom in mol.GetAtoms():
        pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
        charge = atom.GetDoubleProp("_GasteigerCharge")

        charges.append(charge)
        coordinates.append(pos)


    charges_multiply_coordinates = coordinates.copy()
    for charges_multiply_coordinate_index in range(len(charges_multiply_coordinates)):
        for coordinate in charges_multiply_coordinates[charges_multiply_coordinate_index]:
            coordinate *= charges[charges_multiply_coordinate_index]

    dipole_moment_vector = [0, 0, 0]
    for charges_multiply_coordinate_index in range(len(charges_multiply_coordinates)):
        # print(charges_multiply_coordinates[charges_multiply_coordinate_index])
        dipole_moment_vector[0] += charges_multiply_coordinates[charges_multiply_coordinate_index][0]
        dipole_moment_vector[1] += charges_multiply_coordinates[charges_multiply_coordinate_index][1]
        dipole_moment_vector[2] += charges_multiply_coordinates[charges_multiply_coordinate_index][2]
        # print(dipole_moment_vector)

    dipole_moment = math.sqrt(pow(dipole_moment_vector[0], 2) + pow(dipole_moment_vector[1], 2) + pow(dipole_moment_vector[2], 2))

    return dipole_moment


def calculate_sasa(mol):
    mol_classify = rdFreeSASA.classifyAtoms(mol)
    sasa = rdFreeSASA.CalcSASA(mol, mol_classify)
    
    return sasa


def calculate_positive_negative_charges_area(mol):
    pt = GetPeriodicTable()
    rdPartialCharges.ComputeGasteigerCharges(mol)
    positive_charge_area, negative_charge_area = 0, 0
    for atom in mol.GetAtoms():
        rvdw = PeriodicTable.GetRvdw(pt, atom.GetAtomicNum())
        vdw_area = 4 * math.pi * pow(rvdw, 2)
        if atom.GetDoubleProp("_GasteigerCharge") < 0: 
            negative_charge_area += vdw_area
        else: 
            positive_charge_area += vdw_area
    
    return positive_charge_area, negative_charge_area


def calculate_dipole_moment_openbabel(smiles):
    mol = pybel.readstring("smi", smiles)
    
    mol.addh()
    mol.make3D()

    charges = [atom.OBAtom.GetPartialCharge() for atom in mol.atoms]
    coordinates = [(atom.coords) for atom in mol.atoms]
    
    for coordinate_index in range(len(coordinates)):
        coordinates[coordinate_index] = pow(pow(coordinates[coordinate_index][0], 2) + pow(coordinates[coordinate_index][1], 2) + pow(coordinates[coordinate_index][2], 2), 1 / 2.)

    dipole_moment = [charge * coord for charge, coord in zip(charges, coordinates)]


    return sum(dipole_moment)


def calculate_linear_distance(mol):
    coords = mol.GetConformer().GetPositions()

    distances = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1))

    average_distance = np.mean(distances)
    return average_distance


def calculate_f_group_freedom(functional_group):
    functional_group_to_smiles = {
            "CF3": 0, 
            "CH2F": 1, 
            "gem-CF2": 0, 
            "CHF2": 1,
            "CHF": 1,
            "non-F": 1,
        }
    
    return functional_group_to_smiles[functional_group]


def mol_cycles_amount(mol):
    sssr = Chem.GetSSSR(mol)

    num_rings = len(sssr)
    return num_rings


def atoms_num_in_cycles_divide_by_amount_cycles(mol, atoms_in_right):
    sssr = Chem.GetSSSR(mol)
    amount_cycles = len(sssr)
    
    if amount_cycles == 0:
        return 0

    atoms_idxs = set()

    for i, ring in enumerate(sssr):
        for atom_index in ring:
            atoms_idxs.add(atom_index)
    
    atoms_num_in_cycles = len(atoms_idxs)
    # if atoms_in_right != atoms_num_in_cycles:
    #     raise Exception("!!!Calculated data and data from rdkit is not similar!!!")
    
    return atoms_num_in_cycles / amount_cycles


def convert_atom_ring_type(ring_type):
    convert_dict = {
        "Aromatic": 1,
        "Alifatic": 0
    }

    if ring_type not in convert_dict:
        return None
    
    return convert_dict[ring_type]


def nature_of_cycle(mol):
    sssr = Chem.GetSSSR(mol)
    amount_cycles = len(sssr)
    if amount_cycles == 0:
        return 0
    
    aromatic_dict = {}
    for i, ring in enumerate(sssr):
        for atom_index in ring:
            atom = mol.GetAtomWithIdx(atom_index)
            ring_type = "Aromatic" if atom.GetIsAromatic() else "Alifatic"
            # print(f"Атом {atom.GetIdx() + 1} у кільці {i + 1} типу: {ring_type}")
            aromatic_dict[atom.GetIdx() + 1] = convert_atom_ring_type(ring_type)
    
    ring_type_ratio = sum(aromatic_dict.values()) / len(aromatic_dict.values())
    return ring_type_ratio

def is_chirality(mol):
    chirality_centers = Chem.FindMolChiralCenters(mol)
    if chirality_centers:
        return True
    else:
        return False


def get_amount_of_chiral_centers(mol):
    chirality_centers = Chem.FindMolChiralCenters(mol)
    
    amount_of_chiral_centers = len(chirality_centers)
    
    return amount_of_chiral_centers


def detect_outlier_indexes(data=list(), threshold=3):
    z = abs(stats.zscore(data, nan_policy='omit'))
    
    outlier_indexes = np.where(z > threshold)[0]

    return outlier_indexes


def split_features_by_normalization(df):
    normal_features, not_normal_features = [], []
    for feature_name in df.keys():

        if df[feature_name].dtype == object:
            continue

        if len(df[feature_name].unique()) < 5:
            continue

        features = df[feature_name].to_list()
        res_features = stats.shapiro(features)

        if res_features.pvalue < 0.05:
            not_normal_features.append(feature_name)
        else:
            normal_features.append(feature_name)
    
    return normal_features, not_normal_features


def remove_nan_from_corr_matrix(corr):
    corr_indexes = corr.keys()

    index_to_drop = []

    for corr_feature in corr_indexes:
        if np.isnan(corr.loc[corr_feature, corr_feature]):
            index_to_drop.append(corr_feature)

    corr_matrix = corr.drop(index=index_to_drop, columns=index_to_drop)

    return corr_matrix


def get_most_correlated_values(corr_matrix,
                               threshold=0.7,
                               threshold_for_mandatory=0.9):
    corr_matrix = remove_nan_from_corr_matrix(corr_matrix)
    correlated_values = set()

    flds = list(corr_matrix.columns)
    # print(Corr_Matrix)
    corr_values = corr_matrix.values

    for row_index in range(len(flds)):
        if flds[row_index] in correlated_values:
            continue
        
        for column_index in range(row_index+1, len(flds)):
            if flds[column_index] in correlated_values or flds[row_index] in correlated_values:
                continue
            if flds[row_index] in mandatory_features and flds[column_index] in mandatory_features and abs(corr_values[row_index, column_index]) <= threshold_for_mandatory:
                continue
            if flds[row_index] in mandatory_features and flds[column_index] in mandatory_features and (flds[row_index] == 'nF' or flds[column_index] == 'nF'):
                continue
            if abs(corr_values[row_index, column_index]) > threshold:

                if flds[row_index] in mandatory_features:
                    if flds[column_index] in mandatory_features:
                        print(f"Mandatory feature: \033[1m{flds[column_index]}\033[0m was deleted due to high correlation with another mandatory feature: \033[1m{flds[row_index]}\033[0m, corr value: {corr_values[row_index, column_index]}")
                
                    correlated_values.add(flds[column_index])
                
                elif flds[column_index] in mandatory_features:
                    if flds[row_index] in mandatory_features:
                        print(f"Mandatory feature: \033[1m{flds[row_index]}\033[0m was deleted due to high correlation with another mandatory feature: \033[1m{flds[column_index]}\033[0m, corr value: {corr_values[row_index, column_index]}")
                
                    correlated_values.add(flds[row_index])
                else:
                    correlated_values.add(flds[column_index])

    return correlated_values


def remove_features_with_same_values(df):
    df_row = df.iloc[0]

    not_array_columns = []
    for feature_name, value in df_row.items():
        if type(value) != type([]):
            not_array_columns.append(feature_name)
    not_array_df = df[not_array_columns]

    nunique = not_array_df.nunique()
    
    cols_to_drop = nunique[nunique == 1].index

    for col_to_drop in cols_to_drop:
        if col_to_drop in mandatory_features:
            print(f"Mandatory feature: \033[1m{col_to_drop}\033[0m was deleted due to only 1 unique value")
    
    df.drop(cols_to_drop, axis=1)
    
    return df


def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)


def normalize_values(result_df:pd.DataFrame(), 
                     normalize_pKa:bool=True, 
                     normalize_logP:bool=True):
    if normalize_pKa:
        logP = result_df['pKa']
        
        original_pKa_values = np.array(logP)

        ranked_values = stats.rankdata(original_pKa_values)
        quantile_normalized_values = stats.norm.ppf(ranked_values / (len(ranked_values) + 1))

        result_df['pKa_normalized'] = quantile_normalized_values.tolist()
    
    if normalize_logP:
        logP = result_df['logP']
        
        original_LogP_values = np.array(logP)

        ranked_values = stats.rankdata(original_LogP_values)
        quantile_normalized_values = stats.norm.ppf(ranked_values / (len(ranked_values) + 1))

        result_df['logP_normalized'] = quantile_normalized_values.tolist()

    return result_df


def extract_functional_groups(smiles, f_group):
    # returns:
    # dictionary with existance functional groups

    mol = Chem.MolFromSmiles(smiles)
    AllChem.EmbedMolecule(mol, randomSeed=42)

    f_functional_group_smiles = functional_group_to_smiles[f_group]
    
    f_group_mol = Chem.MolFromSmiles(f_functional_group_smiles)
    f_group_matches_amount = len(mol.GetSubstructMatches(f_group_mol))

    f_group_dict = {
            "CF3": 0, 
            "CH2F": 0, 
            "gem-CF2": 0, 
            "CHF2": 0,
            "CHF": 0
        }
    if f_group in f_group_dict:
        f_group_dict[f_group] = f_group_matches_amount

    # 5 functional groups
    functional_groups = {"-C(=O)O": 0, "-O": 0, "=O": 0, "-N": 0}
    fName=os.path.join(RDConfig.RDDataDir,'FunctionalGroups.txt')
    amount_of_functional_groups = 39

    fparams = FragmentCatalog.FragCatParams(1,6,fName)
    fparams.GetNumFuncGroups()

    for func_froup_index in range(amount_of_functional_groups):
        if fparams.GetFuncGroup(func_froup_index).GetProp('_Name') not in functional_groups.keys():
            continue

        pattern = fparams.GetFuncGroup(func_froup_index)
        matches = mol.GetSubstructMatches(pattern)
        if len(matches) == 0:
            continue

        functional_groups[fparams.GetFuncGroup(func_froup_index).GetProp('_Name')] = len(matches)

    f_group_dict.update(functional_groups)

    functional_groups_upd_names = {}
    for func_group_name, amount in f_group_dict.items():
        functional_groups_upd_names[f'amount_of_{func_group_name}'] = amount

    return functional_groups_upd_names, functional_groups


def find_fluorine_indices(mol):
    
    if mol is None:
        print("Невірний SMILES-рядок.")
        return None
    
    fluorine_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'F']
    
    return fluorine_indices


def all_distance_between_functional_groups_and_f(smiles, functional_groups_dict):
    functional_groups_distances_dict = {func_group : 0 for func_group, amount in functional_groups_dict.items()}

    mol_for_matrix = Chem.MolFromSmiles(smiles)
    mol_for_matrix = Chem.AddHs(mol_for_matrix)
    AllChem.EmbedMolecule(mol_for_matrix, randomSeed=42)
    
    distMat = Chem.Get3DDistanceMatrix(mol_for_matrix)

    fluorine_indices = find_fluorine_indices(mol_for_matrix)
    if len(fluorine_indices) == 0:
        return functional_groups_distances_dict

    fName=os.path.join(RDConfig.RDDataDir,'FunctionalGroups.txt')
    amount_of_functional_groups = 39

    fparams = FragmentCatalog.FragCatParams(1,6,fName)
    fparams.GetNumFuncGroups()

    mol = Chem.MolFromSmiles(smiles)
    for func_froup_index in range(amount_of_functional_groups):
        if fparams.GetFuncGroup(func_froup_index).GetProp('_Name') not in functional_groups_distances_dict.keys():
            continue

        pattern = fparams.GetFuncGroup(func_froup_index)
        matches = mol.GetSubstructMatches(pattern)
        if len(matches) == 0:
            continue
        
        avg_distance, amount_of_distances = 0, 0
        for match in matches:
            for fluorine_index in fluorine_indices:
                avg_distance += distMat[match[0]][fluorine_index]
                amount_of_distances += 1

        functional_groups_distances_dict[fparams.GetFuncGroup(func_froup_index).GetProp('_Name')] = \
            avg_distance / amount_of_distances

    functional_groups_distances_upd_name_dict = {}
    for func_group_name, distance in functional_groups_distances_dict.items():
        functional_groups_distances_upd_name_dict[f'{func_group_name}_to_F_distance'] = abs(distance)

    return functional_groups_distances_upd_name_dict


def calculate_dihedral_angle(mol, jAtomId, kAtomId):
    jAtom = mol.GetAtomWithIdx(jAtomId)
    kAtom = mol.GetAtomWithIdx(kAtomId)
    
    neighbors_of_atomJ_neigbors = jAtom.GetNeighbors()
    neighbors_of_atomK_neigbors = kAtom.GetNeighbors()

    iAtomId = None
    for neighbor_atom in neighbors_of_atomJ_neigbors:
        neighbor_atom_id = neighbor_atom.GetIdx()
        if neighbor_atom_id != kAtomId:
            iAtomId = neighbor_atom_id
            break
    
    lAtomId = None
    for neighbor_atom in neighbors_of_atomK_neigbors:
        neighbor_atom_id = neighbor_atom.GetIdx()
        if neighbor_atom_id != jAtomId:
            lAtomId = neighbor_atom_id
            break
    
    if iAtomId == None or lAtomId == None:
        return 0
    
    conf = mol.GetConformer()
    return rdMolTransforms.GetDihedralRad(conf, iAtomId, jAtomId, kAtomId, lAtomId)



def all_dihedral_angles_f_group_molecule(smiles, f_group):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)

    f_group_array = [0 for f_index in range(len(functional_group_to_smiles.keys()))]

    f_functional_group_smiles = functional_group_to_smiles[f_group]
    
    f_group_mol = Chem.MolFromSmiles(f_functional_group_smiles)
    f_group_matches = mol.GetSubstructMatches(f_group_mol)
    
    functional_group_index = list(functional_group_to_smiles.keys()).index(f_group)

    for f_group_match in f_group_matches:
        atom1_idx = f_group_match[0]
        atom2_idx = f_group_match[1]
        break
    
    if len(f_group_matches) > 0:
        f_group_array[functional_group_index] = calculate_dihedral_angle(mol, atom1_idx, atom2_idx)


    # for each functional group find atoms and dihedral angle
    functional_groups = ["-C(=O)O", "-O", "=O", "-N"]
    functional_groups_angles = [0 for func_group_name in functional_groups]

    fName=os.path.join(RDConfig.RDDataDir,'FunctionalGroups.txt')
    amount_of_functional_groups = 39

    fparams = FragmentCatalog.FragCatParams(1,6,fName)
    fparams.GetNumFuncGroups()

    mol_without_hidrogen = Chem.MolFromSmiles(smiles)
    for func_froup_index in range(amount_of_functional_groups):
        func_group_name = fparams.GetFuncGroup(func_froup_index).GetProp('_Name')
        if fparams.GetFuncGroup(func_froup_index).GetProp('_Name') not in functional_groups:
            continue

        pattern = fparams.GetFuncGroup(func_froup_index)
        matches = mol_without_hidrogen.GetSubstructMatches(pattern)
        if len(matches) == 0:
            continue

        dihedral_angles_sum = 0
        for match in matches:
            atom1_idx = match[0]
            atom2_idx = match[1]

            angle = calculate_dihedral_angle(mol, atom1_idx, atom2_idx)
            dihedral_angles_sum += angle

        functional_groups_angles[functional_groups.index(fparams.GetFuncGroup(func_froup_index).GetProp('_Name'))] = dihedral_angles_sum

    # print(f_group_array)
    # print(functional_groups_angles)
    combined_angles = f_group_array + functional_groups_angles
    combined_angles_dict = {}
    for group_index in range(len(combined_angles)):
        group_name = (list(functional_group_to_smiles.keys()) + functional_groups)[group_index]
        combined_angles_dict[f"dihedral_angle_{group_name}"] = combined_angles[group_index]

    return combined_angles_dict

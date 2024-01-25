from rdkit import Chem
from mordred import Calculator, descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors, AllChem, rdFreeSASA, rdPartialCharges, PeriodicTable, GetPeriodicTable
import pandas as pd
import math
import pickle
from tqdm import tqdm
import numpy as np
from scipy import stats

from constants import mandatory_features
from utils_3d import Molecule3DFeatures
from utils import calculate_linear_distance, calculate_f_group_freedom,\
      mol_cycles_amount, atoms_num_in_cycles_divide_by_amount_cycles, nature_of_cycle,\
      get_amount_of_chiral_centers, calculate_dipole_moment, calculate_sasa, calculate_positive_negative_charges_area,\
      detect_outlier_indexes, remove_nan_from_corr_matrix, remove_features_with_same_values,\
      has_numbers, normalize_values, split_features_by_normalization, get_most_correlated_values,\
      extract_functional_groups, all_distance_between_functional_groups_and_f, all_dihedral_angles_f_group_molecule


def obtain_mordred_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)

    calc = Calculator(descriptors, ignore_3D=False)
    df = calc.pandas([mol], quiet=True)

    mordred_dict = {}
    for value, row in df.iterrows():
        for key in row.keys():
            if "ring" in key.lower() and has_numbers(key):
                continue
            if type(row[key]) in [int, float]:
                mordred_dict[key] = row[key]

    return mordred_dict


def obtain_features_rdkit(df_row):
    smiles = df_row['Smiles']
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    pt = GetPeriodicTable()
    # cis/trans
    cis_trans = "" if pd.isnull(df_row['Stereo, F to FG']) else df_row['Stereo, F to FG']
    # Linear path(s) F to FG
    f_to_fg = df_row['Linear path(s) F to FG']
    # F atom fraction
    f_atom_fraction = df_row['F atom fraction']
    # dipole moment
    whole_dipole_momentum = calculate_dipole_moment(smiles) 
    # molecule volume
    molecule_volume = AllChem.ComputeMolVolume(mol)
    # molecule weight
    molecule_weight = df_row['MW']
    # calc sasa
    sasa = calculate_sasa(mol)
    # positive/negative area partial charges
    positive_charge_area, negative_charge_area = calculate_positive_negative_charges_area(mol)
    # TPSA + F
    tpsa = Descriptors.TPSA(mol)
    f = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol().lower() == 'f')
    tpsa_f = tpsa + f
    # linear distance in space
    linear_distance = calculate_linear_distance(mol)
    # Ступінь свободи F групи
    f_group_freedom = calculate_f_group_freedom(df_row['F group'])
    # Кількість циклів
    mol_num_cycles = mol_cycles_amount(mol)
    # Кількість атомів в циклі / кількість циклів
    atoms_num_in_cycles = atoms_num_in_cycles_divide_by_amount_cycles(mol, df_row['Atoms in ring'])
    # nature of cycles
    atom_alim_cycle = nature_of_cycle(mol)
    # Chirality
    chirality = get_amount_of_chiral_centers(mol)
    # functional groups
    functional_groups_amount_dict, functional_groups_dict = extract_functional_groups(smiles, df_row['F group'])
    # print(functional_groups_amount_dict, df_row['F group'], smiles)
    # dihedral angle between functional groups and f groups
    all_distance_from_group_to_f = all_distance_between_functional_groups_and_f(smiles, functional_groups_dict)
    # dihedral angle between functional groups and molecule
    dihedral_angles_f_group_dict = all_dihedral_angles_f_group_molecule(smiles, df_row['F group'])
    print(dihedral_angles_f_group_dict)
    # print(dihedral_angles_f_group_array, df_row['F group'])
    rdkit_features = {"cis/trans": cis_trans,
                      "f_to_fg": f_to_fg,
                      "f_atom_fraction": f_atom_fraction,
                      "dipole_moment": round(whole_dipole_momentum),
                      "mol_volume": round(molecule_volume, 2),
                      "mol_weight": round(molecule_weight, 2),
                      "sasa": round(sasa, 2),
                    #   "negative_area_charges": round(negative_charge_area, 2),
                    #   "positive_area_charges": round(positive_charge_area, 2),
                      "tpsa": tpsa,
                      "tpsa+f": tpsa_f,
                      "linear_distance": round(linear_distance, 2),
                      "f_freedom": round(f_group_freedom, 2),
                      "mol_num_cycles": mol_num_cycles,
                      "avg_atoms_in_cycle": round(atoms_num_in_cycles, 2),
                      "mol_nature": atom_alim_cycle,
                      "chirality": round(chirality, 2)
                      }
    rdkit_features.update(functional_groups_amount_dict)
    rdkit_features.update(all_distance_from_group_to_f)
    rdkit_features.update(dihedral_angles_f_group_dict)
    return rdkit_features


def obtain_features(df_row):
    smiles = df_row['Smiles']
    
    rdkit_features = obtain_features_rdkit(df_row)
    mordred_features = obtain_mordred_features(smiles)

    features = dict(rdkit_features)
    features.update(mordred_features)

    return features


def obtain_y(df_row):
    pKa = df_row['pKa']
    logP = df_row['LogP']

    target_data = {
        'pKa': pKa,
        'logP': logP}
    
    return target_data


def calculate_correlation(features):
    df = pd.DataFrame(features)
    df['cis/trans'] = df['cis/trans'].astype('category').cat.codes

    amount_of_rdkit_features = 14

    flds = list(df.columns)

    rdkit_features_columns = df.iloc[:, :amount_of_rdkit_features]
    amount_of_mordred_features_per_sample = 10

    features_to_remove = set()

    for start_index in range(amount_of_rdkit_features, len(df.keys()), amount_of_mordred_features_per_sample):
        mordred_features_columns = df.iloc[:, start_index : start_index + amount_of_mordred_features_per_sample ]
        temp_df = pd.concat([rdkit_features_columns, mordred_features_columns], axis=1)

        Corr_Matrix = temp_df.corr()
        # print(Corr_Matrix)
        corr_values = Corr_Matrix.values

        for i in range(amount_of_rdkit_features + amount_of_mordred_features_per_sample):
            row_index = i
            if i >= 14:
                row_index = i + start_index
            if flds[row_index] in features_to_remove:
                continue
            
            for j in range(i+1, amount_of_rdkit_features + amount_of_mordred_features_per_sample):
                column_index = j
                if j >= 14:
                    column_index = j + start_index
                print(column_index)
                if flds[column_index] in features_to_remove:
                    continue
                if corr_values[i,j] > 0.7:
                    
                    # print(flds[row_index], ' ', flds[column_index], ' ', corr_values[i,j])
                    features_to_remove.add(flds[column_index])

        # break
    print(len(features_to_remove))


def calculate_correlation_simple(df):
    df['cis/trans'] = df['cis/trans'].astype('category').cat.codes

    features_to_remove = set()

    normal_distributions_features, not_normal_distributions_features = split_features_by_normalization(df)
    df_distributions_normal = df[normal_distributions_features]
    df_distributions_not_normal = df[not_normal_distributions_features]

    corr_matrix_normal = df_distributions_normal.corr(method='pearson')
    corr_matrix_not_normal = df_distributions_not_normal.corr(method='spearman')

    features_to_remove_from_normal_distribution = get_most_correlated_values(corr_matrix=corr_matrix_normal,
                                                                             threshold=0.7,
                                                                             threshold_for_mandatory=0.9)
    features_to_remove_from_not_normal_distribution = get_most_correlated_values(corr_matrix=corr_matrix_not_normal,
                                                                                 threshold=0.7,
                                                                                 threshold_for_mandatory=0.9)
    
    features_to_remove = features_to_remove_from_normal_distribution.union(features_to_remove_from_not_normal_distribution)

    print("remains features:", len(df.keys()) - len(features_to_remove))
    print(features_to_remove)
    
    remained_features = list(set(df.keys()).difference(features_to_remove))
    remained_features_df = df[remained_features].copy()

    return remained_features_df


def detect_and_remove_outliers(features_df, target_df):
    outliers_names = []
    outlier_indexes_set = set()
    # feature outliers
    features_list = features_df.keys()
    for feature_name in features_list:
        feature_data = features_df[feature_name]
        outlier_indexes = detect_outlier_indexes(feature_data, threshold=3)
        print(feature_name, outlier_indexes)

        if len(outlier_indexes.flatten()) > 0:
            outliers_names.append(feature_name)

            if feature_name in mandatory_features:
                print(f"Mandatory feature: \033[1m{feature_name}\033[0m was deleted due to outliers, amount of outliers: {len(outlier_indexes.flatten())}")

        # outlier_indexes_set.update(set(outlier_indexes.flatten()))

    # target outliers
    target_list = target_df.keys()
    for target_name in target_list:
        target_data = target_df[target_name]
        outlier_indexes = detect_outlier_indexes(target_data, threshold=3)

        outlier_indexes_set.update(set(outlier_indexes.flatten()))

    print(len(features_df), len(target_df))

    remained_features_without_outliers = list(set(features_list).difference(set(outliers_names)))

    features_df = features_df.drop(list(outlier_indexes_set), axis=0)
    target_df = target_df.drop(list(outlier_indexes_set), axis=0)

    features_df = features_df[remained_features_without_outliers].copy()
    
    print(len(features_df), len(target_df))
    result_df = pd.concat([features_df, target_df], axis=1)

    return result_df


if __name__ == '__main__':
    excel_file_path = r'ml_part\molecule_features\pKa_Prediction_Starting data_2023.11.22.xlsx'
    csv_features_file_to_save = r'data\updated_features\remained_features.csv'
    df = pd.read_excel(excel_file_path, sheet_name="Main_List")

    smiles_to_features_index = {}
    all_smiles_features = []
    all_rdkit_features, all_mordred_features, all_target_data = [], [], []

    all_features = []
    
    features_index = 0

    for value, row in tqdm(df.iterrows()):
        
        if value > 0:
            features = obtain_features(row)
            target_data = obtain_y(row)

            all_smiles_features.append(features)
            all_target_data.append(target_data)

            # print(features)
            
            smiles_to_features_index[row['Smiles']] = features_index
            features_index += 1
        # if value > 3:
        #     break

    features_df = pd.DataFrame(all_smiles_features)
    remained_features_df = remove_features_with_same_values(features_df)
    remained_corr_features_df = calculate_correlation_simple(remained_features_df)
    target_df = pd.DataFrame(all_target_data)

    for remained_corr_feature_df in remained_corr_features_df.keys():
        if "pka" in remained_corr_feature_df.lower():
            print(remained_corr_feature_df)

    result_df = pd.concat([remained_corr_features_df, target_df], axis=1)
    result_df.to_csv(csv_features_file_to_save)

    # save smiles as pickle file
    with open(r"data\updated_features\smiles_to_index.pkl", 'wb') as handle:
        pickle.dump(smiles_to_features_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # result_df = detect_and_remove_outliers(remained_corr_features_df, target_df)

    # result_df = normalize_values(result_df, normalize_pKa=True, normalize_logP=True)

    # result_df.to_csv(csv_features_file_to_save)

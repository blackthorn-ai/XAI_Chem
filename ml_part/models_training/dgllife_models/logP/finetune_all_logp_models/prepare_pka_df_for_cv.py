import pandas as pd

df = pd.read_csv(r'data\init_data\pKa_Prediction_Starting data_2024.01.25.csv', index_col=0)
index_to_smiles = {index : row['Smiles'] for index, row in df.iterrows()}

train_folds_df = pd.read_csv(r'data\updated_features\csv_for_rulefit\train_pKa_v4_features_2.2.csv', index_col=0)
test_folds_df = pd.read_csv(r'data\updated_features\csv_for_rulefit\test_pKa_v4_features_2.2.csv', index_col=0)

gnn_train_df = train_folds_df[['fold_id', 'pKa']]
gnn_test_df = test_folds_df[['pKa']]

# print(gnn_test_df)

# smiles = []
# for index, row in gnn_test_df.iterrows():
#     print(df.at[index + 1, 'Smiles'], row['pKa'])
#     smiles.append(df.at[index + 1, 'Smiles'])

# gnn_test_df['Smiles_1'] = smiles

smiles, mol_type = [], []
for index, row in gnn_test_df.iterrows():
    print(index_to_smiles[index+1], index)
    smiles.append(index_to_smiles[index+1])
    mol_type.append(df.at[index + 1, 'identificator'])

gnn_test_df['Smiles'] = smiles
gnn_test_df['identificator'] = mol_type

# gnn_test_df.to_csv(r'data\pKa_basicity_data\gnn_cv\test_basic.csv')
print(gnn_test_df)

smiles, mol_type = [], []
for index, row in gnn_train_df.iterrows():
    print(index_to_smiles[index+1], index)
    smiles.append(index_to_smiles[index+1])
    mol_type.append(df.at[index + 1, 'identificator'])

gnn_train_df['Smiles'] = smiles
gnn_train_df['identificator'] = mol_type

# gnn_train_df.to_csv(r'data\pKa_basicity_data\gnn_cv\train_basic.csv')
print(gnn_train_df)

indexes = []
for index, row in gnn_train_df.iterrows():
    if "amine" not in row['identificator']:
        indexes.append(index)
gnn_train_df.loc[indexes].to_csv(r'data\pKa_basicity_data\gnn_cv\train_acid.csv')

indexes = []
for index, row in gnn_test_df.iterrows():
    if "amine" not in row['identificator']:
        indexes.append(index)
gnn_test_df.loc[indexes].to_csv(r'data\pKa_basicity_data\gnn_cv\test_acid.csv')

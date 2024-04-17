import pandas as pd

df = pd.read_csv(r'data\init_data\pKa_Prediction_Starting data_2024.01.25.csv', index_col=0)

train_csv = r'data\logP_lipophilicity_data\train.csv'
test_csv = r'data\logP_lipophilicity_data\test.csv'

train_df = pd.read_csv(train_csv, index_col=0)
test_df = pd.read_csv(test_csv, index_col=0)

train_folds_df = pd.read_csv(r'data\updated_features\csv_for_rulefit\train_logP_v4_features_2.2.csv', index_col=0)
test_folds_df = pd.read_csv(r'data\updated_features\csv_for_rulefit\test_logP_v4_features_2.2.csv', index_col=0)

gnn_train_df = train_folds_df[['fold_id', 'logP']]
gnn_test_df = test_folds_df[['logP']]

smiles = []

for index, row in gnn_test_df.iterrows():
    print(df.at[index + 1, 'Amides for LogP'], row['logP'])
    smiles.append(df.at[index + 1, 'Amides for LogP'])

gnn_test_df['Smiles'] = smiles

gnn_test_df.to_csv(r'data\logP_lipophilicity_data\gnn_cv\test.csv')
print(gnn_test_df)

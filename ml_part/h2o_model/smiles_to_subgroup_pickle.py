import pandas as pd
import pickle

EXCEL_FILEPATH = r'C:\work\DrugDiscovery\main_git\XAI_Chem\ml_part\molecule_features\pKa_Predicition_Starting data_2024.01.19.xlsx'

df = pd.read_excel(EXCEL_FILEPATH, sheet_name='Main_List')

amount_of_subgroups = {}
smiles_to_subgroup = {}
for index, row in df.iterrows():
    if pd.isnull(row['Smiles']): continue
    # print(row['Smiles'], row['Unnamed: 21'])
    if row['Unnamed: 21'] not in amount_of_subgroups:
        amount_of_subgroups[row['Unnamed: 21']] = 0
    amount_of_subgroups[row['Unnamed: 21']] += 1
    smiles_to_subgroup[row['Smiles']] = row['Unnamed: 21']

savepath_pickle_file = r'C:\work\DrugDiscovery\main_git\XAI_Chem\data\updated_features\smiles_to_subgroup.pkl'
# with open(savepath_pickle_file, 'wb') as handle:
#     pickle.dump(smiles_to_subgroup, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open(savepath_pickle_file, 'rb') as handle:
#     b = pickle.load(handle)

print(amount_of_subgroups)
import pandas as pd
from rdkit import Chem

EXCEL_PATH = r'C:\work\DrugDiscovery\main_git\XAI_Chem\ml_part\molecule_features\pKa_Prediction_Starting data_2023.11.22.xlsx'
EXCEL_PATH_UPDATED = r'C:\work\DrugDiscovery\main_git\XAI_Chem\ml_part\molecule_features\pKa_Predicition_Starting data_2024.01.19.xlsx'
PATH_TO_SAVE_CSV = r'C:\work\DrugDiscovery\main_git\XAI_Chem\data\init_data\pKa_Prediction_Starting data_2024.01.25.csv'

df = pd.read_excel(EXCEL_PATH, sheet_name="Main_List")
df_updated = pd.read_excel(EXCEL_PATH_UPDATED, sheet_name="Main_List")

identificators = []
index_to_drop = []
for index, row in df.iterrows():
    if pd.isnull(row['Smiles']):
        index_to_drop.append(index)
        continue
    smiles = row['Smiles']
    
    identificator = df_updated.loc[df_updated.Smiles == smiles, 'Unnamed: 21'].item()
    identificators.append(identificator)

df = df.drop(index_to_drop)
df['identificator'] = identificators

df.to_csv(PATH_TO_SAVE_CSV)

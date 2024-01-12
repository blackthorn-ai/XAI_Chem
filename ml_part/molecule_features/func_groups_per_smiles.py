from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import os
from rdkit import RDConfig
from rdkit.Chem import FragmentCatalog
import pandas as pd

fName=os.path.join(RDConfig.RDDataDir,'FunctionalGroups.txt')
amount_of_functional_groups = 39

fparams = FragmentCatalog.FragCatParams(1,6,fName)
fparams.GetNumFuncGroups()

excel_file_path = r'ml_part\molecule-features\pKa_Prediction_Starting data_2023.11.22.xlsx'
df = pd.read_excel(excel_file_path, sheet_name="Main_List")

functional_groups = {}
for i in range(amount_of_functional_groups):
    functional_groups[fparams.GetFuncGroup(i).GetProp('_Name')] = 0

total = 0
for index, row in df.iterrows():
    smiles = row['Smiles']
    if not pd.notnull(smiles):
        continue
    molecule = Chem.MolFromSmiles(smiles)
    chem_matches = 0
    print(smiles)
    for func_froup_index in range(amount_of_functional_groups):
        pattern = fparams.GetFuncGroup(func_froup_index)
        matches = molecule.GetSubstructMatches(pattern)
        if len(matches) > 0:
            chem_matches += 1
            # print(smiles, matches, Chem.MolToSmiles(pattern))
            functional_groups[fparams.GetFuncGroup(func_froup_index).GetProp('_Name')] += 1
            print(func_froup_index, fparams.GetFuncGroup(func_froup_index).GetProp('_Name'))

    total += chem_matches

print("Groups per SMILES:", total / (len(df) - 1))
print(functional_groups)

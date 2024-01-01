import pandas as pd

csv_file_path = r'C:\work\DrugDiscovery\RT_LogP_with_pKa_model\RTlogD\T-data_predictions(chembl32_logD)_copy.csv'

df = pd.read_csv(csv_file_path)
print(len(df))

amount_of_F = 0
index_to_stop = 0
for _, row in df.iterrows():
    smile = row['smiles']
    print(row)
    if 'f' in smile.lower():
        amount_of_F += 1
        index_to_stop += 1
        if index_to_stop > 10:
            break
        print(smile)
    
    break

print(amount_of_F)
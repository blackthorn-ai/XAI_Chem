import pandas as pd

csv_features_file_to_save = r'data\updated_features\deleted_outliers_entire_features.csv'
df = pd.read_csv(csv_features_file_to_save, index_col=0)

amount_of_mor = 0
print(len(df.keys()))
for key_name in df.keys():
    if "mor" in key_name.lower():
        amount_of_mor += 1
    print(key_name)

print(df[['logP', 'logP_normalized']])

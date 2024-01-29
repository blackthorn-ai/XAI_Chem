import pandas as pd
import numpy as np
from mordred import Calculator, descriptors

from constants import mandatory_features

path = r'C:\work\DrugDiscovery\main_git\XAI_Chem\data\updated_features\remained_features_25.01.csv'

df = pd.read_csv(path, index_col=0)

features_2D = Calculator(descriptors, ignore_3D=True).descriptors
features_2D_names = []
for feature_name in features_2D:
    features_2D_names.append(str(feature_name))

amount_of_2D_features, amount_of_3D_features = 0, 0
features_2D_in_my_data, features_3D_in_my_data = [], []
for key_name in df.columns:
    if key_name in features_2D_names:
        features_2D_in_my_data.append(key_name)
        amount_of_2D_features += 1
    else:
        features_3D_in_my_data.append(key_name)
        amount_of_3D_features += 1

for feature_3D_in_my_data in features_3D_in_my_data:
    if feature_3D_in_my_data in mandatory_features:
        print("Mandatory feature: ", end="")
    if "mor" not in feature_3D_in_my_data.lower():
        print(feature_3D_in_my_data)

print(f"amount of 2D features: {amount_of_2D_features}, amount of 3d: {amount_of_3D_features}")
print(len(df.keys()))

# features_3d_df = df[features_3D_in_my_data]
# corr_matrix = features_3d_df.corr()

# corr_matrix_values = corr_matrix.values

# count = 0
# for row_index in range(len(corr_matrix_values)):
#     for column_index in range(row_index+1, len(corr_matrix_values)):
#         if abs(corr_matrix_values[row_index, column_index]) > 0.7:
#             count += 1
#             print(corr_matrix.keys()[column_index], corr_matrix.keys()[row_index], corr_matrix_values[row_index, column_index])

# print(count)

# corr_matrix_sorted = corr_matrix.keys().sort_values()
# print(corr_matrix_sorted)

# print(features_3d_df['Mor03'])
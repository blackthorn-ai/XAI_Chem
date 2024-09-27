from scipy import stats
import pandas as pd
import numpy as np

path = r'C:\work\DrugDiscovery\main_git\XAI_Chem\data\updated_features\all_features.csv'

df = pd.read_csv(path, index_col=0)

normal, not_normal = 0, 0
normal_features, not_normal_features = [], []
for feature_name in df.keys():

    if df[feature_name].dtype == object:
        continue

    if len(df[feature_name].unique()) < 5:
        continue

    features = df[feature_name].to_list()
    res_features = stats.shapiro(features)

    if res_features.pvalue < 0.05:
        not_normal += 1
        not_normal_features.append(feature_name)
    else:
        normal += 1
        normal_features.append(feature_name)

print(normal, not_normal)

df_normal = df[normal_features]
corr_matrix_normal = df_normal.corr(method='pearson')

df_not_normal = df[not_normal_features]
corr_matrix_not_normal = df_not_normal.corr(method='spearman')
corr_matrix_not_normal_values = corr_matrix_not_normal.values



# for row_index in range(len(corr_matrix_not_normal_values)):
    
#     for column_index in range(row_index+1, len(corr_matrix_not_normal_values)):
#         if abs(corr_matrix_not_normal_values[row_index, column_index]) > 0.7:
#             print(corr_matrix_not_normal.keys()[column_index], corr_matrix_not_normal.keys()[row_index], corr_matrix_not_normal_values[row_index, column_index])
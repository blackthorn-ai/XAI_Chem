import pandas as pd
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ml_part.random_forest.data_prep.preparation import DataPreparation
from ml_part.random_forest.train import RFTrain

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

CSV_PATH = r'C:\work\DrugDiscovery\main_git\XAI_Chem\data\updated_features\remained_features_25.01.csv'
smiles_filepath = r'data\updated_features\smiles_to_index.pkl'

dataPreparation = DataPreparation(CSV_PATH)

X, y = dataPreparation.prepare_data_for_RF(is_pKa=True,
                                           use_mandatory_features=True,
                                           is_remove_outliers=True,
                                           is_remove_nan=True,
                                           outliers_features_to_skip=['dipole_moment'])

features_to_drop = []
for feature_name in X.columns:
    if "angle" in feature_name.lower():
        features_to_drop.append(feature_name)

X = X.drop(features_to_drop, axis=1)

rf_train = RFTrain(X=X, 
                   y=y,
                   smiles_filepath=smiles_filepath,
                   is_pKa=True,
                   k_folds=2)

y_train = rf_train.y_train
X_train = rf_train.X_train

y_test = rf_train.y_test
X_test = rf_train.X_test

print(X_train)

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# print(train_df['fold_id'])

train_df.to_csv(r'C:\work\DrugDiscovery\main_git\XAI_Chem\data\H2O_AutoML_cv2\Only_mol_with_angles_without_outliers(except_dipole)\train_pKa_data.csv')
test_df.to_csv(r'C:\work\DrugDiscovery\main_git\XAI_Chem\data\H2O_AutoML_cv2\Only_mol_with_angles_without_outliers(except_dipole)\test_pKa_data.csv')

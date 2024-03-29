import pandas as pd
import numpy as np
from scipy import stats
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from ml_part.molecule_features.constants import mandatory_features, identificator_to_molecule_type
from ml_part.random_forest.data_prep.utils import detect_outliers_iqr, detect_outliers_zscore, literal_return


class DataPreparation:
    def __init__(self, path_to_data: str, smiles_to_subgroup_pickle_file: str=None, smiles_to_index_pickle_file: str=None):
        np.random.seed(0)

        self.smiles_to_subgroup = None
        if smiles_to_subgroup_pickle_file is not None:
            with open(smiles_to_subgroup_pickle_file, 'rb') as handle:
                self.smiles_to_subgroup = pickle.load(handle)

        self.smiles_to_index = None
        if smiles_to_index_pickle_file is not None:
            with open(smiles_to_index_pickle_file, 'rb') as handle:
                self.smiles_to_index = pickle.load(handle)
        
        self.df = pd.read_csv(path_to_data, index_col=0)
        print("nF" in list(self.df.columns))
        self.targets = ['pKa', 'logP']

    def prepare_data_for_RF(self, 
                            is_pKa:bool=True, 
                            subgroup_type:str=None,
                            molecule_type:str=None,
                            use_mandatory_features:bool=True, 
                            is_remove_outliers:bool=True,
                            outliers_features_to_skip:list()=None,
                            is_remove_nan:bool=True):
        
        if subgroup_type is not None and self.smiles_to_subgroup is None:
            raise ValueError("DataPreparation needs pickle file, smiles_to_subgroup is None")
        
        if subgroup_type is not None and self.smiles_to_index is None:
            raise ValueError("DataPreparation needs pickle file, smiles_to_index is None")
        
        _df = self.df.copy()

        if molecule_type is not None:
            _df = DataPreparation.select_specific_molecule_type(_df=_df,
                                                                molecule_type_to_select=molecule_type)

        if subgroup_type is not None:
            _df = DataPreparation.select_specific_subgroup(_df=_df, 
                                                     subgroup_name=subgroup_type, 
                                                     smiles_to_subgroup=self.smiles_to_subgroup,
                                                     smiles_to_index=self.smiles_to_index)

        print(len(_df))
        X_y = self.prepare_data(_df, 
                                use_mandatory_features=use_mandatory_features,
                                is_remove_outliers=is_remove_outliers,
                                outliers_features_to_skip=outliers_features_to_skip)

        if is_remove_nan: X_y = X_y.dropna()
        print(f"Remains rows:{len(X_y)}, amount of features: {len(X_y.keys())}")

        y = X_y['logP']
        if is_pKa == True:
            y = X_y['pKa']

        X = X_y.copy().drop(self.targets, axis=1)

        return X, y


    @staticmethod
    def select_specific_molecule_type(_df:pd.DataFrame,
                                      molecule_type_to_select:str):
        subgroup_indexes_from_dataframe = []
        for index, row in _df.iterrows():
            molecule_type = identificator_to_molecule_type[row['identificator']]
            if molecule_type_to_select.lower() in molecule_type.lower():
                subgroup_indexes_from_dataframe.append(index)

        return _df.loc[subgroup_indexes_from_dataframe]


    @staticmethod
    def select_specific_subgroup(_df:pd.DataFrame,
                                 subgroup_name:str,
                                 smiles_to_subgroup:dict(),
                                 smiles_to_index:dict()):
        subgroup_indexes_from_dataframe = []
        for index, row in _df.iterrows():
            smiles = list(smiles_to_index.keys())[list(smiles_to_index.values()).index(index)]
            if smiles_to_subgroup[smiles].lower() == subgroup_name.lower():
                subgroup_indexes_from_dataframe.append(index)

        return _df.loc[subgroup_indexes_from_dataframe]


    @staticmethod
    def prepare_data(_df:pd.DataFrame, 
                       use_mandatory_features:bool=True,
                       is_remove_outliers:bool=True,
                       outliers_features_to_skip:list()=None):
        
        # x_df = _df.copy().drop(['pKa', 'logP'], axis=1)
        target_features = ['pKa', 'logP']
        _df_local = _df.copy()
        if use_mandatory_features:
            columns = []
            for feature_name in _df.keys():
                if feature_name in mandatory_features or feature_name in target_features or "dihedral_angle" in feature_name or "amount_of_" in feature_name:
                    columns.append(feature_name)
            
            _df_local = _df[columns].copy()
        
        print(columns)
        for feature_name in _df_local.keys():
            _df_local[feature_name] = literal_return(_df_local[feature_name])

        if is_remove_outliers:
            outlier_indexes_set = set()
            for feature_name in _df_local.keys():
                if feature_name in outliers_features_to_skip:
                    continue
                
                if _df_local[feature_name].dtype == object or len(_df_local[feature_name].unique()) < 10:
                    continue

                outlier_indexes = DataPreparation.detect_outliers(features=_df_local[feature_name], 
                                                                  threshold=3)
                
                if len(outlier_indexes) > 0:
                    print(f"{feature_name} outliers indexes: {outlier_indexes}")
                
                
                outlier_indexes_set.update(set(outlier_indexes))

            outlier_indexes_set.add(124)
            dataframe_indexes = [_df_local.index[index] for index in list(outlier_indexes_set)]
            # print(_df_local.index)
            _df_local = _df_local.drop(index=dataframe_indexes)
        
        _df_local = _df_local.dropna(subset=target_features)
        
        return _df_local


    @staticmethod
    def detect_outliers(features: list(),
                        threshold: int = 3):
        pvalue = stats.shapiro(features).pvalue
        # normal distribution
        if pvalue >= 0.05:
            outlier_indeces = detect_outliers_zscore(data=features,
                                                     threshold=threshold)
        # not normal ditribution
        else:
            outlier_indeces = detect_outliers_iqr(data=features)
        
        return outlier_indeces


    @staticmethod
    def normalize_data(data):
        original_values = np.array(data)
        
        ranked_values = stats.rankdata(original_values)
        quantile_normalized_values = stats.norm.ppf(ranked_values / (len(ranked_values) + 1))

        return quantile_normalized_values

    @staticmethod
    def unnormalize_data(normalized_value, 
                         original_values):
        rank = stats.norm.cdf(normalized_value) * (len(original_values) + 1)

        original_value = np.sort(original_values)[int(rank) - 1]

        return original_value
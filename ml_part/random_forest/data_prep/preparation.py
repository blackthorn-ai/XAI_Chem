import pandas as pd
import numpy as np
from scipy import stats
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from ml_part.molecule_features.constants import mandatory_features
from ml_part.random_forest.data_prep.utils import detect_outliers_iqr, detect_outliers_zscore, literal_return

class DataPreparation:
    def __init__(self, path_to_data: str):
        np.random.seed(0)
        self.df = pd.read_csv(path_to_data, index_col=0)
        print("nF" in list(self.df.columns))
        self.targets = ['pKa', 'logP']

    def prepare_data_for_RF(self, 
                            is_pKa:bool=True, 
                            use_mandatory_features:bool=True, 
                            is_remove_outliers:bool=True):
        
        _df = self.df.copy()
        X_y = self.prepare_data(_df, 
                              use_mandatory_features=use_mandatory_features,
                              is_remove_outliers=is_remove_outliers)

        y = X_y['logP']
        if is_pKa == True:
            y = X_y['pKa']

        X = X_y.copy().drop(self.targets, axis=1)

        return X, y

    @staticmethod
    def prepare_data(_df:pd.DataFrame, 
                       use_mandatory_features:bool=True,
                       is_remove_outliers:bool=True):
        
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
                if _df_local[feature_name].dtype == object or len(_df_local[feature_name].unique()) < 10:
                    continue

                outlier_indexes = DataPreparation.detect_outliers(features=_df_local[feature_name], 
                                                                  threshold=3)
                if "dihedral_angle_" in feature_name:
                    continue
                if len(outlier_indexes) > 0:
                    print(feature_name, outlier_indexes)
                
                
                outlier_indexes_set.update(set(outlier_indexes))

            _df_local = _df_local.drop(index=list(outlier_indexes_set))
        
        _df_local = _df_local.dropna(subset=target_features)
        print(f"Remains rows:{len(_df_local)}, amount of features: {len(_df_local.keys())}")
        return _df_local


    @staticmethod
    def detect_outliers(features:list(),
                        threshold:int=3):
        pvalue = stats.shapiro(features).pvalue
        # normal distribution
        if pvalue >= 0.05:
            outlier_indeces = detect_outliers_zscore(data=features,
                                                     threshold=3)
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
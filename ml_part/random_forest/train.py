from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from hyperopt import fmin, tpe, hp
from functools import partial
import pickle
import pandas as pd
import numpy as np

class RFTrain:
    def __init__(self, smiles_filepath, X, y, is_pKa=True, k_folds=None):
        with open(smiles_filepath, 'rb') as handle:
            self.smiles_to_index = pickle.load(handle)

        np.random.seed(42)

        self.type = "pKa" if is_pKa else "logP"
        self.smiles_column_name = "smiles"
        self.X = X
        self.y = y

        self.train_data_paths = [r'C:\work\DrugDiscovery\main_git\XAI_Chem\data\FGroupData\train_logP_data.csv', 
                                r'C:\work\DrugDiscovery\main_git\XAI_Chem\data\FGroupData\train_pKa_acid_data.csv', 
                                r'C:\work\DrugDiscovery\main_git\XAI_Chem\data\FGroupData\train_pKa_amine_data.csv']
        self.test_data_paths = [r'C:\work\DrugDiscovery\main_git\XAI_Chem\data\FGroupData\test_logP_data.csv', 
                               r'C:\work\DrugDiscovery\main_git\XAI_Chem\data\FGroupData\test_pKa_acid_data.csv', 
                               r'C:\work\DrugDiscovery\main_git\XAI_Chem\data\FGroupData\test_pKa_amine_data.csv']

        self.X_train, self.X_test, self.y_train, self.y_test = self.split_train_test(k_folds=k_folds)
        self.X_train.fillna(0, inplace=True)
        self.X_test.fillna(0, inplace=True)
        self.y_train.fillna(0, inplace=True)
        self.y_test.fillna(0, inplace=True)
        # self.split_train_test()
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.space = {
            'n_estimators': hp.choice('n_estimators', range(10, 2000)),
            'max_depth': hp.choice('max_depth', range(1, 20)),
            'min_samples_split': hp.choice('min_samples_split', range(2, 100)),
            'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 100)),
            'max_features': hp.choice('max_features', range(1, 150))
        }


    def split_train_test(self, k_folds=None):
        test_indexes = []
        for test_data_path in self.test_data_paths:
            if self.type not in test_data_path:
                continue
            test_df = pd.read_csv(test_data_path)
            for smiles in test_df[self.smiles_column_name]:
                if self.smiles_to_index[smiles] in self.X.index:
                    test_indexes.append(self.smiles_to_index[smiles])

        train_indexes = []
        for index in self.X.index:
            if index not in test_indexes:
                train_indexes.append(index)

        X_train = self.X.loc[train_indexes]
        y_train = self.y.loc[train_indexes]

        X_test = self.X.loc[test_indexes]
        y_test = self.y.loc[test_indexes]

        if k_folds is not None:
            q_amount = 10
            X_train['bin'] = pd.qcut(y_train, q=q_amount, labels=False)

            X_train['fold_id'] = [0] * len(X_train)

            for q in range(q_amount):
                train_indexes = X_train.loc[X_train['bin'] == q, 'fold_id'].index

                fold_id_array = np.array([0, 1] * (len(train_indexes) // 2))
                if len(train_indexes) % 2:
                    fold_id_array = np.append(fold_id_array, np.random.randint(0, 2))

                np.random.shuffle(fold_id_array)

                for train_index in range(len(train_indexes)):
                    X_train.at[train_indexes[train_index], 'fold_id'] = fold_id_array[train_index]

            X_train = X_train.drop(['bin'], axis=1)

        return X_train, X_test, y_train, y_test


    def train(self, 
              max_depth:int=10, 
              max_features:int=77, 
              min_samples_leaf:int=1, 
              min_samples_split:int=9, 
              n_estimators:int=262):
        model = RandomForestRegressor(n_estimators=n_estimators, 
                                      max_depth=max_depth,
                                      max_features=max_features,
                                      min_samples_leaf=min_samples_leaf,
                                      min_samples_split=min_samples_split,
                                      random_state=42)

        self.X_train.drop(columns=['fold_id'])
        self.y_train.drop(columns=['fold_id'])
        model.fit(self.X_train.drop(columns=['fold_id']), self.y_train.drop(columns=['fold_id']))

        y_pred = model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        print("Mean Squared Error:", mse)

        return model
    

    @staticmethod
    def objective(params, X_train, y_train):
        model = RandomForestRegressor(**params)
        
        # cv_indices_dict = {0: [], 1: []}
        # index = 0
        # for _, row in X_train.iterrows():
        #     cv_indices_dict[row['fold_id']].append(index)
        #     index += 1
        # cv_indices = [[cv_indices_dict[0], cv_indices_dict[1]], [cv_indices_dict[1], cv_indices_dict[0]]]

        X_train.drop(columns=['fold_id'])
        # y_train.drop(columns=['fold_id'])

        score = cross_val_score(model, X_train, y_train, cv=2, scoring='neg_mean_squared_error').mean()
        return -score


    def find_best_params_with_hyperopt(self):
        algo = tpe.suggest

        objective_partial = partial(RFTrain.objective, X_train=self.X_train, y_train=self.y_train)

        best_hyperparams = fmin(fn=objective_partial, space=self.space, algo=algo, max_evals=200, verbose=1)

        print("Найкращі гіперпараметри:", best_hyperparams)
        return best_hyperparams

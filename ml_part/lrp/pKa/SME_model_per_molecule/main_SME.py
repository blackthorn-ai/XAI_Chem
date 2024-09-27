import os

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

# from lrp_explainer import PkaLRP
from lrp_explainer_SME_correct import PkaLRP
from pKa_model_service import PkaModelService
from constants import Identificator
from utils import prepare_data

if __name__ == '__main__':

    SMILES_to_fgroup, SMILES_to_identificator, SMILES_to_cycle_type = prepare_data()

    df_train_acid = pd.read_csv(r'data\pKa_basicity_data\gnn_cv_canon_smiles\train_acid.csv', index_col=0)
    df_train_basic = pd.read_csv(r'data\pKa_basicity_data\gnn_cv_canon_smiles\train_basic.csv', index_col=0)
    df_train = pd.concat([df_train_acid, df_train_basic], axis=0)

    df_test_acid = pd.read_csv(r'data\pKa_basicity_data\gnn_cv_canon_smiles\test_acid.csv', index_col=0)
    df_test_basic = pd.read_csv(r'data\pKa_basicity_data\gnn_cv_canon_smiles\test_basic.csv', index_col=0)
    df_test = pd.concat([df_test_acid, df_test_basic], axis=0)
    
    df = pd.concat([df_train, df_test], axis=0)

    df_for_train = pd.read_csv(r'data\pKa_basicity_data\gnn_cv_canon_smiles\train_test_for_SMILES_per_model.csv')

    molecules_relevance_fluorine_group = {}

    amount_of_predictions = 0
    for index_train, row_train in tqdm(df.iterrows()):
        SMILES = row_train['Smiles']
        canon_SMILES = Chem.CanonSmiles(SMILES)

        identificator = SMILES_to_identificator[SMILES]
        fluorine_group = SMILES_to_fgroup[SMILES]
        cycle_type = SMILES_to_cycle_type[SMILES]

        if "secon" in identificator.lower():
            identificator = Identificator.secondary_amine
        elif "primary" in identificator.lower():
            identificator = Identificator.primary_amine
        elif "acid" in identificator.lower():
            identificator = Identificator.carboxilic_acid

        is_exist_in_train = False
        smiles_non_f = None
        for index, row in df_for_train.iterrows():
            if row['Smiles'] == canon_SMILES:
                is_exist_in_train = True
                smiles_non_f = row['non-F Smiles']
                break
        
        model_name = f"pKa_{index_train}_{canon_SMILES}_{smiles_non_f}.pkl"
        path_to_models = r'ml_part\weights\pKa\separate_model_for_each_molecule'
        model_path = rf'{path_to_models}\{model_name}'
        
        # models_from_folder = os.listdir(path_to_models)
        # print(model_name, model_name in models_from_folder)
        amount_of_predictions += 1
        

        model_service = PkaModelService(identificator=identificator,
                                        is_combined_model=False,
                                        amine_model_weights_path=model_path,
                                        acid_model_weights_path=model_path)

        pka_lrp = PkaLRP(
            model_service=model_service,
            SMILES=SMILES,
            identificator=identificator,
            fluorine_group=fluorine_group,
            is_centrize_relevances=False
        )

        molecules_relevance_fluorine_group[SMILES] = pka_lrp.importance_fluorine_group
        # print(logp_lrp.node_relevances)
        # break

    print("=" * 30)
    print("Fluorine group with C average")
    print(molecules_relevance_fluorine_group)
    print("=" * 30)
    print(f"Number of mols: {amount_of_predictions}")

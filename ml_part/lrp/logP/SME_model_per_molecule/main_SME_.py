import os

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

# from lrp_explainer import PkaLRP
from lrp_explainer_SME_correct import LogpLRP
from logP_model_service import LogPModelService
from utils import prepare_data

if __name__ == '__main__':

    SMILES_to_fgroup, SMILES_to_identificator, SMILES_to_cycle_type = prepare_data()

    df_train = pd.read_csv(r'data\logP_lipophilicity_data\gnn_cv\train.csv', index_col=0)
    df_test = pd.read_csv(r'data\logP_lipophilicity_data\gnn_cv\test.csv', index_col=0)
    df = pd.concat([df_train, df_test], axis=0)

    df_for_train = pd.read_csv(r'data\logP_lipophilicity_data\gnn_cv\train_test_for_SMILES_per_model.csv')

    molecules_relevance_fluorine_group = {}

    amount_of_predictions = 0
    for index_train, row_train in tqdm(df.iterrows()):
        SMILES = row_train['Smiles']
        canon_SMILES = Chem.CanonSmiles(SMILES)

        identificator = SMILES_to_identificator[SMILES]
        fluorine_group = SMILES_to_fgroup[SMILES]
        cycle_type = SMILES_to_cycle_type[SMILES]

        is_exist_in_train = False
        smiles_non_f = None
        for index, row in df_for_train.iterrows():
            if row['Smiles'] == canon_SMILES:
                is_exist_in_train = True
                smiles_non_f = row['non-F Smiles']
                break
        
        # model_path = r'ml_part\weights\logP\3_models_cv_oos\GNN_attentivefp_model_logP_best_loss.pth'
        # if row['fold_id'] == 0:
        #     model_path = r'ml_part\weights\logP\3_models_cv_oos\cv_1_GCN_attentivefp_Lipophilicity.pth'
        # if row['fold_id'] == 1:
        #     model_path = r'ml_part\weights\logP\3_models_cv_oos\cv_0_GCN_attentivefp_Lipophilicity.pth'

        # print(is_exist_in_train)
        model_name = f"{index_train}_{canon_SMILES}_{smiles_non_f}_logP_best_loss.pth"
        path_to_models = r'ml_part\weights\logP\separate_model_for_each_molecule'
        model_path = rf'{path_to_models}\{model_name}'
        
        models_from_folder = os.listdir(path_to_models)
        # print(model_name, model_name in models_from_folder)
        amount_of_predictions += 1
        

        model_service = LogPModelService(
            model_name="gcn",
            model_weights_path=model_path)

        logP_lrp = LogpLRP(
            model_service=model_service,
            SMILES=SMILES,
            identificator=identificator,
            fluorine_group=fluorine_group,
            is_centrize_relevances=False
        )

        molecules_relevance_fluorine_group[SMILES] = logP_lrp.importance_fluorine_group
        # print(logP_lrp.importance_fluorine_group)
        # break

    print("=" * 30)
    print("Fluorine group with C average")
    print(molecules_relevance_fluorine_group)
    print("=" * 30)
    print(f"Number of mols: {amount_of_predictions}")

import pandas as pd
from utils import prepare_data
from tqdm import tqdm

# from lrp_explainer import PkaLRP
from lrp_explainer_SME_correct import PkaLRP
from pKa_model_service import PkaModelService
from constants import Identificator

if __name__ == '__main__':

    SMILES_to_fgroup, SMILES_to_identificator, SMILES_to_cycle_type = prepare_data()

    df_train = pd.read_csv(r'data\pKa_basicity_data\gnn_cv_canon_smiles\train_basic.csv', index_col=0)
    df_test = pd.read_csv(r'data\pKa_basicity_data\gnn_cv_canon_smiles\test_basic.csv', index_col=0)
    df = pd.concat([df_train, df_test], axis=0)

    # df = pd.read_csv(r'data\init_data\pKa_Prediction_Starting data_2024.05.07.csv')

    molecules_relevance_fluorine_group = {}

    for index, row in tqdm(df.iterrows()):
        SMILES = row['Smiles']
        # SMILES = 'FC(F)(F)C1CCN1'
        # print(SMILES)
        identificator = SMILES_to_identificator[SMILES]
        fluorine_group = SMILES_to_fgroup[SMILES]
        cycle_type = SMILES_to_cycle_type[SMILES]

        if "amine" not in identificator.lower():
            continue
        if "secon" not in identificator.lower() and cycle_type != "methane":
            continue

        if "secon" in identificator.lower():
            identificator = Identificator.secondary_amine
        elif "primary" in identificator.lower():
            identificator = Identificator.primary_amine
        elif "acid" in identificator.lower():
            identificator = Identificator.carboxilic_acid

        amine_model_path = r'ml_part\weights\pKa\combined_dataset\acid_best_loss_daily-morning-84.pkl'
        if row['fold_id'] == 0:
            amine_model_path = r'ml_part\weights\pKa\combined_dataset\cv_models\acidic\cv_1_best_loss_lr_0.0007452146740113421_wd_0.0025791056042483687_train_type_predictor_and_readout.pkl'
        if row['fold_id'] == 1:
            amine_model_path = r'ml_part\weights\pKa\combined_dataset\cv_models\acidic\cv_0_best_loss_lr_0.0007452146740113421_wd_0.0025791056042483687_train_type_predictor_and_readout.pkl'

        # print(row['fold_id'], amine_model_path)
        model_service = PkaModelService(identificator=identificator,
                                        is_combined_model=False,
                                        amine_model_weights_path=amine_model_path)

        logp_lrp = PkaLRP(
            model_service=model_service,
            SMILES=SMILES,
            identificator=identificator,
            fluorine_group=fluorine_group,
            is_centrize_relevances=False
        )

        molecules_relevance_fluorine_group[SMILES] = logp_lrp.importance_fluorine_group
        # print(logp_lrp.node_relevances)
        # break

    print("=" * 20)
    print("Fluorine group with C average")
    print(molecules_relevance_fluorine_group)

import os

import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from utils import prepare_data

# from lrp_explainer import LogpLRP
from lrp_explainer_sme import LogpLRP
from logP_model_service import LogPModelService

if __name__ == '__main__':

    SMILES_to_fgroup, SMILES_to_identificator, SMILES_to_cycle_type = prepare_data()

    df_train = pd.read_csv(r'data\logP_lipophilicity_data\gnn_cv\train.csv', index_col=0)
    df_test = pd.read_csv(r'data\logP_lipophilicity_data\gnn_cv\test.csv', index_col=0)
    df = pd.concat([df_train, df_test], axis=0)

    # df = pd.read_csv(r'data\init_data\pKa_Prediction_Starting data_2024.05.07.csv')

    df_for_train = pd.read_csv(r'data\logP_lipophilicity_data\gnn_cv\train_test_for_SMILES_per_model.csv')

    molecules_fluorine_derivatives_relevance = {}
    molecules_only_fluor_derivatives = {}
    molecules_atom_in_cycle_and_edge_to_fluor_relevance = {}
    molecules_relevance_fluorine_group = {}

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

        if "prim" not in identificator.lower():
            continue

        # model_path = r'ml_part\weights\logP\3_models_cv_oos\GNN_attentivefp_model_logP_best_loss.pth'
        # if row['fold_id'] == 0:
        #     model_path = r'ml_part\weights\logP\3_models_cv_oos\cv_1_GCN_attentivefp_Lipophilicity.pth'
        # if row['fold_id'] == 1:
        #     model_path = r'ml_part\weights\logP\3_models_cv_oos\cv_0_GCN_attentivefp_Lipophilicity.pth'

        model_name = f"{index_train}_{canon_SMILES}_{smiles_non_f}_logP_best_loss.pth"
        path_to_models = r'ml_part\weights\logP\separate_model_for_each_molecule'
        model_path = rf'{path_to_models}\{model_name}'
        
        models_from_folder = os.listdir(path_to_models)
    
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

        output_svg_path = rf'data\lrp_results\logp\secondary_amines\separate_model_for_each_molecule_for_oos(secondary,primary,acid)\logP_{SMILES}.svg'
        output_png_path = rf'data\lrp_results\logp\secondary_amines\separate_model_for_each_molecule_for_oos(secondary,primary,acid)\logP_{SMILES}.png'
        logP_lrp.save_molecule_with_relevances(output_svg_path=output_svg_path,
                                               output_png_path=output_png_path)
        
        # molecules_fluorine_derivatives_relevance[SMILES] = logp_lrp.relevance_entire_derivatives
        # molecules_only_fluor_derivatives[SMILES] = logp_lrp.relevance_only_fluorine
        # molecules_atom_in_cycle_and_edge_to_fluor_relevance[SMILES] = logp_lrp.relevance_fluorine_derivative_atom_in_cycle_and_edge_to_fluor
        molecules_relevance_fluorine_group[SMILES] = logP_lrp.relevance_fluorine_group
        # print(logp_lrp.node_relevances)

        # break

    print("Entire fluorine derivatives")
    print(molecules_fluorine_derivatives_relevance)
    print("=" * 20)
    print("Only fluorine derivatives")
    print(molecules_only_fluor_derivatives)
    print("=" * 20)
    print("Atom and edge to fluorine")
    print(molecules_atom_in_cycle_and_edge_to_fluor_relevance)
    print("=" * 20)
    print("All fluorine subgroup")
    print(molecules_relevance_fluorine_group)
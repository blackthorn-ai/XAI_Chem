import pandas as pd
from utils import prepare_data
from tqdm import tqdm

from lrp_explainer import PkaLRP
# from lrp_explainer_sme import PkaLRP
from pKa_model_service import PkaModelService
from constants import Identificator

if __name__ == '__main__':

    SMILES_to_fgroup, SMILES_to_identificator, SMILES_to_cycle_type = prepare_data()

    df_train = pd.read_csv(r'data\pKa_basicity_data\gnn_cv_canon_smiles\train_basic.csv', index_col=0)
    df_test = pd.read_csv(r'data\pKa_basicity_data\gnn_cv_canon_smiles\test_basic.csv', index_col=0)
    df = pd.concat([df_train, df_test], axis=0)

    # df = pd.read_csv(r'data\init_data\pKa_Prediction_Starting data_2024.05.07.csv')

    molecules_fluorine_derivatives_relevance = {}
    molecules_only_fluor_derivatives = {}
    molecules_atom_in_cycle_and_edge_to_fluor_relevance = {}
    molecules_two_atoms_from_derivatives_and_edge_to_fluor = {}
    molecules_relevance_fluorine_group = {}

    for index, row in tqdm(df.iterrows()):
        SMILES = row['Smiles']
        # SMILES = 'FC(F)(F)C1CCN1'
        # print(SMILES)
        identificator = SMILES_to_identificator[SMILES]
        fluorine_group = SMILES_to_fgroup[SMILES]
        cycle_type = SMILES_to_cycle_type[SMILES]

        # for debugging
        # if "[H]C(F)(F)C1CCNCC1" != SMILES:
        #     continue
        # if cycle_type != "methane":
        #     continue
        # print(SMILES, identificator)
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

        amine_model_path = r'ml_part\weights\pKa\overfit_amine_best_loss_logical-bee-26.pkl'
        if row['fold_id'] == 0:
            amine_model_path = r'ml_part\weights\pKa\overfit_amine_best_loss_logical-bee-26.pkl'
        if row['fold_id'] == 1:
            amine_model_path = r'ml_part\weights\pKa\overfit_amine_best_loss_logical-bee-26.pkl'

        # print(row['fold_id'], amine_model_path)
        model_service = PkaModelService(identificator=identificator,
                                        is_combined_model=False,
                                        amine_model_weights_path=amine_model_path)

        logp_lrp = PkaLRP(
            model_service=model_service,
            SMILES=SMILES,
            identificator=identificator,
            fluorine_group=fluorine_group,
            is_centrize_relevances=True
        )

        # output_svg_path = rf'data\lrp_results\pka\SME_by_groups_combined_dataset_splitted_by_cv_models_and_combined_model\pKa_{SMILES}.svg'
        # output_png_path = rf'data\lrp_results\pka\SME_by_groups_combined_dataset_splitted_by_cv_models_and_combined_model\pKa_{SMILES}.png'
        # logp_lrp.save_molecule_with_relevances(output_svg_path=output_svg_path,
        #                                        output_png_path=output_png_path)
        
        molecules_fluorine_derivatives_relevance[SMILES] = logp_lrp.relevance_entire_derivatives
        molecules_only_fluor_derivatives[SMILES] = logp_lrp.relevance_only_fluorine
        molecules_atom_in_cycle_and_edge_to_fluor_relevance[SMILES] = logp_lrp.relevance_fluorine_derivative_atom_in_cycle_and_edge_to_fluor
        molecules_two_atoms_from_derivatives_and_edge_to_fluor[SMILES] = logp_lrp.relevance_two_atoms_from_fluor_derivatives_and_edge
        molecules_relevance_fluorine_group[SMILES] = logp_lrp.relevance_fluorine_group
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
    print("Two Atoms and edge to fluorine")
    print(molecules_two_atoms_from_derivatives_and_edge_to_fluor)
    print("=" * 20)
    print("Fluorine group with C average")
    print(molecules_relevance_fluorine_group)

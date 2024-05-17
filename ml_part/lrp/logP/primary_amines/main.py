import pandas as pd
from utils import prepare_data

# from lrp_explainer import LogpLRP
from lrp_explainer_sme import LogpLRP

if __name__ == '__main__':

    SMILES_to_fgroup, SMILES_to_identificator, SMILES_to_cycle_type = prepare_data()

    # df_train = pd.read_csv(r'data\logP_lipophilicity_data\gnn_cv\train.csv', index_col=0)
    # df_test = pd.read_csv(r'data\logP_lipophilicity_data\gnn_cv\test.csv', index_col=0)
    # df = pd.concat([df_train, df_test], axis=0)

    df = pd.read_csv(r'data\init_data\pKa_Prediction_Starting data_2024.05.07.csv')

    molecules_fluorine_derivatives_relevance = {}
    molecules_only_fluor_derivatives = {}
    molecules_atom_in_cycle_and_edge_to_fluor_relevance = {}
    molecules_relevance_fluorine_group = {}

    for index, row in df.iterrows():
        SMILES = row['Amides for LogP']
        # SMILES = 'O=C(c1ccccc1)N1CCCC1C(F)(F)F'
        identificator = SMILES_to_identificator[SMILES]
        fluorine_group = SMILES_to_fgroup[SMILES]
        cycle_type = SMILES_to_cycle_type[SMILES]

        if ("primary" not in identificator.lower() and cycle_type != "methane") or "acid" in identificator.lower():
            continue
    
        logp_lrp = LogpLRP(
            SMILES=SMILES,
            identificator=identificator,
            fluorine_group=fluorine_group,
            is_centrize_relevances=False,
        )

        output_svg_path = rf'data\lrp_results\logp\primary_amines\SME_by_groups\logP_{SMILES}.svg'
        output_png_path = rf'data\lrp_results\logp\primary_amines\SME_by_groups\logP_{SMILES}.png'
        logp_lrp.save_molecule_with_relevances(output_svg_path=output_svg_path,
                                               output_png_path=output_png_path)
        
        # molecules_fluorine_derivatives_relevance[SMILES] = logp_lrp.relevance_entire_derivatives
        # molecules_only_fluor_derivatives[SMILES] = logp_lrp.relevance_only_fluorine
        # molecules_atom_in_cycle_and_edge_to_fluor_relevance[SMILES] = logp_lrp.relevance_fluorine_derivative_atom_in_cycle_and_edge_to_fluor
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
    print("All fluorine subgroup")
    print(molecules_relevance_fluorine_group)
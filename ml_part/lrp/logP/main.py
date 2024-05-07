import pandas as pd
from utils import prepare_data

from lrp_explainer import LogpLRP

if __name__ == '__main__':

    SMILES_to_fgroup, SMILES_to_identificator, SMILES_to_cycle_type = prepare_data()

    df_train = pd.read_csv(r'data\logP_lipophilicity_data\gnn_cv\train.csv', index_col=0)
    df_test = pd.read_csv(r'data\logP_lipophilicity_data\gnn_cv\test.csv', index_col=0)
    df = pd.concat([df_train, df_test], axis=0)

    molecules_fluorine_derivatives_relevance = {}
    molecules_only_fluor_derivatives = {}
    molecules_atom_in_cycle_and_edge_to_fluor_relevance = {}

    for index, row in df.iterrows():
        SMILES = row['Smiles']
        # SMILES = 'FC(F)(F)CNC(=O)C1=CC=CC=C1'
        identificator = SMILES_to_identificator[SMILES]
        fluorine_group = SMILES_to_fgroup[SMILES]
        cycle_type = SMILES_to_cycle_type[SMILES]

        if "secon" not in identificator.lower() and cycle_type != "methane":
            continue
    
        logp_lrp = LogpLRP(
            SMILES=SMILES,
            identificator=identificator,
            fluorine_group=fluorine_group
        )

        output_svg_path = rf'data\lrp_results\logp\train_scaled\logP_{SMILES}.svg'
        output_png_path = rf'data\lrp_results\logp\train_scaled\logP_{SMILES}.png'
        logp_lrp.save_molecule_with_relevances(output_svg_path=output_svg_path,
                                               output_png_path=output_png_path)
        
        molecules_fluorine_derivatives_relevance[SMILES] = logp_lrp.relevance_entire_derivatives
        molecules_only_fluor_derivatives[SMILES] = logp_lrp.relevance_only_fluorine
        molecules_atom_in_cycle_and_edge_to_fluor_relevance[SMILES] = logp_lrp.relevance_fluorine_derivative_atom_in_cycle_and_edge_to_fluor
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
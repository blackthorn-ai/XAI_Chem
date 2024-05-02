from enum import Enum

functional_group_to_smiles = {
    "CF3": "C(F)(F)F", 
    "CH2F": "CF", 
    "gem-CF2": "F", 
    "CHF2": "C(F)(F)",
    "CHF": "F",
    "non-F": ""
}

class Identificator(Enum):
    carboxilic_acid = 'carboxilic_acid'
    primary_amine = 'primary_amine'
    secondary_amine = 'secondary_amine'

class RelevanceMode(Enum):
    only_fluor = 0
    entire_derivatives_fluor_only_nodes = 1
    entire_derivatives_fluor_nodes_and_edges = 2
    fluorine_derivative_atom_in_cycle_and_edge_to_fluor = 3


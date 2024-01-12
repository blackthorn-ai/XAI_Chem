mandatory_features = ["cis/trans", 
                      "f_to_fg",
                      "f_atom_fraction",
                      "dipole_moment", 
                      "mol_volume", 
                      "mol_weight",
                      "sasa",  
                      "tpsa+f", 
                      "linear_distance", 
                      "f_freedom", 
                      "mol_num_cycles", 
                      "avg_atoms_in_cycle", 
                      "chirality",
                      "PNSA5",
                      "PPSA5",
                      "angle_between_functional_group_and_molecule",
                      
                      "nHRing", "naRing", "naHRing", "nARing", "nAHRing", "nFRing", "nFHRing", "nFaRing", "nFaHRing", "nFARing", "nFAHRing",

                      "nF", "nC", "nN", "nO",
                      
                    #   "Mor01", "Mor02", "Mor03", "Mor04", "Mor05", "Mor06", "Mor07", "Mor08", "Mor09", "Mor10",
                    #   "Mor11", "Mor12", "Mor13", "Mor14", "Mor15", "Mor16", "Mor17", "Mor18", "Mor19", "Mor20",
                    #   "Mor21", "Mor22", "Mor23", "Mor24", "Mor25", "Mor26", "Mor27", "Mor28", "Mor29", "Mor30",
                    #   "Mor31", "Mor32",
                      
                      "FPSA3", "PBF", "TASA", "RPCS", "GeomShapeIndex"]


functional_group_to_smiles = {
            "CF3": "CC(F)(F)F", 
            "CH2F": "CCF", 
            "gem-CF2": "CC(F)(F)", 
            "CHF2": "CC(F)(F)",
            "CHF": "CCF",
            "non-F": ""
        }
import json
import pickle
import os
import sys
sys.path.append("..") 
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from utils import setup, load_model
from model_load import load_pKa_acidic_model, load_pKa_basic_model, load_logP_model
from lrp import Lrp
from constants import ModelType

SMILES = "FC(F)(F)C1CNC1​"
functional_group = "CF3"
output_svg_path = r'data/output_EAI/output_logP.svg'

if __name__ == '__main__':
    with open(r'ml_part\configs\args.pickle', 'rb') as file:
        args =pickle.load(file)
        args['task'] = ['pKa']
    with open(r'ml_part\configs\configure.json', 'r') as f:
        exp_config = json.load(f)
    args['device'] = torch.device('cpu')
    args = setup(args)

    logP_model = load_logP_model(exp_config, args)
    amine_model = load_pKa_basic_model(args)
    acid_model = load_pKa_acidic_model(args)

    lrp_class = Lrp(logP_model=logP_model, 
                    amine_pKa_model=amine_model,
                    acid_pKa_model=acid_model,
                    smiles=SMILES, 
                    functional_group=functional_group, 
                    exp_config=exp_config, 
                    args=args,
                    model_type=ModelType.pKa_acid)
    
    lrp_class.save_molecule_with_relevances(output_svg_path)

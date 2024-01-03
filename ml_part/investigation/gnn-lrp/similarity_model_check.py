import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pickle
import json
from utils import setup, load_model
from My_Pka_Model import Pka_acidic_view,Pka_basic_view
import torch

def load_logP_model(exp_config, args, checkpoint_path):
    logP_model = load_model(exp_config).to(args['device'])
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))#my_model/CRT76-logD/model_pretrain_76.pth"
    logP_model.load_state_dict(checkpoint)

    return logP_model


def load_pKa_acidic_model(args, checkpoint_path):
    pka1_model = Pka_acidic_view(node_feat_size = 74,
                                 edge_feat_size = 12,
                                 output_size = 1,
                                 num_layers= 6,
                                 graph_feat_size=200,
                                 dropout=0).to(args['device'])
    pka1_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

    return pka1_model


def load_pKa_basic_model(args, checkpoint_path):
    pka2_model = Pka_basic_view(node_feat_size = 74,
                                edge_feat_size = 12,
                                output_size = 1,
                                num_layers= 6,
                                graph_feat_size=200,
                                dropout=0).to(args['device'])
    pka2_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    return pka2_model


def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            print('Mismtach not found at', key_item_1[0])
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')


logP_default_checkpoint_path = r'ml_part\weights\logP\model_pretrain_76_default_weights.pth'
logP_checkpoint_path = r'ml_part\weights\logP\logP_RTLogD_best_loss_comfy-wave-9.pth'

pKa_acid_default_checkpoint_path = r'ml_part\weights\pKa\site_acidic.pkl'
pKa_acid_checkpoint_path = r'ml_part\weights\pKa\site_acidic_best_loss_distinctive-butterfly-11.pkl'
if __name__ == "__main__":
    with open(r'ml_part\configs\args.pickle', 'rb') as file:
        args =pickle.load(file)
        args['task'] = ['pKa']
    with open(r'ml_part\configs\configure.json', 'r') as f:
        exp_config = json.load(f)
    args['device'] = torch.device('cpu')
    args = setup(args)

    logP_model_default = load_logP_model(exp_config, args, logP_default_checkpoint_path)
    logP_model_finetuned = load_logP_model(exp_config, args, logP_checkpoint_path)

    pKa_acidic_model_default = load_pKa_acidic_model(args, pKa_acid_default_checkpoint_path)
    pKa_acidic_model = load_pKa_acidic_model(args, pKa_acid_checkpoint_path)

    print(compare_models(pKa_acidic_model_default, pKa_acidic_model))


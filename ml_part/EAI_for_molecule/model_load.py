from utils import setup, load_model
from My_Pka_Model import Pka_acidic_view,Pka_basic_view
import torch

def load_logP_model(exp_config, args):
    logP_model = load_model(exp_config).to(args['device'])
    # checkpoint = torch.load(r"C:\work\DrugDiscovery\RT_LogP_with_pKa_model\RTlogD\final_model/RTlogD/model_pretrain_76.pth",map_location=torch.device('cpu'))#my_model/CRT76-logD/model_pretrain_76.pth"
    checkpoint = torch.load(r"ml_part\weights\logP\logP_RTLogD_best_loss_comfy-wave-9.pth",map_location=torch.device('cpu'))#my_model/CRT76-logD/model_pretrain_76.pth"
    logP_model.load_state_dict(checkpoint)

    return logP_model


def load_pKa_acidic_model(args):
    pka1_model = Pka_acidic_view(node_feat_size = 74,
                                 edge_feat_size = 12,
                                 output_size = 1,
                                 num_layers= 6,
                                 graph_feat_size=200,
                                 dropout=0).to(args['device'])
    # pka1_model.load_state_dict(torch.load(r'C:\work\DrugDiscovery\RT_LogP_with_pKa_model\RTlogD\Trained_model/site_acidic.pkl',map_location='cpu'))
    pka1_model.load_state_dict(torch.load(r'ml_part\weights\pKa\site_acidic_best_loss_distinctive-butterfly-11.pkl',map_location='cpu'))

    return pka1_model


def load_pKa_basic_model(args):
    pka2_model = Pka_basic_view(node_feat_size = 74,
                                edge_feat_size = 12,
                                output_size = 1,
                                num_layers= 6,
                                graph_feat_size=200,
                                dropout=0).to(args['device'])
    pka2_model.load_state_dict(torch.load(r'ml_part\weights\pKa\site_amine_best_loss_sweet-capybara-11.pkl',map_location='cpu'))
    return pka2_model

from functools import partial

import pickle
import pandas as pd
import torch
from torch.utils.data import DataLoader
from rdkit import Chem
from dgllife.data import MoleculeCSVDataset
from dgllife.model import load_pretrained
from dgllife.utils import smiles_to_bigraph
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from dgllife.utils import AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer
from dgllife.utils import PretrainAtomFeaturizer, PretrainBondFeaturizer
from dgllife.utils.mol_to_graph import SMILESToBigraph
from rdkit import Chem

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import os
import sys
sys.path.append("..") 
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from My_Pka_Model import Pka_acidic_view,Pka_basic_view
from utils import collate_molgraphs

def calculate_metrics(true_values, pred_values):
    mse = round(mean_squared_error(true_values, pred_values),3)
    mae = round(mean_absolute_error(true_values, pred_values),3)
    r_score = round(r2_score(true_values, pred_values),3)

    return {"mse": mse,
            "mae": mae,
            "r^2": r_score,}

def load_pKa_acidic_model():
    pka1_model = Pka_acidic_view(node_feat_size = 74,
                                 edge_feat_size = 12,
                                 output_size = 1,
                                 num_layers= 6,
                                 graph_feat_size=200,
                                 dropout=0).to('cpu')
    # pka1_model.load_state_dict(torch.load(r'ml_part\weights\pKa\site_acidic.pkl',map_location='cpu'))
    pka1_model.load_state_dict(torch.load(r'ml_part\weights\pKa\acid_best_loss_blooming-deluge-52.pkl',map_location='cpu'))
    return pka1_model


def load_pKa_basic_model():
    pka2_model = Pka_basic_view(node_feat_size = 74,
                                 edge_feat_size = 12,
                                 output_size = 1,
                                 num_layers= 6,
                                 graph_feat_size=200,
                                 dropout=0).to('cpu')
    # pka2_model.load_state_dict(torch.load(r'ml_part\weights\pKa\site_basic.pkl',map_location='cpu'))
    pka2_model.load_state_dict(torch.load(r'ml_part\weights\pKa\basic_best_loss_glamorous-totem-39.pkl',map_location='cpu'))
    return pka2_model

def prepare_graphs(smiles):
    
    smiles_to_graph = SMILESToBigraph(node_featurizer=CanonicalAtomFeaturizer(),
                                      edge_featurizer=CanonicalBondFeaturizer())

    return smiles_to_graph(smiles)

if __name__ == "__main__":
    device = 'cpu'

    csv_path = r'data\init_data\pKa_Prediction_Starting data_2024.01.25.csv'
    csv_pka_test_path = r'data\pKa_basicity_data\gnn_cv\test.csv'

    df = pd.read_csv(csv_path, index_col=0)
    df_test_splited_by_tanimoto = pd.read_csv(csv_pka_test_path, index_col=0)

    basic_model = load_pKa_basic_model()
    acid_model = load_pKa_acidic_model()

    basic_model.eval()
    acid_model.eval()

    true_logP, predicted_logP = [], []

    for index, row in df_test_splited_by_tanimoto.iterrows():

        pKa_smiles = row['Smiles']
        true_value = row['pKa']

        bg = prepare_graphs(pKa_smiles)

        with torch.no_grad():
            
            if 'n' in pKa_smiles.lower():
                model_prediction, _ = basic_model(bg, bg.ndata['h'], bg.edata['e'])
            elif '=o' in pKa_smiles.lower():
                model_prediction, _ = acid_model(bg, bg.ndata['h'], bg.edata['e'])

            print(pKa_smiles, true_value, model_prediction)
        
            true_logP.append(true_value)
            predicted_logP.append(model_prediction.item())

    print(f"TRUE: {true_logP}")
    print(f"PRED: {predicted_logP}")
    print(calculate_metrics(true_values=true_logP,
                            pred_values=predicted_logP))
    

import pandas as pd
import torch
from rdkit import Chem
from dgllife.model import load_pretrained
from dgllife.utils import smiles_to_bigraph
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from dgllife.utils import AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer
from rdkit import Chem

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt

def calculate_metrics(true_values, pred_values):
    mse = round(mean_squared_error(true_values, pred_values),3)
    mae = round(mean_absolute_error(true_values, pred_values),3)
    r_score = round(r2_score(true_values, pred_values),3)

    return {"mse": mse,
            "mae": mae,
            "r^2": r_score,}

if __name__ == "__main__":

    csv_path = r'data\init_data\pKa_Prediction_Starting data_2024.01.25.csv'
    csv_logP_test_path = r'data\logP_lipophilicity_data\gnn_cv\test.csv'

    df = pd.read_csv(csv_path, index_col=0)
    df_test_splited_by_tanimoto = pd.read_csv(csv_logP_test_path, index_col=0)

    lipophilicity_model = load_pretrained("AttentiveFP_canonical_Lipophilicity")
    lipophilicity_model.load_state_dict(torch.load(r'ml_part\weights\logP_dgllife_lipophilicity\logP_best_loss_fragrant-meadow-26.pth'))

    lipophilicity_model.eval()

    model = lipophilicity_model.to('cpu')  # перенесіть модель на GPU, якщо ви використовуєте GPU
    model.eval()

    true_logP, predicted_logP = [], []

    for index, row in df_test_splited_by_tanimoto.iterrows():
        logP_smiles = row['Smiles']
        true_value = row['logP']

        g = smiles_to_bigraph(logP_smiles, add_self_loop=True, node_featurizer=CanonicalAtomFeaturizer(), edge_featurizer=CanonicalBondFeaturizer(self_loop=True))
        # g = smiles_to_bigraph(logP_smiles, add_self_loop=True, node_featurizer=AttentiveFPAtomFeaturizer(), edge_featurizer=AttentiveFPBondFeaturizer(self_loop=True))

        g = g.to('cpu')

        with torch.no_grad():
            prediction = model(g, g.ndata['h'], g.edata['e'])
            # prediction = model(g, g.ndata['h'])
            print(prediction.item(), true_value)
            true_logP.append(true_value)
            predicted_logP.append(prediction.item())

    print(calculate_metrics(true_values=true_logP,
                            pred_values=predicted_logP))

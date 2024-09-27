import pandas as pd
import torch
from rdkit import Chem
from dgllife.model import load_pretrained
from dgllife.utils import smiles_to_bigraph
from dgllife.data import Lipophilicity
from dgllife.utils import SMILESToBigraph, CanonicalAtomFeaturizer
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from dgllife.utils import AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer
from dgllife.utils import PretrainAtomFeaturizer, PretrainBondFeaturizer
from rdkit import Chem

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt
from tqdm import tqdm

def calculate_metrics(true_values, pred_values):
    mse = round(mean_squared_error(true_values, pred_values),3)
    mae = round(mean_absolute_error(true_values, pred_values),3)
    r_score = round(r2_score(true_values, pred_values),3)

    return {"mse": mse,
            "mae": mae,
            "r^2": r_score,}

smiles_to_g = SMILESToBigraph(node_featurizer=CanonicalAtomFeaturizer())
dataset = Lipophilicity(smiles_to_g, load=True)

df = dataset.df

if __name__ == "__main__":
    device = 'cpu'

    smiles_to_g = SMILESToBigraph(node_featurizer=CanonicalAtomFeaturizer())
    dataset = Lipophilicity(smiles_to_g, load=True)

    df = dataset.df

    csv_logP_train_path = r'data\logP_lipophilicity_data\gnn_cv\train.csv'
    csv_logP_test_path = r'data\logP_lipophilicity_data\gnn_cv\test.csv'

    df_train = pd.read_csv(csv_logP_train_path, index_col=0)
    df_test = pd.read_csv(csv_logP_test_path, index_col=0)

    df = pd.concat([df_train, df_test], axis=0)

    model_name = 'GCN_attentivefp_Lipophilicity'
    lipophilicity_model = load_pretrained(model_name)
    print(lipophilicity_model)
    lipophilicity_model.load_state_dict(torch.load(r'C:\work\DrugDiscovery\main_git\XAI_Chem\ml_part\weights\logP_dgllife_lipophilicity\best_models\GNN_attentivefp_model_logP_best_loss.pth'))

    lipophilicity_model.eval()

    model = lipophilicity_model.to(device)  # перенесіть модель на GPU, якщо ви використовуєте GPU
    model.eval()

    true_logP, predicted_logP = [], []

    for index, row in tqdm(df.iterrows()):
        logP_smiles = row['Smiles']
        true_value = row['logP']

        # g = smiles_to_bigraph(logP_smiles, add_self_loop=True, node_featurizer=CanonicalAtomFeaturizer(), edge_featurizer=CanonicalBondFeaturizer(self_loop=True))
        g = smiles_to_bigraph(logP_smiles, add_self_loop=True, node_featurizer=AttentiveFPAtomFeaturizer(), edge_featurizer=AttentiveFPBondFeaturizer(self_loop=True))
        # g = smiles_to_bigraph(logP_smiles, add_self_loop=True, node_featurizer=PretrainAtomFeaturizer(), edge_featurizer=PretrainBondFeaturizer(self_loop=True))

        g = g.to(device)

        with torch.no_grad():
            # prediction = model(g, g.ndata['h'], g.edata['e'])
            prediction = model(g, g.ndata['h'])
            
            # MPNN model:
            # h = g.ndata.pop('h')
            # e = g.edata.pop('e')
            # prediction = model(g, h, e)

            # supervised models
            # node_feats = [
            #     g.ndata.pop('atomic_number'),
            #     g.ndata.pop('chirality_type')
            # ]
            # edge_feats = [
            #     g.edata.pop('bond_type'),
            #     g.edata.pop('bond_direction_type')
            # ]
            # node_feats = [n.to(device) for n in node_feats]
            # edge_feats = [e.to(device) for e in edge_feats]
            # prediction = model(g, node_feats, edge_feats)
            
            # print(prediction.item(), true_value)
            true_logP.append(true_value)
            predicted_logP.append(prediction.item())

            print(logP_smiles, true_value, prediction.item())

    print(calculate_metrics(true_values=true_logP,
                            pred_values=predicted_logP))
    print(df['Smiles'].tolist())

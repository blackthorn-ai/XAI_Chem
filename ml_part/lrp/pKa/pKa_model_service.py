import torch
import torch.nn.functional as F
from dgllife.model.model_zoo.gcn_predictor import GCNPredictor

from gnn_models import Pka_acidic_view, Pka_basic_view
from constants import Identificator

class PkaModelService:
    def __init__(self, 
                 identificator: Identificator = Identificator.secondary_amine,
                 is_combined_model: bool = False,
                 amine_model_weights_path: str = r"ml_part\weights\best_weights\pKa_canon_smiles\amine_best_loss_morning-sky-87.pkl",
                 acid_model_weights_path: str = r"ml_part\weights\best_weights\pKa_canon_smiles\acid_best_loss_lilac-pine-66.pkl") -> None:
        if is_combined_model:
            acid_model_weights_path = r'ml_part\weights\pKa\combined_dataset\acid_best_loss_daily-morning-84.pkl'
            amine_model_weights_path = r'ml_part\weights\pKa\combined_dataset\acid_best_loss_daily-morning-84.pkl'

            amine_model_weights_path = r'ml_part\weights\pKa\combined_dataset\cv_models\acidic\cv_1_best_loss_lr_0.0007452146740113421_wd_0.0025791056042483687_train_type_predictor_and_readout.pkl'
        
        if "amine" in identificator.name.lower():
            self.pKa_model = PkaModelService.load_pKa_basic_model(amine_model_weights_path)
        elif "acid" in identificator.name.lower():
            self.pKa_model = PkaModelService.load_pKa_acidic_model(acid_model_weights_path)

    @staticmethod
    def load_pKa_acidic_model(model_path):
        """
        Load the pre-trained pKa acidic prediction model.

        Args:
            model_path (str): The path to the pre-trained model file.

        Returns:
            pka_model (Pka_acidic_view): The loaded pKa acidic prediction model.
        """
        pka_model = Pka_acidic_view(
            node_feat_size = 74,
            edge_feat_size = 12,
            output_size = 1,
            num_layers= 6,
            graph_feat_size=200,
            dropout=0).to('cpu')
        
        pka_model.load_state_dict(torch.load(model_path,map_location='cpu'))
        
        return pka_model
    
    def load_pKa_basic_model(model_path):
        """
        Load the pre-trained pKa basic prediction model.

        Args:
            model_path (str): The path to the pre-trained model file.

        Returns:
            pka2_model (Pka_basic_view): The loaded pKa basic prediction model.
        """
        pka_model = Pka_basic_view(
            node_feat_size = 74,
            edge_feat_size = 12,
            output_size = 1,
            num_layers= 6,
            graph_feat_size=200,
            dropout=0).to('cpu')
        
        pka_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        return pka_model

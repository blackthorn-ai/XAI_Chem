import torch
import torch.nn.functional as F
from dgllife.model.model_zoo.gcn_predictor import GCNPredictor

from gnn_models import Pka_acidic_view, Pka_basic_view
from constants import Identificator

class PkaModelService:
    def __init__(self, 
                 identificator: Identificator = Identificator.secondary_amine,
                 amine_model_weights_path: str = r"ml_part\weights\best_weights\basic_best_loss_glamorous-totem-39.pkl",
                 acid_model_weights_path: str = r"ml_part\weights\best_weights\acid_best_loss_blooming-deluge-52.pkl") -> None:
        
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

import torch
import torch.nn.functional as F
from dgllife.model.model_zoo.gcn_predictor import GCNPredictor

class LogPModelService:
    def __init__(self, 
                 model_name: str,
                 model_weights_path: str) -> None:
        
        if "gcn" not in model_name.lower():
            raise ValueError("This model is not supported yet.")
        
        self.logP_model = LogPModelService.load_logP_model(model_weights_path)

    @staticmethod
    def load_logP_model(model_path):
        """
        Load the pre-trained logP prediction model.

        Args:
            model_path (str): The path to the pre-trained model file.

        Returns:
            logP_model (GCNPredictor): The loaded logP prediction model.
        """
        dropout = 0.0690767663743611
        num_gnn_layers = 2
        logP_model =  GCNPredictor(
            in_feats=39,
            hidden_feats=[64] * num_gnn_layers,
            activation=[F.relu] * num_gnn_layers,
            residual=[True] * num_gnn_layers,
            batchnorm=[False] * num_gnn_layers,
            dropout=[dropout] * num_gnn_layers,
            predictor_hidden_feats=128,
            predictor_dropout=dropout
        ).to('cpu')

        logP_model.load_state_dict(torch.load(model_path, map_location='cpu'))

        return logP_model
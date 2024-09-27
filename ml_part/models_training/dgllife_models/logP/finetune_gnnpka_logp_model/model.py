import torch
from dgllife.model import load_pretrained
from dgllife.model.model_zoo.attentivefp_predictor import AttentiveFPPredictor

class LipophilicityModelService:
    def __init__(self,
                 model_name: str = None,
                 is_finetune: bool = True) -> None:
        self.model = self.load_model(model_name=model_name)
        self.is_finetune = is_finetune
        
        self.freeze_all()

        # return self.model

    def load_model(self,
                   model_name: str = None):
        if model_name is None:
            model_name = 'AttentiveFP_canonical_Lipophilicity'

        model = load_pretrained(model_name)

        model = model.to('cpu')
        # if torch.cuda.is_available():
        #     model = model.to('cuda')

        return model
    
    def freeze_all(self):
        self.model.eval()

    def train_mode(self, train_type: str = None):

        if train_type == 'all_layers':
            self.model.gnn.train()
            self.model.readout.train()
            self.model.predict.train()
        elif train_type == 'predictor_and_readout':
            self.model.gnn.eval()
            self.model.readout.train()
            self.model.predict.train()
        elif train_type == 'all_layers' or train_type is None:
            self.model.gnn.eval()
            self.model.readout.eval()
            self.model.predict.train()


    def eval_mode(self):
        self.model.gnn.eval()
        self.model.readout.eval()
        self.model.predict.eval()

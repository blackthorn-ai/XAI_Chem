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
        if torch.cuda.is_available():
            model = model.to('cuda')

        return model

    def freeze_all(self):
        self.model.eval()

    def train_mode(self, model=None, train_type: str = None):
        # print(train_type)

        if model is None:
            model = self.model

        if train_type == 'all_layers':
            model.gnn.train()
            model.readout.train()
            model.predict.train()
        elif train_type == 'predictor_and_readout':
            print('predictor_and_readout train mode')
            model.gnn.train()
            model.readout.train()
            model.predict.train()
        elif train_type == 'only_predictor' or train_type is None:
            print('only_predictor train mode')
            model.gnn.train()
            model.readout.train()
            model.predict.train()

        return model


    def eval_mode(self, model=None):
        if model is None:
            model = self.model
        model.gnn.eval()
        model.readout.eval()
        model.predict.eval()

        return model

import pandas as pd

from trainer import Trainer
from trainer import crossvals_list

if __name__ == "__main__":
    Trainer()
    print(crossvals_list)

    df = pd.DataFrame(crossvals_list)
    df.to_csv(r'data\logP_lipophilicity_data\gnn_cv\attentivefp_cv_result.csv')

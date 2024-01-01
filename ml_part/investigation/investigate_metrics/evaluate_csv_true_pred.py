from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt
import pandas as pd

def calculate_metrics(true_values, pred_values):
    mse = round(sqrt(mean_squared_error(true_values, pred_values)),3)
    mae = round(mean_absolute_error(true_values, pred_values),3)
    r_score = round(r2_score(true_values, pred_values),3)

    return {"mse": mse,
            "mae": mae,
            "r_score": r_score,}

df = pd.read_csv(r'C:\work\DrugDiscovery\RT_LogP_with_pKa_model\RTlogD\RD_dataset\test_logP_data_evaluated.csv')

true_value = df['logP']
pred_value = df['logP_predicted']

print(calculate_metrics(true_value, pred_value))
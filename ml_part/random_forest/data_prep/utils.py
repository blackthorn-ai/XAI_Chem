import numpy as np
from scipy import stats
from ast import literal_eval


def detect_outliers_iqr(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_indices = [i for i, x in enumerate(data) if x < lower_bound or x > upper_bound]

    return outlier_indices


def detect_outliers_zscore(data:list(),
                           threshold:int=3):
    z = abs(stats.zscore(data, nan_policy='omit'))
    
    outlier_indices = np.where(z > threshold)[0]

    return outlier_indices


def literal_return(val):
    try:
        return literal_eval(val)
    except (ValueError, SyntaxError) as e:
        return val
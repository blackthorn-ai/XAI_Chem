import numpy as np

def detect_outliers_iqr(data):
    # Calculate Q1, Q3 and IQR
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Get the indices of the outliers
    outlier_indices = [i for i, x in enumerate(data) if x < lower_bound or x > upper_bound]

    return outlier_indices


# Приклад використання
your_data = [1, 2, 3, 4, 5, 10]
outlier_indices = detect_outliers_iqr(your_data)
print("Індекси з аутлаєрами:", outlier_indices)

import pandas as pd

excel_file_path = r'ml_part\molecule-features\pKa_Prediction_Starting data_2023.11.22.xlsx'
df = pd.read_excel(excel_file_path, sheet_name="Main_List")

print(df.keys())

print(df['Unnamed: 13'])
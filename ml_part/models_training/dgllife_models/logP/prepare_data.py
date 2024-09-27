import pandas as pd

if __name__ == "__main__":

    csv_path = r'data\init_data\pKa_Prediction_Starting data_2024.01.25.csv'
    csv_logP_test_path = r'data\RD_dataset\train_logP_data.csv'

    df = pd.read_csv(csv_path, index_col=0)
    df_test_splited_by_tanimoto = pd.read_csv(csv_logP_test_path, index_col=0)

    smiles_to_amides = {}

    logP_values, amide_smiles_values = [], []
    for index, row in df.iterrows():
        smiles_to_amides[row['Smiles']] = row['Amides for LogP']

    for index, row in df_test_splited_by_tanimoto.iterrows():
        logP_values.append(row['logP'])
        amide_smiles_values.append(smiles_to_amides[row['smiles']])

        # print(row['smiles'], smiles_to_amides[row['smiles']])

    print(df_test_splited_by_tanimoto.index.to_list())

    df_prepared_for_train = pd.DataFrame({"Smiles": amide_smiles_values,
                                          "logP": logP_values,
                                          "y_category": df_test_splited_by_tanimoto['y_category']},
                                          index=df_test_splited_by_tanimoto.index.to_list())

    df_prepared_for_train.to_csv(r'data\logP_lipophilicity_data\train.csv')
    print(df_prepared_for_train)
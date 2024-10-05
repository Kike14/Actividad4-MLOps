import requests
import pandas as pd
import pickle
from scipy.stats import ks_2samp

with open('./models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

data_to_predict = pd.read_csv('./data/credit_pred.csv')
data_train = pd.read_csv('./data/credit_train.csv')

df_predictions = pd.DataFrame()
batch_size = 1000

feature_columns = data_to_predict.columns

for index, row in data_to_predict.iterrows():
    x_i = pd.DataFrame([row], columns=feature_columns)  

    x_i_scaled = scaler.transform(x_i)

    response = requests.post("http://127.0.0.1:1234/predict", json=x_i_scaled[0].tolist())

    prediction = response.json()["prediction"]

    row['Y'] = prediction

    df_predictions = pd.concat([df_predictions, row.to_frame().T], ignore_index=True)

    if len(df_predictions) % batch_size == 0:
        print(f"Realizando pruebas de hipótesis para el lote {len(df_predictions) // batch_size}...")

        for column in df_predictions.drop('Y', axis=1).columns:
            ks_stat, p_val = ks_2samp(df_predictions[column], data_train[column])
            if p_val < 0.05:
                print(
                    f"El test Kolmogorov-Smirnov falló para la columna {column}: las distribuciones son significativamente diferentes.")

df_predictions.to_csv('./data/predictions.csv', index=False)

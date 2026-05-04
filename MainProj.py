import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

base_dir = os.path.dirname(os.path.abspath(__file__))
dados_path = os.path.join(base_dir, 'GYN_DADOS')

arquivos = glob.glob(os.path.join(dados_path, '*.CSV')) + glob.glob(os.path.join(dados_path, '*.csv'))

if len(arquivos) == 0:
    raise FileNotFoundError('Nenhum CSV encontrado na pasta GYN_DADOS.')

lista_df = []

for arquivo in arquivos:
    df = pd.read_csv(arquivo, sep=';', encoding='latin1', skiprows=8)

    df.columns = (
        df.columns
        .str.strip()
        .str.replace('\ufeff', '', regex=False)
        .str.upper()
    )

    col_data = [col for col in df.columns if 'DATA' in col][0]
    col_hora = [col for col in df.columns if 'HORA' in col][0]
    col_prec = [col for col in df.columns if 'PRECIP' in col][0]

    df = df[[col_data, col_hora, col_prec]].copy()
    df.columns = ['data', 'hora', 'precipitacao']

    df['data'] = df['data'].astype(str).str.strip()
    df['hora'] = (
        df['hora']
        .astype(str)
        .str.strip()
        .str.replace(' UTC', '', regex=False)
        .str.replace(':', '', regex=False)
        .str.zfill(4)
    )

    df['precipitacao'] = (
        df['precipitacao']
        .astype(str)
        .str.strip()
        .str.replace(',', '.', regex=False)
    )

    dt1 = pd.to_datetime(df['data'] + ' ' + df['hora'], format='%Y/%m/%d %H%M', errors='coerce')
    dt2 = pd.to_datetime(df['data'] + ' ' + df['hora'], format='%Y-%m-%d %H%M', errors='coerce')
    dt3 = pd.to_datetime(df['data'] + ' ' + df['hora'], format='%d/%m/%Y %H%M', errors='coerce')
    dt4 = pd.to_datetime(df['data'] + ' ' + df['hora'], format='%d-%m-%Y %H%M', errors='coerce')

    df['datetime'] = dt1.fillna(dt2).fillna(dt3).fillna(dt4)

    df['precipitacao'] = pd.to_numeric(df['precipitacao'], errors='coerce')
    df.loc[df['precipitacao'] < 0, 'precipitacao'] = pd.NA

    df = df[['datetime', 'precipitacao']].dropna()
    lista_df.append(df)

df = pd.concat(lista_df, ignore_index=True)

df = df.sort_values('datetime')
df = df.drop_duplicates(subset='datetime')
df.set_index('datetime', inplace=True)

df_mensal = df.resample('ME').sum()

scaler = MinMaxScaler()
dados = scaler.fit_transform(df_mensal[['precipitacao']])

def criar_sequencias(dados, passos=12):
    X, y = [], []
    for i in range(len(dados) - passos):
        X.append(dados[i:i + passos])
        y.append(dados[i + passos])
    return np.array(X), np.array(y)

X, y = criar_sequencias(dados, 12)

n = len(X)
fim_treino = int(n * 0.7)
fim_validacao = int(n * 0.85)

X_train = X[:fim_treino]
y_train = y[:fim_treino]

X_val = X[fim_treino:fim_validacao]
y_val = y[fim_treino:fim_validacao]

X_test = X[fim_validacao:]
y_test = y[fim_validacao:]

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1, activation='relu'))

model.compile(optimizer='adam', loss='mse')

historico = model.fit(
    X_train,
    y_train,
    epochs=50,
    validation_data=(X_val, y_val),
    verbose=1
)

pred = model.predict(X_test)

pred_real = scaler.inverse_transform(pred)
pred_real = np.maximum(pred_real, 0)

y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

mae = mean_absolute_error(y_test_real, pred_real)
mse = mean_squared_error(y_test_real, pred_real)
rmse = np.sqrt(mse)

print('MAE:', mae)
print('MSE:', mse)
print('RMSE:', rmse)

datas_test = df_mensal.index[12 + fim_validacao:]

plt.figure(figsize=(12, 6))
plt.plot(datas_test, y_test_real, label='Real')
plt.plot(datas_test, pred_real, label='Previsto')
plt.title('Precipitação Mensal - Real vs Previsto')
plt.xlabel('Data')
plt.ylabel('Precipitação (mm)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(historico.history['loss'], label='Treino')
plt.plot(historico.history['val_loss'], label='Validação')
plt.title('Perda do Modelo')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(df_mensal.head(15))
print(df_mensal.tail(15))
print(df_mensal.shape)
print(df.index.min(), 'até', df.index.max())

df_mensal.to_csv('precipitacao_mensal_goiania.csv', encoding='utf-8-sig')
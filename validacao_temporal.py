"""Validacao temporal exploratoria: LSTM atual, Modelo A e climatologia.

Execute MainProj.py primeiro para gerar a serie mensal em resultados/.
Os periodos de avaliacao aqui ficam ANTES do teste final do MainProj.py.
Cada janela utiliza exclusivamente meses anteriores ao alvo.
"""
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Sequential

BASE = os.path.dirname(os.path.abspath(__file__))
PASTA = os.path.join(BASE, "resultados")
ENTRADA = os.path.join(PASTA, "precipitacao_mensal_goiania_multivariado.csv")
PASSOS = 12
EPOCAS = 80
BATCH_SIZE = 8
SEMENTES = (42, 123, 456)
COLUNAS = [
    "precipitacao", "temperatura", "umidade", "pressao",
    "vento", "mes_sin", "mes_cos"
]
# Cada corte reserva os 12 meses anteriores para parada antecipada,
# e os 12 meses seguintes para avaliacao fora da amostra.
CORTES = ("2016-01-31", "2018-01-31", "2020-01-31")


def construir_modelo(nome, dimensoes):
    if nome == "Atual":
        camadas = [
            Input(shape=(PASSOS, dimensoes)),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dense(16, activation="relu"),
            Dense(1, activation="linear")
        ]
    else:
        camadas = [
            Input(shape=(PASSOS, dimensoes)),
            LSTM(32),
            Dense(1, activation="linear")
        ]
    modelo = Sequential(camadas)
    modelo.compile(optimizer="adam", loss="mse")
    return modelo


def sequencias(valores, inicio, fim):
    entradas, alvos, indices = [], [], []
    for i in range(max(PASSOS, inicio), fim):
        janela = valores[i - PASSOS:i]
        alvo = valores[i, 0]
        if np.isfinite(janela).all() and np.isfinite(alvo):
            entradas.append(janela)
            alvos.append(alvo)
            indices.append(i)
    if not entradas:
        return (
            np.empty((0, PASSOS, valores.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.array([], dtype=int)
        )
    return (
        np.asarray(entradas, dtype=np.float32),
        np.asarray(alvos, dtype=np.float32),
        np.asarray(indices, dtype=int)
    )


def metricas(real, previsto):
    return (
        mean_absolute_error(real, previsto),
        np.sqrt(mean_squared_error(real, previsto))
    )


def main():
    if not os.path.isfile(ENTRADA):
        raise FileNotFoundError(
            "Execute MainProj.py primeiro para gerar: " + ENTRADA
        )
    df = pd.read_csv(ENTRADA, index_col=0, parse_dates=True)
    df = df.sort_index()
    if not all(c in df.columns for c in COLUNAS):
        raise ValueError("A serie mensal nao contem todas as colunas esperadas.")
    if not df.index.is_unique:
        raise ValueError("Datas mensais duplicadas.")
    if not (df.index == pd.date_range(
        df.index.min(), df.index.max(), freq="ME"
    )).all():
        raise ValueError("Calendario mensal incompleto: nao unir meses separados.")

    limite_teste = int(len(df) * 0.85)
    registros = []
    for corte_texto in CORTES:
        corte = pd.Timestamp(corte_texto)
        inicio_avaliacao = int(df.index.searchsorted(corte))
        fim_avaliacao = min(inicio_avaliacao + 12, limite_teste)
        inicio_validacao = inicio_avaliacao - 12
        if inicio_validacao <= PASSOS or fim_avaliacao <= inicio_avaliacao:
            raise ValueError("Corte fora do periodo de desenvolvimento: " + corte_texto)

        # Ajuste do scaler exclusivamente com meses anteriores a validacao.
        base_scaler = df.iloc[:inicio_validacao][COLUNAS].dropna()
        if base_scaler.empty:
            raise ValueError("Sem dados para ajustar scaler em " + corte_texto)
        scaler = MinMaxScaler().fit(base_scaler)
        valores = scaler.transform(df[COLUNAS])
        X_t, y_t, _ = sequencias(valores, PASSOS, inicio_validacao)
        X_v, y_v, _ = sequencias(valores, inicio_validacao, inicio_avaliacao)
        X_a, y_a, idx_a = sequencias(valores, inicio_avaliacao, fim_avaliacao)
        if min(len(y_t), len(y_v), len(y_a)) == 0:
            print("Ignorando corte sem sequencias completas:", corte_texto)
            continue

        escala, minimo = scaler.scale_[0], scaler.min_[0]
        real = (y_a - minimo) / escala
        climatologia = (
            df.iloc[:inicio_avaliacao]["precipitacao"]
            .groupby(df.index[:inicio_avaliacao].month)
            .mean()
        )
        meses = df.index[idx_a].month
        if climatologia.reindex(meses).isna().any():
            print("Ignorando corte sem climatologia:", corte_texto)
            continue
        p_clima = climatologia.reindex(meses).to_numpy()
        mae_c, rmse_c = metricas(real, p_clima)
        registros.append({
            "Periodo": corte_texto[:4],
            "Modelo": "Climatologia",
            "Semente": "-",
            "Meses": len(real),
            "MAE_mm": mae_c,
            "RMSE_mm": rmse_c
        })

        for nome in ("Atual", "Modelo A"):
            for semente in SEMENTES:
                tf.keras.backend.clear_session()
                random.seed(semente)
                np.random.seed(semente)
                tf.random.set_seed(semente)
                modelo = construir_modelo(nome, X_t.shape[2])
                modelo.fit(
                    X_t, y_t,
                    validation_data=(X_v, y_v),
                    epochs=EPOCAS,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    verbose=0,
                    callbacks=[EarlyStopping(
                        monitor="val_loss", patience=15,
                        restore_best_weights=True
                    )]
                )
                p = modelo.predict(X_a, verbose=0).reshape(-1)
                previsto = np.maximum((p - minimo) / escala, 0)
                mae, rmse = metricas(real, previsto)
                registros.append({
                    "Periodo": corte_texto[:4],
                    "Modelo": nome,
                    "Semente": semente,
                    "Meses": len(real),
                    "MAE_mm": mae,
                    "RMSE_mm": rmse
                })
                print(f"{corte_texto[:4]} | {nome} | {semente} | MAE {mae:.2f}")

    if not registros:
        raise ValueError("Nenhum periodo valido; revise as lacunas dos dados.")
    detalhado = pd.DataFrame(registros)
    detalhado.to_csv(
        os.path.join(PASTA, "validacao_temporal_detalhada.csv"),
        index=False, encoding="utf-8-sig"
    )
    resumo = (
        detalhado.groupby("Modelo", as_index=False)
        .agg(MAE_Medio_mm=("MAE_mm", "mean"),
             MAE_Desvio_mm=("MAE_mm", "std"),
             RMSE_Medio_mm=("RMSE_mm", "mean"),
             Avaliacoes=("MAE_mm", "size"))
        .sort_values("MAE_Medio_mm")
    )
    resumo.to_csv(
        os.path.join(PASTA, "validacao_temporal_resumo.csv"),
        index=False, encoding="utf-8-sig"
    )
    print("\nResumo exploratorio (medias por execucao, nao por mes):")
    print(resumo.to_string(index=False))
    print("\nResultados em:", PASTA)


if __name__ == "__main__":
    main()

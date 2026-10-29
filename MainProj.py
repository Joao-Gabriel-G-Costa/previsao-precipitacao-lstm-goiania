
import os
import glob
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# 1. CONFIGURACOES
# ============================================================

SEED = 42
PASSOS = 12
EPOCAS = 80
BATCH_SIZE = 8

# Proporcao minima de registros horarios com precipitacao
# valida para aceitar um mes.
COBERTURA_MINIMA = 0.90

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

base_dir = os.path.dirname(os.path.abspath(__file__))
dados_path = os.path.join(base_dir, "GYN_DADOS")
resultados_path = os.path.join(base_dir, "resultados")

os.makedirs(resultados_path, exist_ok=True)


# ============================================================
# 2. LEITURA DOS ARQUIVOS
# ============================================================

arquivos = sorted(
    set(
        glob.glob(os.path.join(dados_path, "*.CSV"))
        + glob.glob(os.path.join(dados_path, "*.csv"))
    )
)

if not arquivos:
    raise FileNotFoundError(
        "Nenhum arquivo CSV encontrado na pasta GYN_DADOS."
    )

lista_df = []


def encontrar_coluna(colunas, termos):
    for coluna in colunas:
        if any(termo in coluna for termo in termos):
            return coluna

    raise ValueError(
        f"Coluna nao encontrada. Termos procurados: {termos}"
    )


for arquivo in arquivos:
    print("Lendo:", os.path.basename(arquivo))

    df = pd.read_csv(
        arquivo,
        sep=";",
        encoding="latin1",
        skiprows=8,
        low_memory=False
    )

    df.columns = (
        df.columns
        .str.strip()
        .str.replace("\ufeff", "", regex=False)
        .str.upper()
    )

    col_data = encontrar_coluna(df.columns, ["DATA"])
    col_hora = encontrar_coluna(df.columns, ["HORA"])
    col_prec = encontrar_coluna(df.columns, ["PRECIP"])
    col_temp = encontrar_coluna(
        df.columns, ["TEMPERATURA DO AR"]
    )
    col_umid = encontrar_coluna(
        df.columns, ["UMIDADE RELATIVA DO AR"]
    )
    col_press = encontrar_coluna(
        df.columns,
        [
            "PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO",
            "PRESSÃO ATMOSFERICA AO NIVEL DA ESTACAO"
        ]
    )
    col_vento = encontrar_coluna(
        df.columns, ["VENTO, VELOCIDADE"]
    )

    df = df[
        [
            col_data,
            col_hora,
            col_prec,
            col_temp,
            col_umid,
            col_press,
            col_vento
        ]
    ].copy()

    df.columns = [
        "data",
        "hora",
        "precipitacao",
        "temperatura",
        "umidade",
        "pressao",
        "vento"
    ]

    df["data"] = df["data"].astype(str).str.strip()

    df["hora"] = (
        df["hora"]
        .astype(str)
        .str.strip()
        .str.replace(" UTC", "", regex=False)
        .str.replace(":", "", regex=False)
        .str.zfill(4)
    )

    datas = [
        pd.to_datetime(
            df["data"] + " " + df["hora"],
            format=formato,
            errors="coerce"
        )
        for formato in [
            "%Y/%m/%d %H%M",
            "%Y-%m-%d %H%M",
            "%d/%m/%Y %H%M",
            "%d-%m-%Y %H%M"
        ]
    ]

    df["datetime"] = datas[0]

    for datas_alternativas in datas[1:]:
        df["datetime"] = df["datetime"].fillna(
            datas_alternativas
        )

    for coluna in [
        "precipitacao",
        "temperatura",
        "umidade",
        "pressao",
        "vento"
    ]:
        df[coluna] = (
            df[coluna]
            .astype(str)
            .str.strip()
            .str.replace(",", ".", regex=False)
        )

        df[coluna] = pd.to_numeric(
            df[coluna],
            errors="coerce"
        )

    df.loc[df["precipitacao"] < 0, "precipitacao"] = np.nan
    df.loc[df["temperatura"] < -50, "temperatura"] = np.nan
    df.loc[
        (df["umidade"] < 0) | (df["umidade"] > 100),
        "umidade"
    ] = np.nan
    df.loc[df["pressao"] <= 0, "pressao"] = np.nan
    df.loc[df["vento"] < 0, "vento"] = np.nan

    df = df[
        [
            "datetime",
            "precipitacao",
            "temperatura",
            "umidade",
            "pressao",
            "vento"
        ]
    ]

    df = df.dropna(subset=["datetime"])

    lista_df.append(df)


# ============================================================
# 3. ORGANIZACAO DOS DADOS
# ============================================================

df = pd.concat(lista_df, ignore_index=True)

df = df.sort_values("datetime")
df = df.drop_duplicates(subset="datetime")
df = df.set_index("datetime")

print("\nPeriodo dos dados:")
print(df.index.min(), "ate", df.index.max())

# Agregacao mensal.
# A precipitacao nao depende da disponibilidade das outras
# variaveis, evitando descartar chuva registrada.

df_mensal = df.resample("ME").agg({
    "precipitacao": lambda valores: valores.sum(min_count=1),
    "temperatura": "mean",
    "umidade": "mean",
    "pressao": "mean",
    "vento": "mean"
})

# Quantidade de registros horarios validos por mes.
horas_validas = df["precipitacao"].resample("ME").count()

# Quantidade esperada de horas em cada mes.
# Este calculo considera meses completos, inclusive os
# meses inicial e final da serie.
horas_esperadas = (
    df_mensal.index.days_in_month.to_numpy() * 24
)

df_mensal["cobertura_precipitacao"] = (
    horas_validas.to_numpy() / horas_esperadas
)

# Diagnostico de qualidade dos dados.
diagnostico = df_mensal[
    ["precipitacao", "cobertura_precipitacao"]
].copy()

diagnostico.to_csv(
    os.path.join(resultados_path, "diagnostico_mensal.csv"),
    encoding="utf-8-sig"
)

print("\nMeses com cobertura insuficiente:")
print(
    diagnostico[
        diagnostico["cobertura_precipitacao"]
        < COBERTURA_MINIMA
    ].to_string()
)

# Meses com cobertura insuficiente nao sao utilizados.
# Nao sao substituidos por zero.
df_mensal.loc[
    df_mensal["cobertura_precipitacao"] < COBERTURA_MINIMA,
    "precipitacao"
] = np.nan

df_mensal = df_mensal.drop(
    columns=["cobertura_precipitacao"]
)

# Variaveis sazonais.
df_mensal["mes"] = df_mensal.index.month

df_mensal["mes_sin"] = np.sin(
    2 * np.pi * df_mensal["mes"] / 12
)

df_mensal["mes_cos"] = np.cos(
    2 * np.pi * df_mensal["mes"] / 12
)

df_mensal = df_mensal.drop(columns=["mes"])

# Mantemos a linha temporal completa.
# Os meses invalidos serao excluidos das sequencias,
# sem juntar meses que nao sao consecutivos.

colunas = [
    "precipitacao",
    "temperatura",
    "umidade",
    "pressao",
    "vento",
    "mes_sin",
    "mes_cos"
]

print("\nTotal de meses no calendario:", len(df_mensal))
print(
    "Meses com todas as variaveis validas:",
    df_mensal[colunas].notna().all(axis=1).sum()
)


# ============================================================
# 4. DIVISAO TEMPORAL
# ============================================================

# A divisao e feita pelo calendario antes da normalizacao.
# O alvo de cada sequencia pertence ao periodo correspondente.

n_meses = len(df_mensal)

fim_treino = int(n_meses * 0.70)
fim_validacao = int(n_meses * 0.85)

if fim_treino <= PASSOS or fim_validacao >= n_meses:
    raise ValueError(
        "Serie temporal insuficiente para a divisao."
    )

dados_treino = df_mensal.iloc[:fim_treino][colunas]

# Ajustar o normalizador somente com registros completos
# pertencentes ao periodo de treinamento.
dados_treino_validos = dados_treino.dropna()

if dados_treino_validos.empty:
    raise ValueError(
        "Nao existem dados validos para ajustar o scaler."
    )

scaler = MinMaxScaler()
scaler.fit(dados_treino_validos)

# Transformar a serie completa com o scaler do treino.
dados = scaler.transform(df_mensal[colunas])

datas = df_mensal.index


# ============================================================
# 5. CRIACAO DAS SEQUENCIAS
# ============================================================

def criar_sequencias(dados, datas, passos):
    X = []
    y = []
    indices_alvo = []

    for i in range(passos, len(dados)):
        janela = dados[i - passos:i]
        alvo = dados[i, 0]

        # A janela e o alvo precisam ter valores validos.
        if not np.isfinite(janela).all():
            continue

        if not np.isfinite(alvo):
            continue

        # A serie mantem o calendario mensal completo.
        # Assim, cada janela corresponde a meses consecutivos.
        X.append(janela)
        y.append(alvo)
        indices_alvo.append(i)

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        np.array(indices_alvo, dtype=int)
    )


X, y, indices_alvo = criar_sequencias(
    dados,
    datas,
    PASSOS
)

mascara_treino = indices_alvo < fim_treino

mascara_validacao = (
    (indices_alvo >= fim_treino)
    & (indices_alvo < fim_validacao)
)

mascara_teste = indices_alvo >= fim_validacao

X_train = X[mascara_treino]
y_train = y[mascara_treino]

X_val = X[mascara_validacao]
y_val = y[mascara_validacao]

X_test = X[mascara_teste]
y_test = y[mascara_teste]

indices_test = indices_alvo[mascara_teste]

if (
    len(X_train) == 0
    or len(X_val) == 0
    or len(X_test) == 0
):
    raise ValueError(
        "Uma das divisoes ficou sem sequencias validas. "
        "Verifique o diagnostico de cobertura mensal."
    )

print("\nQuantidade de sequencias:")
print("Treinamento:", len(X_train))
print("Validacao:", len(X_val))
print("Teste:", len(X_test))


# ============================================================
# 6. MODELO LSTM
# ============================================================

model = Sequential([
    Input(shape=(PASSOS, X_train.shape[2])),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dense(16, activation="relu"),
    Dense(1, activation="linear")
])

model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()


# ============================================================
# 7. TREINAMENTO
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

historico = model.fit(
    X_train,
    y_train,
    epochs=EPOCAS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    shuffle=False,
    verbose=1
)


# ============================================================
# 8. PREVISOES
# ============================================================

pred = model.predict(X_test, verbose=0).reshape(-1)

# Inversao da normalizacao somente para precipitacao.
# A formula usa os parametros aprendidos no treino.
indice_precipitacao = colunas.index("precipitacao")

escala = scaler.scale_[indice_precipitacao]
minimo = scaler.min_[indice_precipitacao]

pred_real = (pred - minimo) / escala
y_test_real = (y_test - minimo) / escala

# Precipitacao nao pode ser negativa.
pred_real = np.maximum(pred_real, 0)


# ============================================================
# 9. METRICAS
# ============================================================

mae = mean_absolute_error(y_test_real, pred_real)
mse = mean_squared_error(y_test_real, pred_real)
rmse = np.sqrt(mse)

print("\nRESULTADOS DA LSTM")
print(f"MAE: {mae:.2f} mm")
print(f"MSE: {mse:.2f} mm²")
print(f"RMSE: {rmse:.2f} mm")


# ============================================================
# 10. MODELO DE REFERENCIA SAZONAL
# ============================================================

# Previsao de referencia:
# utilizar a precipitacao do mesmo mes do ano anterior.
#
# Como a entrada possui 12 meses consecutivos,
# o primeiro elemento da janela corresponde ao mesmo
# mes do ano anterior ao alvo.

baseline_normalizado = X_test[:, 0, indice_precipitacao]

baseline_real = (
    baseline_normalizado - minimo
) / escala

baseline_real = np.maximum(baseline_real, 0)

mae_baseline = mean_absolute_error(
    y_test_real,
    baseline_real
)

rmse_baseline = np.sqrt(
    mean_squared_error(y_test_real, baseline_real)
)

print("\nRESULTADOS DA REFERENCIA SAZONAL")
print(f"MAE: {mae_baseline:.2f} mm")
print(f"RMSE: {rmse_baseline:.2f} mm")

if mae < mae_baseline:
    print("\nA LSTM obteve MAE menor que a referencia.")
else:
    print("\nA referencia obteve MAE menor ou igual a LSTM.")


# ============================================================
# 11. COMPARACAO DAS PREVISOES
# ============================================================

datas_test = datas[indices_test]

comparacao = pd.DataFrame({
    "Data": datas_test,
    "Real_mm": y_test_real,
    "Previsto_LSTM_mm": pred_real,
    "Referencia_Sazonal_mm": baseline_real
})

comparacao["Erro_Absoluto_LSTM"] = abs(
    comparacao["Real_mm"]
    - comparacao["Previsto_LSTM_mm"]
)

# Referencia climatologica: media historica de cada mes do ano,
# calculada exclusivamente com a precipitacao valida do treino.
climatologia_treino = (
    df_mensal.iloc[:fim_treino]["precipitacao"]
    .groupby(df_mensal.index[:fim_treino].month)
    .mean()
)

meses_test = datas_test.month
meses_sem_referencia = sorted(
    set(meses_test) - set(climatologia_treino.dropna().index)
)

if meses_sem_referencia:
    raise ValueError(
        "Nao ha climatologia de treino valida para os meses: "
        + str(meses_sem_referencia)
    )

climatologia_real = np.array(
    [climatologia_treino.loc[mes] for mes in meses_test],
    dtype=float
)

mae_climatologia = mean_absolute_error(
    y_test_real, climatologia_real
)
rmse_climatologia = np.sqrt(
    mean_squared_error(y_test_real, climatologia_real)
)

comparacao["Climatologia_Mensal_mm"] = climatologia_real
comparacao["Erro_Absoluto_Climatologia"] = abs(
    comparacao["Real_mm"] - comparacao["Climatologia_Mensal_mm"]
)

print("\\nRESULTADOS DA CLIMATOLOGIA MENSAL")
print(f"MAE: {mae_climatologia:.2f} mm")
print(f"RMSE: {rmse_climatologia:.2f} mm")

resumo_metricas = pd.DataFrame({
    "Modelo": ["LSTM", "Referencia sazonal", "Climatologia mensal"],
    "MAE_mm": [mae, mae_baseline, mae_climatologia],
    "RMSE_mm": [rmse, rmse_baseline, rmse_climatologia]
})

resumo_metricas.to_csv(
    os.path.join(resultados_path, "resumo_metricas.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\\nRESUMO DOS TRES METODOS")
print(resumo_metricas.to_string(index=False))

print("\\nCOMPARACAO DAS PREVISOES")
print(comparacao.to_string(index=False))

comparacao.to_csv(
    os.path.join(resultados_path, "comparacao_previsoes.csv"),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 12. GRAFICO REAL VS PREVISTO
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    datas_test,
    y_test_real,
    label="Real",
    linewidth=2
)

plt.plot(
    datas_test,
    pred_real,
    label="Previsto LSTM",
    linewidth=2
)

plt.plot(
    datas_test,
    baseline_real,
    label="Referencia sazonal",
    linestyle="--",
    alpha=0.8
)

plt.plot(
    datas_test,
    climatologia_real,
    label="Climatologia mensal (treino)",
    linestyle=":"
)

plt.title("Precipitacao Mensal - Real vs Previsto")
plt.xlabel("Data")
plt.ylabel("Precipitacao (mm)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    os.path.join(resultados_path, "real_vs_previsto.png"),
    dpi=150
)

plt.show()


# ============================================================
# 13. GRAFICO DA PERDA
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    historico.history["loss"],
    label="Treinamento"
)

plt.plot(
    historico.history["val_loss"],
    label="Validacao"
)

plt.title("Perda do Modelo LSTM")
plt.xlabel("Epocas")
plt.ylabel("MSE normalizado")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(resultados_path, "perda_modelo.png"),
    dpi=150
)

plt.show()


# ============================================================
# 14. EXPORTACAO DOS DADOS
# ============================================================

df_mensal.to_csv(
    os.path.join(
        resultados_path,
        "precipitacao_mensal_goiania_multivariado.csv"
    ),
    encoding="utf-8-sig"
)

print("\nPeriodo da serie:")
print(df_mensal.index.min(), "ate", df_mensal.index.max())

print("\nFormato da serie mensal:")
print(df_mensal.shape)

print("\nArquivos salvos em:")
print(resultados_path)

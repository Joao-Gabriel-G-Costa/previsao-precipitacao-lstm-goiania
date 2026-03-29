import pandas as pd
import glob

arquivos = glob.glob(r'C:\Users\Admin\Downloads\IC\GYN_DADOS\*.CSV')

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

    # corrigir valores inválidos do INMET
    df.loc[df['precipitacao'] < 0, 'precipitacao'] = pd.NA

    df = df[['datetime', 'precipitacao']].dropna()

    lista_df.append(df)

df = pd.concat(lista_df, ignore_index=True)

df = df.sort_values('datetime')
df = df.drop_duplicates(subset='datetime')
df.set_index('datetime', inplace=True)

df_mensal = df.resample('ME').sum()

print(df_mensal.head(15))
print(df_mensal.tail(15))
print(df_mensal.shape)
print(df.index.min(), 'até', df.index.max())
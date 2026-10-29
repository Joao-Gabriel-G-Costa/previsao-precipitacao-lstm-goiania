# Previsão de precipitação mensal em Goiânia com LSTM

Projeto acadêmico de previsão de precipitação mensal em Goiânia (GO), com dados meteorológicos históricos do INMET e rede neural **Long Short-Term Memory (LSTM)**.

## Objetivo

Investigar a capacidade de uma LSTM multivariada de prever a precipitação do mês seguinte com base nos **12 meses anteriores**. O projeto é experimental e não substitui previsões meteorológicas operacionais.

## Tecnologias

Python, Pandas, NumPy, TensorFlow/Keras, Scikit-learn e Matplotlib.

## Dados e preparação

Os arquivos meteorológicos são lidos da pasta `GYN_DADOS/` (CSV, separador `;`, codificação `latin1`, ignorando as oito primeiras linhas). O script interpreta datas e horários, converte campos numéricos, elimina valores fisicamente inválidos e remove registros com datas inválidas ou duplicadas.

As variáveis utilizadas são **precipitação, temperatura, umidade relativa, pressão atmosférica e velocidade do vento**. Os dados são agregados por mês: soma da precipitação e média das outras variáveis. O mês do ano também é representado por seno e cosseno, para capturar sazonalidade.

A qualidade dos registros de chuva é verificada por uma **cobertura horária mínima configurável**, atualmente `COBERTURA_MINIMA = 0.90`. Meses abaixo do limite não são considerados válidos para o alvo. As lacunas permanecem no calendário mensal, evitando juntar meses não consecutivos em uma sequência. Janelas com valores ausentes são descartadas.

> **Limitação metodológica:** o limite de 90% é uma escolha experimental, ainda não validada. A cobertura é calculada com base nas horas do mês inteiro, inclusive nos meses inicial e final do conjunto de dados. Meses sem observações suficientes precisam ser investigados nos arquivos originais do INMET.

## Arquitetura e treinamento

| Parâmetro | Implementação atual |
| --- | --- |
| Entrada | 12 meses, 7 variáveis |
| Camadas recorrentes | LSTM(64) e LSTM(32) |
| Regularização | Dropout(0,2) |
| Camada intermediária | Dense(16), ReLU |
| Saída | Dense(1), **linear** |
| Otimizador / perda | Adam / MSE |
| Máximo de épocas | 80 |
| Batch size | 8 |
| EarlyStopping | `val_loss`, paciência 15, restauração dos melhores pesos |
| Divisão temporal | 70% treino, 15% validação, 15% teste |
| Semente | 42 |

A separação é feita em ordem cronológica. O `MinMaxScaler` é **ajustado somente com dados válidos do período de treinamento** e aplicado aos demais períodos, evitando vazamento de informações da validação e do teste durante a normalização.

A saída linear permite valores negativos intermediários; após a transformação inversa, as previsões de precipitação são limitadas a zero ou mais.

## Executar localmente

Recomendado: Python 3.11 e ambiente virtual.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe MainProj.py
```

Se o Windows apresentar erro de caminho longo ao instalar o TensorFlow, é possível criar o ambiente em um caminho mais curto, por exemplo `C:\venvs\lstm`, e utilizar seu executável Python.

Os arquivos de entrada devem estar em `GYN_DADOS/`.

## Resultados

O programa imprime **MAE, MSE e RMSE** do conjunto de teste, além de comparar a LSTM com uma referência sazonal simples: a precipitação observada no mesmo mês do ano anterior.

A pasta `resultados/` é criada automaticamente e contém:

- `diagnostico_mensal.csv` — totais de chuva e cobertura horária;
- `comparacao_previsoes.csv` — valores reais, previstos e referência sazonal;
- `precipitacao_mensal_goiania_multivariado.csv` — série mensal preparada;
- `real_vs_previsto.png` — comparação gráfica das previsões;
- `perda_modelo.png` — curvas de treinamento e validação.

**Esses arquivos são locais, são sobrescritos em novas execuções bem-sucedidas e não são versionados no Git.** Se uma execução falhar antes de gerar determinado arquivo, a versão anterior poderá permanecer na pasta.

Os valores das métricas podem variar entre ambientes e execuções. Resultados isolados não comprovam desempenho generalizável.

## Próximas etapas de pesquisa

1. Investigar os meses ausentes e a qualidade dos registros originais do INMET.
2. Justificar e testar o critério de cobertura horária.
3. Comparar com a climatologia mensal calculada apenas sobre o treinamento.
4. Executar experimentos com múltiplas sementes e validação temporal progressiva.
5. Avaliar arquiteturas menores sem usar o conjunto de teste para selecionar hiperparâmetros.

## Estrutura

```text
.
├── GYN_DADOS/
├── Documentação/
├── MainProj.py
├── requirements.txt
├── .gitignore
└── README.md
```

A pasta `resultados/` é gerada localmente.

## Autor

**João Gabriel Gonçalves Costa** — estudante de Engenharia de Software, SENAI FATESG.

[GitHub](https://github.com/Joao-Gabriel-G-Costa) · [LinkedIn](https://www.linkedin.com/in/jo%C3%A3o-gabriel-gon%C3%A7alves-costa-29b908309)

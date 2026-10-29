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

## Resultados e documentação científica

O projeto inclui comparações com **referência sazonal ingênua** (mesmo mês do ano anterior) e **climatologia mensal** (média histórica daquele mês calculada somente com dados de treinamento).

No teste cronológico principal, com **19 meses válidos**, os resultados registrados foram:

| Método | MAE (mm) | RMSE (mm) |
| --- | ---: | ---: |
| Climatologia mensal | **60,84** | **82,25** |
| LSTM principal (semente 42) | 64,59 | 89,44 |
| Referência sazonal | 89,84 | 127,03 |

Na validação temporal exploratória com cortes em 2016, 2018 e 2020, a climatologia também apresentou o menor MAE médio (42,83 mm), seguida do Modelo A — LSTM(32) (56,42 mm) e da LSTM principal (57,78 mm). **Esses números descrevem execuções específicas, não uma previsão operacional nem evidência de superioridade universal.**

Leia o **[relatório técnico completo](RELATORIO_TECNICO.md)** para metodologia, comparação de arquiteturas, sementes, limitações e interpretação dos resultados.

### Execução dos experimentos

```powershell
python MainProj.py
python validacao_temporal.py
```

Execute `MainProj.py` primeiro: ele gera a série mensal utilizada por `validacao_temporal.py`. O primeiro script também executa os experimentos de sementes e arquiteturas, portanto pode demorar.

A pasta `resultados/` contém, entre outros:

- `diagnostico_mensal.csv` — totais e cobertura de precipitação;
- `precipitacao_mensal_goiania_multivariado.csv` — série mensal;
- `comparacao_previsoes.csv` e `resumo_metricas.csv` — teste principal e referências;
- `experimentos_sementes.csv` e `resumo_sementes.csv` — estabilidade da LSTM;
- `experimentos_arquiteturas.csv` e `resumo_arquiteturas.csv` — comparação de arquiteturas;
- `validacao_temporal_detalhada.csv` e `validacao_temporal_resumo.csv` — validação histórica;
- `real_vs_previsto.png` e `perda_modelo.png` — gráficos do modelo principal.

**Os arquivos gerados são locais e não são versionados no Git.** Cada execução bem-sucedida sobrescreve os arquivos que produz. Se a execução falhar antes da gravação, algum arquivo antigo pode permanecer.

## Estrutura

```text
.
├── GYN_DADOS/
├── Documentação/
├── MainProj.py
├── validacao_temporal.py
├── RELATORIO_TECNICO.md
├── requirements.txt
├── .gitignore
└── README.md
```

A pasta `resultados/` é gerada localmente.

## Autor

**João Gabriel Gonçalves Costa** — estudante de Engenharia de Software, SENAI FATESG.

[GitHub](https://github.com/Joao-Gabriel-G-Costa) · [LinkedIn](https://www.linkedin.com/in/jo%C3%A3o-gabriel-gon%C3%A7alves-costa-29b908309)

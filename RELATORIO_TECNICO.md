# Relatório técnico — Previsão mensal de precipitação em Goiânia com LSTM

**Natureza:** síntese dos experimentos de iniciação científica; resultados exploratórios, não previsão operacional.  
**Fonte dos resultados:** arquivos CSV gerados localmente e analisados em outubro de 2026. Os números abaixo representam as execuções registradas, não uma garantia de reprodução bit a bit.

## 1. Objetivo

Investigar se uma rede neural recorrente do tipo Long Short-Term Memory (LSTM) consegue prever a precipitação mensal de Goiânia a partir dos 12 meses anteriores e como seu desempenho se compara a referências simples.

## 2. Dados e pré-processamento

Foram utilizados arquivos meteorológicos históricos do INMET, abrangendo registros de 2001 a 2025. O programa reúne as observações por mês: soma da precipitação e média de temperatura, umidade, pressão e vento. Seno e cosseno do mês representam a sazonalidade.

O projeto preserva os meses no calendário, não une meses separados por lacunas e exige cobertura mínima experimental de 90% para aceitar a precipitação mensal. Janelas com dados ausentes são excluídas. A normalização MinMax é ajustada somente com o período de treinamento de cada experimento.

**Limitação importante:** o critério de 90% não foi validado como norma do INMET. Falhas de observação podem afetar totais de chuva, e a quantidade de meses efetivamente avaliados é menor do que o número de meses do calendário.

## 3. Modelo e avaliações

A arquitetura principal utiliza duas camadas LSTM (64 e 32 unidades), Dropout(0,2), Dense(16) e saída linear. O treinamento usa Adam, MSE, até 80 épocas, batch size 8 e parada antecipada pela perda de validação. A divisão principal é cronológica (70% treino, 15% validação e 15% teste, antes do descarte de janelas incompletas).

As referências são:
- **Sazonal ingênua:** precipitação do mesmo mês do ano anterior;
- **Climatologia mensal:** média histórica do respectivo mês, calculada com dados anteriores ao período avaliado.

Também foram realizadas avaliações com cinco sementes, comparação exploratória de quatro arquiteturas na validação e validação temporal em períodos iniciados em 2016, 2018 e 2020.

## 4. Resultados observados

### 4.1. Teste cronológico principal

O conjunto de teste efetivamente utilizado contém **19 meses**.

| Método | MAE (mm) | RMSE (mm) |
| --- | ---: | ---: |
| Climatologia mensal | **60,84** | **82,25** |
| LSTM principal, semente 42 | 64,59 | 89,44 |
| Referência sazonal | 89,84 | 127,03 |

A LSTM supera a referência sazonal, mas não a climatologia mensal neste teste.

### 4.2. Estabilidade da arquitetura principal

Nas cinco sementes (42, 123, 456, 789 e 2026), o **MAE médio da LSTM foi 65,86 mm**, com desvio-padrão de aproximadamente **4,69 mm**. A inicialização altera os resultados; não se deve escolher a melhor semente observando o erro no teste.

### 4.3. Comparação de arquiteturas na validação

| Arquitetura | MAE médio de validação (mm) | RMSE médio de validação (mm) |
| --- | ---: | ---: |
| Modelo A — LSTM(32) | **39,11** | 74,88 |
| Modelo C — LSTM(16) | 39,42 | 76,93 |
| Arquitetura principal | 39,49 | **69,93** |
| Modelo B — LSTM(32) e Dropout | 40,11 | 76,82 |

Os valores são médias de três sementes. O Modelo A apresentou menor MAE, mas a diferença para a arquitetura principal foi pequena; esta teve o menor RMSE.

### 4.4. Validação temporal exploratória — comparação com corte histórico alinhado

A climatologia foi recalculada utilizando somente dados anteriores à janela de validação da LSTM. A comparação inclui períodos iniciados em 2016, 2018 e 2020.

| Método | MAE médio (mm) | RMSE médio (mm) |
| --- | ---: | ---: |
| Climatologia mensal | **42,83** | **63,65** |
| Modelo A — LSTM(32) | 56,42 | 78,17 |
| Arquitetura principal | 57,78 | 81,31 |

As médias resumem três períodos da climatologia e nove execuções por arquitetura LSTM (três sementes em cada período). **Não são médias ponderadas por mês**; a quantidade de meses válidos varia entre períodos. Os resultados favorecem a climatologia, mas o conjunto de janelas ainda é pequeno.

## 5. Discussão e conclusão

As LSTMs aprenderam parte da sazonalidade das chuvas e melhoraram em relação à referência sazonal ingênua no teste principal. Contudo, a climatologia mensal obteve menor erro nos experimentos consolidados. Modelos recorrentes menores não demonstraram uma vantagem consistente que justificasse substituir a arquitetura principal.

O resultado é cientificamente informativo: para os dados e o protocolo utilizados, maior complexidade não implicou melhor previsão. O trabalho demonstra a importância de baselines, separação temporal, controle de dados ausentes e múltiplas avaliações.

## 6. Limitações e trabalhos futuros

- Cobertura e qualidade dos registros INMET, especialmente meses incompletos;
- Apenas 19 meses válidos no teste principal e poucos períodos na validação temporal;
- Comparação exploratória de arquiteturas, sem avaliação independente adicional da arquitetura escolhida;
- Possível sensibilidade à inicialização, ao período histórico e a eventos extremos;
- Investigar critérios de cobertura, mais janelas temporais e outras variáveis preditoras antes de alegar ganho de generalização.

## 7. Reproduzir

Instale as dependências de `requirements.txt`, coloque os CSVs originais em `GYN_DADOS/` e execute:

```powershell
python MainProj.py
python validacao_temporal.py
```

`MainProj.py` gera a série mensal e os relatórios do teste e dos experimentos. `validacao_temporal.py` lê a série mensal já gerada e produz `validacao_temporal_detalhada.csv` e `validacao_temporal_resumo.csv`.

Os arquivos em `resultados/` são sobrescritos localmente e ignorados pelo Git. Este documento registra as métricas de uma execução analisada; **não** incorpora os CSVs ou PNGs produzidos pelo programa ao repositório.

## 8. Integridade da implementação

Este relatório documenta o código existente. Nenhuma arquitetura, rotina de treinamento, divisão temporal ou regra de pré-processamento foi alterada para sua elaboração.

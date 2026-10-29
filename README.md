# Previsão de precipitação mensal em Goiânia com LSTM

Projeto de iniciação científica voltado à previsão da **precipitação mensal em Goiânia (GO)** por meio de uma rede neural recorrente do tipo **Long Short-Term Memory (LSTM)**. O código utiliza registros meteorológicos históricos do **INMET**, organizados em arquivos CSV, para preparar séries temporais, treinar o modelo e comparar valores previstos e observados.

## Objetivo

Investigar o uso de redes neurais LSTM na previsão de precipitação mensal, considerando tanto a chuva registrada quanto outras variáveis meteorológicas. O projeto tem caráter acadêmico e experimental; as previsões não devem ser interpretadas como um serviço operacional de meteorologia.

## Tecnologias

- Python
- Pandas e NumPy — tratamento e organização dos dados
- TensorFlow/Keras — implementação e treinamento da rede LSTM
- Scikit-learn — normalização e métricas de avaliação
- Matplotlib — visualização dos resultados

## Dados e pré-processamento

O script lê os arquivos `.CSV` ou `.csv` da pasta `GYN_DADOS/`. Os arquivos são interpretados com separador `;`, codificação `latin1` e oito linhas iniciais ignoradas, conforme o formato utilizado no projeto.

As variáveis selecionadas são:

- Precipitação
- Temperatura do ar
- Umidade relativa do ar
- Pressão atmosférica ao nível da estação
- Velocidade do vento

O processamento inclui padronização dos nomes das colunas, interpretação de datas e horas, conversão de valores numéricos, remoção de registros inválidos ou incompletos e ordenação cronológica. Em seguida, os dados são agregados mensalmente: **soma** da precipitação e **média** das demais variáveis.

O mês do ano também é representado por duas variáveis cíclicas (`mes_sin` e `mes_cos`), para auxiliar a representação da sazonalidade.

## Modelo

A implementação atual, disponível em `MainProj.py`, utiliza:

| Configuração | Valor |
| --- | --- |
| Tipo de modelo | LSTM multivariada |
| Janela de entrada | 12 meses |
| Primeira camada LSTM | 64 unidades |
| Segunda camada LSTM | 32 unidades |
| Regularização | Dropout de 0,2 |
| Camada intermediária | Dense com 16 unidades e ReLU |
| Saída | Dense com 1 unidade e ReLU |
| Otimizador | Adam |
| Função de perda | MSE |
| Épocas | 80 |
| Tamanho do lote | 8 |
| Divisão temporal | 70% treino, 15% validação e 15% teste |
| Semente aleatória | 42 |

O alvo é a **precipitação mensal do mês seguinte à janela de entrada**. A divisão das sequências preserva a ordem cronológica, sem embaralhamento explícito dos conjuntos.

## Estrutura do repositório

```text
.
├── GYN_DADOS/           # Arquivos meteorológicos de entrada
├── Documentação/
│   ├── Instalacoes.txt  # Dependências utilizadas
│   └── Rodarprojeto.txt # Comando de execução
├── MainProj.py          # Pré-processamento, treinamento e avaliação
├── .gitignore
└── README.md
```

## Como executar

**1.** Clone o repositório:

```bash
git clone https://github.com/Joao-Gabriel-G-Costa/previsao-precipitacao-lstm-goiania.git
cd previsao-precipitacao-lstm-goiania
```

**2.** Crie e ative um ambiente virtual (recomendado):

```bash
python -m venv .venv
```

No Windows:

```powershell
.venv\Scripts\activate
```

No Linux/macOS:

```bash
source .venv/bin/activate
```

**3.** Instale as bibliotecas utilizadas:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
```

**4.** Confira se os arquivos CSV de entrada estão na pasta `GYN_DADOS/` e execute:

```bash
python MainProj.py
```

> A execução depende de arquivos de entrada compatíveis com o formato esperado pelo script e de um ambiente Python com as bibliotecas instaladas.

## Saídas e avaliação

Ao finalizar o treinamento, o script:

- Exibe as métricas **MAE**, **MSE** e **RMSE** no conjunto de teste;
- Gera um gráfico comparando a precipitação mensal **real e prevista**;
- Exibe as curvas de perda de **treinamento e validação**;
- Exporta a série mensal processada para `precipitacao_mensal_goiania_multivariado.csv`.

As métricas são calculadas após a transformação inversa dos valores de precipitação para a escala original, em milímetros.

**Observação:** não há neste README um valor fixo de desempenho porque ele precisa ser reproduzido e conferido com a versão atual do código e dos dados.

## Limitações e melhorias previstas

**Atenção à avaliação:** na implementação atual, o `MinMaxScaler` é ajustado sobre a série completa **antes** da separação entre treino, validação e teste. Isso pode introduzir vazamento de informações dos períodos posteriores e influenciar as métricas. Uma próxima revisão deverá ajustar a normalização apenas com os dados de treinamento e aplicar a transformação aos demais períodos.

Outras melhorias previstas:

- Registrar versões das dependências para facilitar a reprodução;
- Revisar o tratamento de meses com observações ausentes ou incompletas;
- Comparar a LSTM com modelos de referência mais simples;
- Salvar os gráficos e documentar resultados reproduzidos;
- Organizar o código em etapas de preparação, treinamento e avaliação.

## Autor

**João Gabriel Gonçalves Costa**

Estudante de Engenharia de Software — SENAI FATESG.

[GitHub](https://github.com/Joao-Gabriel-G-Costa) · [LinkedIn](https://www.linkedin.com/in/jo%C3%A3o-gabriel-gon%C3%A7alves-costa-29b908309)

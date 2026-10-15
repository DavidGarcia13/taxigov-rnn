# Previsão de Demanda de Corridas TaxiGOV com Redes Neurais Recorrentes

Previsão da demanda **diária** de corridas do [TaxiGOV](https://www.gov.br/) no
Distrito Federal (DF) usando redes neurais recorrentes. O projeto compara três
famílias de arquiteturas — **SimpleRNN**, **GRU** e **LSTM** — sob um protocolo
equivalente, com busca de hiperparâmetros via Optuna e baselines de referência.

## Objetivo

Modelar a série temporal de solicitações de corridas (todas as situações) e
prever a demanda do próximo dia (horizonte de 1 passo), avaliando qual
arquitetura recorrente oferece o melhor compromisso entre acurácia e custo
computacional.

## Estrutura do projeto

```
.
├── src/                        # código-fonte
│   ├── unificar_datasets.py    # unifica os CSVs anuais num único arquivo canônico
│   ├── preprocess.py           # agrega por dia, cria features e janelas (gera o .npz)
│   ├── eda.py                  # análise exploratória (estatísticas + figuras)
│   ├── models.py               # arquiteturas (SimpleRNN/GRU/LSTM), baselines e métricas
│   ├── tune.py                 # busca de hiperparâmetros com Optuna
│   ├── window_experiment.py    # efeito do tamanho da janela (lookback)
│   └── evaluate.py             # avaliação final das 3 arquiteturas + figuras
├── data/
│   ├── raw/                    # CSVs brutos do TaxiGOV (NÃO versionados — ver abaixo)
│   └── processed/              # série diária, dataset .npz e parâmetros do scaler
├── results/                    # métricas, melhores hiperparâmetros e figuras
│   └── figuras/                # gráficos de EDA, curvas de perda e previsto × real
├── docs/                       # especificações do pacote de dados (datapackage.yaml)
├── requirements.txt
└── README.md
```

## Dados

Os dados brutos (CSVs anuais de 2017 a 2024 e o consolidado) são grandes
(centenas de MB cada) e **não são versionados no Git**. Baixe os conjuntos de
corridas do TaxiGOV no portal de dados abertos do Governo Federal e coloque-os
em `data/raw/`, seguindo o layout esperado:

```
data/raw/
├── taxigov-corridas-2017/taxigov-corridas-2017.csv
├── taxigov-corridas-2018/taxigov-corridas-2018.csv
├── ...
├── taxigov-corridas-2024/taxigov-corridas-2024.csv
└── taxigov-corridas-completo2.csv
```

A especificação de colunas está em `docs/datapackage.yaml`.

## Como reproduzir

Pré-requisitos: Python 3.11.

```bash
# 1. ambiente
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt

# 2. pipeline (a partir da raiz do projeto)
python src/unificar_datasets.py     # une os CSVs brutos -> data/raw/taxigov-corridas-unificado.csv
python src/preprocess.py            # gera a série diária e o dataset de janelas (.npz)
python src/eda.py                   # análise exploratória -> results/figuras/eda/

python src/tune.py --todas --trials 25   # busca de hiperparâmetros (3 arquiteturas)
python src/window_experiment.py          # efeito do tamanho da janela
python src/evaluate.py                    # avaliação final + figuras comparativas
```

## Decisões metodológicas

- **Granularidade diária** e escopo no **DF** (série mais densa e consistente,
  ~87% dos dados).
- **Split cronológico** 70/15/15 (treino/validação/teste), sem embaralhar, com
  embargo do tamanho da janela entre partições para evitar vazamento.
- **Normalização** ajustada apenas no treino.
- **Features**: calendário cíclico (dia da semana, mês, dia do ano), feriados do
  DF, lags do alvo (1, 7, 14, 28), média móvel de 7 dias, exógenas sempre
  defasadas (anti-leakage) e flag de regime de pandemia.
- **Janela (lookback) de 7 dias**: a sazonalidade semanal domina (ACF no lag 7
  ≈ 0,80); janelas maiores pioram o desempenho e aumentam custo/overfitting
  (ver `window_experiment.py`).
- **Baselines**: naive sazonal (demanda de 7 dias atrás) e média móvel de 7 dias.

## Resultados

As métricas finais (MAE, RMSE, R²), os custos (parâmetros, tempo de treino e de
inferência) e os melhores hiperparâmetros ficam em `results/`:

- `results/metrics_final.json` — comparação das 3 arquiteturas vs. baselines
- `results/best_params_<ARCH>.json` — melhores hiperparâmetros por arquitetura
- `results/figuras/` — curvas de perda, previsto × real e gráficos da EDA

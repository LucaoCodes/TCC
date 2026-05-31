# Modelos Explicativos com Inteligência Artificial Aplicados à Economia e Finanças

Trabalho de Conclusão de Curso — Bacharelado em Ciência da Computação  
**Autor:** Lucas de Oliveira Freitas  
**Orientador:** Prof. Dr. Renato Cesar Sato  
**Instituição:** Universidade Federal de São Paulo — UNIFESP / Instituto de Ciência e Tecnologia  

---

## Sobre o projeto

Este TCC investiga os fatores macroeconômicos que explicam o comportamento do mercado de ações brasileiro, com foco no Ibovespa, por meio de modelos estatísticos e de aprendizado de máquina interpretável.

O trabalho parte do argumento de que interpretabilidade não é um diferencial técnico, mas uma exigência ética e regulatória em decisões de alto risco no mercado financeiro. Seguindo a linha de Rudin (2019), o problema da caixa-preta é atacado na raiz — por meio de arquiteturas inerentemente transparentes, e não por explicações post hoc sobre modelos opacos.

O contexto de economia emergente torna o problema ainda mais relevante: o Brasil apresenta dinâmicas próprias de câmbio, risco-país e resposta a choques que exigem modelos que possam ser auditados e compreendidos, não apenas acurados.

---

## Variáveis utilizadas

| Variável | Fonte | Transformação |
|---|---|---|
| Ibovespa (`^BVSP`) | Yahoo Finance | Retorno logarítmico |
| Taxa Selic | BCB / SGS (id: 11) | Primeira diferença |
| Câmbio (USD/BRL) | BCB / SGS (id: 1) | Primeira diferença |
| IPCA | BCB / SGS (id: 433) | Nenhuma (já estacionária) |
| PIB | BCB / SGS (id: 4380) | Retorno logarítmico |
| Expectativa IPCA 12m | BCB / Expectativas | Primeira diferença |
| Risco-País (EMBI+) | IPEA (`JPM366_EMBI366`) | Primeira diferença |

Período de análise: **novembro de 2001 a julho de 2024** (dias úteis). Split treino/teste em `2020-01-01` (treino: 2001–2019; teste: 2020–2024).

---

## Modelos

| Modelo | Status | R² |
|---|---|---|
| ARIMA(4,0,3) | Implementado | — (univariado) |
| OLS / MQO | Implementado | 0,209 (in-sample) |
| GAM (Generalized Additive Model) | Implementado | 0,3045 (out-of-sample) |
| BSTS (Bayesian Structural Time Series) | Implementado | 0,2436 (out-of-sample) |
| Causal Forest | Implementado | 0,2321 (out-of-sample) |

> Os valores comparativos definitivos (métricas out-of-sample sobre o split 2020–2024) estão na tabela de síntese da monografia (`tab:metricas_split`), que é a fonte autoritativa.

---

## Estrutura do repositório

```
notebooks/
  01_coleta_tratamento.ipynb          # Coleta via APIs (BCB, IPEA, yfinance)
  02_modelos_base_tcc1.ipynb          # ARIMA + OLS (linha de base)
  03_eda_pre_tcc2.ipynb               # Análise exploratória
  04_gam.ipynb                        # GAM com tuning
  05_bsts.ipynb                       # BSTS (orbit-ml)
  06_causal_forest.ipynb              # Causal Forest (econml)
  07_analise_comparativa.ipynb        # Síntese comparativa de todos os modelos
src/
  tcc_utils.py                        # Paths, constantes e helpers compartilhados
data/
  br_transformado.csv                 # Série diária transformada (fonte única)
  metricas_comparativo.csv            # Acumulador de métricas (gerado pelos notebooks)
Overleaf Latex/
  monografia_bcc.tex                  # Documento LaTeX (ABNT / abnTeX2)
  figs/                               # Figuras geradas pelos notebooks
```

---

## Como executar

Instale todas as dependências de uma vez:

```bash
pip install -r requirements.txt
```

Ordem de execução: `01` (apenas ao atualizar dados) → `02` → `03` → `04` → `05` → `06` → `07`. Os notebooks 02–07 leem CSVs de `data/` e não dependem de APIs externas.

---

## Referências principais

- RUDIN, C. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 2019.
- ZHANG et al. Impact of macroeconomic variables on financial volatility, 2025.
- QIU et al. Multivariate BSTS, 2020.
- GULEN et al. Balancing with Causal Forest, 2024.

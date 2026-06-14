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

Período de análise: **novembro de 2001 a julho de 2024** (dias úteis).  
Split treino/teste em `2020-01-01` (treino: 2001–2019; teste: 2020–2024).

---

## Modelos

Todos os R² abaixo são **out-of-sample** sobre o período de teste 2020–2024, exceto OLS (in-sample e out-of-sample reportados por comparabilidade). Os valores autoritativos estão na tabela `tab:metricas_split` da monografia.

| Modelo | Status | R² (treino) | R² (teste OOS) | MAE (teste) |
|---|---|---|---|---|
| ARIMA(4,0,3) | Implementado | — (univariado) | — | 0,01223 |
| OLS / MQO | Implementado | 0,2086 | 0,2339 | 0,01035 |
| GAM (Generalized Additive Model) | Implementado | 0,2946 | 0,3045 | 0,00999 |
| BSTS (Bayesian Structural Time Series) | Implementado | — | 0,2436 | 0,01015 |
| Causal Forest (CausalForestDML) | Implementado | — | 0,2321 | 0,01051 |

**Melhor modelo preditivo OOS:** GAM (R² = 0,3045).  
**Causal Forest:** destaque para análise causal — ATE ≈ −0,000086 (não significativo); heterogeneidade de efeito (GATES) e análise BLP incluídas.

---

## Estrutura do repositório

```
notebooks/
  01_coleta_tratamento.ipynb          # Coleta via APIs (BCB, IPEA, yfinance)
  02_modelos_base_tcc1.ipynb          # ARIMA + OLS — linha de base (intocável)
  03_eda_pre_tcc2.ipynb               # Análise exploratória pré-modelos
  04_gam.ipynb                        # GAM com tuning de n_splines + PDPs
  05_bsts.ipynb                       # BSTS via orbit-ml (MCMC pesado)
  06_causal_forest.ipynb              # CausalForestDML (econml) — ATE/CATE/GATES/BLP
  07_analise_comparativa.ipynb        # Síntese comparativa — gera corpo LaTeX de tab:metricas_split
src/
  tcc_utils.py                        # Paths absolutos, constantes e helpers compartilhados
  regenera_bsts_figs.py               # Regenera figuras BSTS sem re-executar MCMC
data/
  br_transformado.csv                 # Série diária transformada (fonte única)
  bsts_train.csv / bsts_test.csv      # Splits pré-formatados para o BSTS
  causal_forest_train.csv / ...       # Splits pré-formatados para o Causal Forest
  metricas_comparativo.csv            # Acumulador de métricas (gerado pelos notebooks 02–06)
Overleaf Latex/
  monografia_bcc.tex                  # Documento LaTeX (ABNT / abnTeX2)
  references.bib                      # Referências bibliográficas
  figs/                               # Figuras geradas pelos notebooks (únicas usadas no .tex)
legado/
  TCC_original.ipynb                  # Notebook monolítico original — congelado para auditoria
```

---

## Como executar

Instale todas as dependências de uma vez:

```bash
pip install -r requirements.txt
```

**Ordem de execução:** `01` (apenas ao atualizar dados) → `02` → `03` → `04` → `05` → `06` → `07`.  
Os notebooks 02–07 leem CSVs de `data/` e não dependem de APIs externas.

> **Atenção:** o notebook `05_bsts.ipynb` executa amostragem MCMC e pode levar 20–60 minutos. O script `src/regenera_bsts_figs.py` regenera as figuras a partir do modelo salvo (`bsts_final_model.pkl`) sem re-executar o MCMC.

---

## Referências principais

- RUDIN, C. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 2019.
- ATHEY, S.; TIBSHIRANI, J.; WAGER, S. Generalized random forests. *Annals of Statistics*, 2019.
- WAGER, S.; ATHEY, S. Estimation and inference of heterogeneous treatment effects using random forests. *Journal of the American Statistical Association*, 2018.
- ZHANG et al. Impact of macroeconomic variables on financial volatility, 2025.
- QIU et al. Multivariate BSTS, 2020.
- GULEN et al. Balancing with Causal Forest, 2024.

---

## Nota sobre ferramentas de IA

O assistente Claude (Anthropic) foi utilizado exclusivamente para auxiliar na estruturação de mensagens de commit e pull requests no Git — facilitando a adoção consistente das convenções Conventional Commits definidas no projeto. Nenhum código de modelagem, análise estatística ou texto da monografia foi gerado por IA.

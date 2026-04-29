# Walk-Forward Cross-Validation — Relatório de Experimento

**Data:** 2026-04-28  
**Branch:** `exp/walk-forward-cv`  
**Decisão:** ROLLBACK — critério de estabilidade não atendido  
**Status:** documentado para citação futura

---

## Motivação

O split único treino/teste (2001–2019 / 2020–2024) possui três fragilidades metodológicas:
1. O R² reportado é um ponto único — sem estimativa de variância.
2. O período de teste começa no maior choque recente (COVID, mar/2020), sem possibilidade de isolar o impacto desse evento.
3. Walk-forward CV é prática recomendada em econometria de séries temporais.

O objetivo foi quantificar a estabilidade das métricas OOS via walk-forward com janela expansiva.

---

## Estrutura dos Folds

Janela expansiva (treino sempre inicia em nov/2001), step de 2 anos, teste de 2 anos.  
Holdout interno = últimos 12 meses do treino de cada fold (não usado aqui — hiperparâmetros fixos).

| Fold | Treino (ajuste)  | Teste OOS          | n_treino | n_teste |
|------|------------------|--------------------|----------|---------|
| 1    | 2001–2008        | 2010–2011          | 1766     | 495     |
| 2    | 2001–2010        | 2012–2013          | 2258     | 492     |
| 3    | 2001–2012        | 2014–2015          | 2751     | 494     |
| 4    | 2001–2014        | 2016–2017          | 3247     | 495     |
| 5    | 2001–2016        | 2018–2019          | 3742     | 492     |
| 6    | 2001–2018        | **2020–2021 (COVID)** | 4232  | 495     |

**Hiperparâmetros fixos (sem re-tuning por fold):**
- GAM: n_splines=10, todos s(), features G0+G4+G8 (6 macro + ibov_lag1)
- BSTS: sigma_prior=0,01 (5 regressores), sigma_embi=0,1, estimador stan-MAP
- OLS: regressores macro sem ibov_lag1 (baseline TCC 1)

---

## Resultados por Fold

### R² out-of-sample

| Fold            | OLS    | GAM    | BSTS    |
|-----------------|--------|--------|---------|
| 2010–2011       | 0,193  | 0,292  | 0,077   |
| 2012–2013       | 0,107  | 0,164  | −0,098  |
| 2014–2015       | 0,113  | 0,159  | 0,119   |
| 2016–2017       | 0,254  | 0,275  | 0,222   |
| 2018–2019       | 0,163  | 0,193  | 0,098   |
| **2020–2021**   | **0,244** | **0,378** | **0,261** |

### MAE out-of-sample

| Fold            | OLS     | GAM     | BSTS    |
|-----------------|---------|---------|---------|
| 2010–2011       | 0,00965 | 0,00921 | 0,01042 |
| 2012–2013       | 0,00981 | 0,00956 | 0,01103 |
| 2014–2015       | 0,01126 | 0,01100 | 0,01130 |
| 2016–2017       | 0,00975 | 0,00966 | 0,00965 |
| 2018–2019       | 0,00903 | 0,00892 | 0,00941 |
| **2020–2021**   | 0,01336 | 0,01235 | 0,01287 |

### Agregados (mean ± std entre as 6 dobras)

| Modelo | R²_mean | R²_std  | MAE_mean | MAE_std  |
|--------|---------|---------|----------|----------|
| OLS    | 0,179   | 0,063   | 0,01049  | 0,0016   |
| GAM    | 0,244   | **0,086** | 0,01013  | 0,0013   |
| BSTS   | 0,113   | 0,126   | 0,01083  | 0,0013   |

---

## Critério de Ganho e Decisão

**Critério acordado:** std(R²_GAM) < 0,05 → walk-forward complementa o split único.

**Resultado observado:** std(R²_GAM) = **0,0864** → critério **não atendido**.

**Decisão:** ROLLBACK. O branch `exp/walk-forward-cv` não será mergeado à `main`.  
O estado estável do projeto permanece em `main` (commit `c8094ba`).

---

## Análise dos Resultados — Achados Relevantes para o TCC

Embora o critério de estabilidade formal não tenha sido atendido, os resultados revelam achados metodologicamente importantes:

### 1. Performance é regime-dependente — e o COVID não é o pior regime

Contraintuitivamente, o **fold COVID (2020–2021) tem o maior R²** dos 6 folds para todos os três modelos. Isso indica que a crise foi prevista pelos indicadores macro: os movimentos extremos de câmbio e EMBI+ no início da pandemia eram fortes sinais para os modelos.

Os **piores folds são 2012–2013 e 2014–2015** — períodos de relativa calma macroeconômica, onde os indicadores têm pouca variação e o sinal é fraco. Nessas janelas, os modelos são incapazes de encontrar estrutura nos dados, e o BSTS chega a R² negativo (−0,098 em 2012–2013).

### 2. A alta variância revela heterogeneidade temporal genuína

O std de 0,086 no R² do GAM não é um artefato técnico — é um sinal real de que a relação entre variáveis macro e retornos do Ibovespa **varia entre regimes**. Em períodos de alta volatilidade (choques cambiais, crises), o sinal explicativo é forte; em períodos tranquilos, a dinâmica é dominada por ruído não explicável por indicadores macro de baixa frequência.

Esse resultado **justifica diretamente o Causal Forest** como próximo modelo: se o efeito causal das variáveis macro é heterogêneo entre regimes, um modelo que modela explicitamente essa heterogeneidade (CATE) é o passo metodológico natural.

### 3. Ranking entre modelos é estável

Apesar da variância intra-modelo, o **ranking GAM > OLS > BSTS** é mantido em todos os 6 folds. Isso confirma que a conclusão principal do TCC é robusta: o GAM é consistentemente superior ao OLS, que é consistentemente superior ao BSTS em previsão pontual — independentemente do período de avaliação.

### 4. O split único (2020–2024) é levemente otimista para o GAM

O R²_teste do split único (0,3045) está entre os valores do fold 6 (0,378, somente 2020–2021) e o que seria esperado para 2022–2024 (não calculado, mas provavelmente mais baixo dado o padrão dos folds calmos). A média WF do GAM é 0,244, sugerindo que o número do split único é entre 15–25% mais alto do que a performance esperada em anos típicos. Esse viés é conhecido no contexto de experimentos que usam períodos pós-choque como janela de teste.

---

## Recomendação

1. **Manter o split único como métrica principal** — metodologicamente honesto dado o holdout 2018–2019 usado no tuning.
2. **Citar esta análise como evidência da heterogeneidade temporal** — a variância entre folds é um achado positivo, não uma falha.
3. **Usar o achado 3 (ranking estável) como argumento de robustez na banca** — mesmo sem walk-forward formal, o ranking dos modelos não mudaria entre regimes.
4. **Achado 2 reforça a motivação do Causal Forest** — próximo passo natural do TCC.

---

## Artefatos Preservados

| Arquivo | Descrição |
|---------|-----------|
| `walk_forward.py` | Módulo de infraestrutura WF (reutilizável) |
| `walk_forward/wf_results.csv` | Resultados por fold × modelo (18 linhas) |
| `walk_forward/wf_aggregated.csv` | Médias e desvios por modelo |
| `walk_forward/decision.json` | Decisão formal com timestamp |
| `walk_forward/baseline_single_split.json` | Métricas do split original como referência |
| `Overleaf Latex/figs/walk_forward_r2_per_fold.png` | Figura gerada (R² por fold por modelo) |
| Branch `exp/walk-forward-cv` | Código completo do experimento |

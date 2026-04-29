"""
Regenera bsts_previsao_oos.png e bsts_decomposicao.png com o modelo final B0+B7+MAP.
REGRESSORS_SELECTED confirmados pela analise de selecao de variaveis (celula 76):
  - br_selic_diff, br_dolar_diff, br_ipca, br_pib_ret_log, ipca_expectativa_diff (P=1.0)
  - embi_brasil_diff REMOVIDA pela selecao (P=0.59), mas RE-ADICIONADA pelo B7
B0: sigma_opt=0.01 (holdout 2018-2019)
B7: embi_brasil_diff com sigma=0.1 (10x sigma_opt)
Estimador: stan-map
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shutil, os

# --- Dados ---
bsts_train_raw = pd.read_csv('bsts_train.csv', index_col=0, parse_dates=True)
bsts_test_raw  = pd.read_csv('bsts_test.csv',  index_col=0, parse_dates=True)

TARGET   = 'br_ibov_ret_log'
DATE_COL = 'date'

def to_orbit_df(df):
    d = df.reset_index().rename(columns={'index': DATE_COL})
    d[DATE_COL] = pd.to_datetime(d[DATE_COL]).dt.strftime('%Y-%m-%d')
    return d

train_df = to_orbit_df(bsts_train_raw)
test_df  = to_orbit_df(bsts_test_raw)

# --- Configuracao final B0+B7 ---
REGRESSORS_SELECTED = ['br_selic_diff', 'br_dolar_diff', 'br_ipca', 'br_pib_ret_log', 'ipca_expectativa_diff']
SIGMA_OPT  = 0.01
SIGMA_EMBI = SIGMA_OPT * 10   # 0.1 — prior heterogeneo B7
REGRESSORS_FINAL   = REGRESSORS_SELECTED + ['embi_brasil_diff']
SIGMA_PRIOR_FINAL  = [SIGMA_OPT] * len(REGRESSORS_SELECTED) + [SIGMA_EMBI]

print('Ajustando BSTS final (B0 + B7, stan-map)...')
print(f'  Variaveis ({len(REGRESSORS_FINAL)}): {REGRESSORS_FINAL}')
print(f'  sigma_prior: {SIGMA_PRIOR_FINAL}')

from orbit.models import DLT

dlt_final = DLT(
    response_col=TARGET, date_col=DATE_COL,
    regressor_col=REGRESSORS_FINAL,
    seasonality=5, seed=42,
    estimator='stan-map',
    regressor_sigma_prior=SIGMA_PRIOR_FINAL,
)
dlt_final.fit(df=train_df)
print('Modelo ajustado.')

# --- Metricas OOS ---
pred_test = dlt_final.predict(df=test_df)
y_true    = test_df[TARGET].values
y_pred    = pred_test['prediction'].values

mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2   = r2_score(y_true, y_pred)
print(f'MAE={mae:.6f}  RMSE={rmse:.6f}  R2={r2:.4f}')

def _get_ci(df, *keys):
    for k in keys:
        if k in df.columns:
            return df[k].values
    return None

dates_test = pd.to_datetime(test_df[DATE_COL])
ci_low90  = _get_ci(pred_test, 'prediction_5',  'lower_5',  'lower_10')
ci_high90 = _get_ci(pred_test, 'prediction_95', 'upper_95', 'upper_90')
ci_low60  = _get_ci(pred_test, 'prediction_20', 'lower_20')
ci_high60 = _get_ci(pred_test, 'prediction_80', 'upper_80')

# --- Figura 1: Previsao OOS ---
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(dates_test, y_true, color='black',     lw=0.8, alpha=0.7, label='Retorno real')
ax.plot(dates_test, y_pred, color='steelblue', lw=1.2,            label='BSTS — previsto')
if ci_low90 is not None and ci_high90 is not None:
    ax.fill_between(dates_test, ci_low90, ci_high90, alpha=0.20, color='steelblue', label='IC 90%')
if ci_low60 is not None and ci_high60 is not None:
    ax.fill_between(dates_test, ci_low60, ci_high60, alpha=0.35, color='steelblue', label='IC 60%')
ax.axhline(0, color='gray', lw=0.5, ls='--')
ax.set_title(f'BSTS — Previsao Out-of-Sample (2020-2024)  |  MAE={mae:.6f}  RMSE={rmse:.6f}')
ax.set_xlabel('Data')
ax.set_ylabel('Retorno log do Ibovespa')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('Overleaf Latex/figs/bsts_previsao_oos.png', dpi=150, bbox_inches='tight')
plt.close()
print('Salvo: Overleaf Latex/figs/bsts_previsao_oos.png')

# --- Figura 2: Decomposicao ---
try:
    pred_decomp = dlt_final.predict(df=test_df, decompose=True)
    main_comps  = ['trend', 'seasonality', 'regression', 'leveling']
    comp_cols   = [c for c in pred_decomp.columns if c in main_comps]
    print(f'Componentes disponiveis: {pred_decomp.columns.tolist()}')
    print(f'Componentes plotados:    {comp_cols}')

    n_comp = len(comp_cols)
    fig = plt.figure(figsize=(14, 3 * (n_comp + 1)))
    gs  = gridspec.GridSpec(n_comp + 1, 1, hspace=0.50)

    ax0 = fig.add_subplot(gs[0])
    ax0.plot(dates_test, y_true, color='black',     lw=0.8, alpha=0.7, label='Real')
    ax0.plot(dates_test, y_pred, color='steelblue', lw=1.2, label='Previsto')
    ax0.axhline(0, color='gray', lw=0.5, ls='--')
    ax0.set_title('Retorno Real vs BSTS Previsto')
    ax0.legend(fontsize=8)

    palette = plt.cm.tab10.colors
    comp_labels = {
        'trend':       'Tendencia (trend) — direcao de longo prazo',
        'regression':  'Componente de Regressao — efeito das variaveis macro',
        'seasonality': 'Sazonalidade — padrao semanal (5 pregoes)',
        'leveling':    'Nivel local',
    }
    for k, comp in enumerate(comp_cols):
        ax = fig.add_subplot(gs[k + 1])
        ci_lo = pred_decomp.get(f'{comp}_5')
        ci_hi = pred_decomp.get(f'{comp}_95')
        vals  = pred_decomp[comp].values
        ax.plot(dates_test, vals, color=palette[k % 10], lw=1.0)
        if ci_lo is not None and ci_hi is not None:
            ax.fill_between(dates_test, ci_lo.values, ci_hi.values, alpha=0.15, color=palette[k % 10])
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.set_title(comp_labels.get(comp, comp))

    fig.suptitle(f'BSTS — Decomposicao Estrutural dos Componentes (2020-2024)', y=1.01, fontsize=13)
    plt.savefig('Overleaf Latex/figs/bsts_decomposicao.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Salvo: Overleaf Latex/figs/bsts_decomposicao.png')
except Exception as e:
    print(f'Erro na decomposicao: {e}')

print('Concluido.')

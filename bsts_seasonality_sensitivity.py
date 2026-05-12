"""
Analise de sensibilidade da sazonalidade do BSTS.
Testa seasonality in {5, 10, 21, 22} com configuracao B0+B7+MAP (celula 78 do notebook).

Produz:
  - bsts_seasonality_results.csv
  - Overleaf Latex/figs/bsts_seasonality_comparison.png
  - Overleaf Latex/figs/bsts_seasonality_diagnostico.png

Configuracao espelha regenera_bsts_figs.py: mesmos regressores, sigmas, estimador e seed.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.stattools import acf, pacf
from scipy.signal import periodogram

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

# --- Configuracao final B0+B7 (identica ao notebook celula 78) ---
REGRESSORS_SELECTED = ['br_selic_diff', 'br_dolar_diff', 'br_ipca', 'br_pib_ret_log', 'ipca_expectativa_diff']
SIGMA_OPT  = 0.01
SIGMA_EMBI = SIGMA_OPT * 10
REGRESSORS_FINAL  = REGRESSORS_SELECTED + ['embi_brasil_diff']
SIGMA_PRIOR_FINAL = [SIGMA_OPT] * len(REGRESSORS_SELECTED) + [SIGMA_EMBI]

SEASONALITY_GRID = [5, 10, 21, 22]

from orbit.models import DLT

# ============================================================
# 1. Loop de experimentos
# ============================================================
results = []

for s in SEASONALITY_GRID:
    print(f'\n--- seasonality={s} ---')

    dlt = DLT(
        response_col=TARGET, date_col=DATE_COL,
        regressor_col=REGRESSORS_FINAL,
        seasonality=s, seed=42,
        estimator='stan-map',
        regressor_sigma_prior=SIGMA_PRIOR_FINAL,
    )
    dlt.fit(df=train_df)

    # In-sample
    pred_train   = dlt.predict(df=train_df)
    y_true_train = train_df[TARGET].values
    y_pred_train = pred_train['prediction'].values

    # OOS
    pred_test   = dlt.predict(df=test_df)
    y_true_test = test_df[TARGET].values
    y_pred_test = pred_test['prediction'].values

    # Amplitude do componente sazonal (std no periodo de teste)
    pred_decomp = dlt.predict(df=test_df, decompose=True)
    if 'seasonality' in pred_decomp.columns:
        amp_seasonal = float(pred_decomp['seasonality'].std())
    else:
        amp_seasonal = float('nan')

    row = {
        'seasonality':       s,
        'mae_train':         mean_absolute_error(y_true_train, y_pred_train),
        'rmse_train':        np.sqrt(mean_squared_error(y_true_train, y_pred_train)),
        'r2_train':          r2_score(y_true_train, y_pred_train),
        'mae_test':          mean_absolute_error(y_true_test, y_pred_test),
        'rmse_test':         np.sqrt(mean_squared_error(y_true_test, y_pred_test)),
        'r2_test':           r2_score(y_true_test, y_pred_test),
        'amp_seasonal_test': amp_seasonal,
    }
    results.append(row)
    print(
        f"  MAE_test={row['mae_test']:.6f}  RMSE_test={row['rmse_test']:.6f}"
        f"  R2_test={row['r2_test']:.4f}  amp_saz={amp_seasonal:.6f}"
    )

# ============================================================
# 2. CSV de resultados
# ============================================================
df_results = pd.DataFrame(results)
df_results.to_csv('bsts_seasonality_results.csv', index=False)
print('\nSalvo: bsts_seasonality_results.csv')

print('\n=== Tabela de Resultados ===')
print(df_results.to_string(index=False, float_format=lambda x: f'{x:.6f}'))

# ============================================================
# 3. Figura 1 — Comparacao de metricas (2x2)
# ============================================================
x          = np.arange(len(SEASONALITY_GRID))
x_labels   = [str(s) for s in SEASONALITY_GRID]
val_s5     = df_results.loc[df_results['seasonality'] == 5]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Sensibilidade da Sazonalidade do BSTS — Metricas OOS (2020-2024)', fontsize=13)

configs = [
    (axes[0, 0], 'r2_test',           'steelblue',     'navy',    'R² (maior = melhor)',   'R²'),
    (axes[0, 1], 'mae_test',          'coral',         'darkred', 'MAE (menor = melhor)',   'MAE'),
    (axes[1, 0], 'rmse_test',         'mediumseagreen','darkgreen','RMSE (menor = melhor)', 'RMSE'),
    (axes[1, 1], 'amp_seasonal_test', 'mediumpurple',  'indigo',  'Amplitude da Componente Sazonal (std OOS)', 'std(sazonal)'),
]

for ax, col, color, edge, title, ylabel in configs:
    ax.bar(x, df_results[col], color=color, alpha=0.8, edgecolor=edge, width=0.55)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('Sazonalidade (pregoes)')
    ax.set_ylabel(ylabel)
    if col != 'amp_seasonal_test' and len(val_s5) > 0:
        ref = float(val_s5[col].values[0])
        ax.axhline(ref, color='red', lw=1.2, ls='--', alpha=0.7, label='Atual (s=5)')
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('Overleaf Latex/figs/bsts_seasonality_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('Salvo: Overleaf Latex/figs/bsts_seasonality_comparison.png')

# ============================================================
# 4. Figura 2 — Diagnostico ACF / PACF / Periodograma
# ============================================================
series_train = bsts_train_raw[TARGET].dropna().values
n_lags       = 40
conf_bound   = 1.96 / np.sqrt(len(series_train))

acf_vals  = acf(series_train,  nlags=n_lags, fft=True)[1:]   # lag 0 omitido
pacf_vals = pacf(series_train, nlags=n_lags)[1:]
lags      = np.arange(1, n_lags + 1)

freqs, power = periodogram(series_train, fs=1.0)
mask         = (freqs > 0) & (freqs <= 0.5)
periods      = 1.0 / freqs[mask]
power_mask   = power[mask]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    'Diagnostico de Sazonalidade — Retorno Log do Ibovespa (Treino: 2001-2019)',
    fontsize=13
)

# ACF
axes[0].bar(lags, acf_vals, color='steelblue', alpha=0.75, width=0.6)
axes[0].fill_between([0, n_lags + 1], -conf_bound, conf_bound, alpha=0.12, color='gray', label='IC 95%')
axes[0].axhline(0, color='black', lw=0.5)
axes[0].axvline(5,  color='red',    lw=1.8, ls=':', alpha=0.9, label='lag 5 (semanal)')
axes[0].axvline(21, color='orange', lw=1.8, ls=':', alpha=0.9, label='lag 21 (mensal)')
axes[0].set_title('ACF')
axes[0].set_xlabel('Lag (pregoes)')
axes[0].set_ylabel('Autocorrelacao')
axes[0].set_xlim(0.5, n_lags + 0.5)
axes[0].legend(fontsize=8)

# PACF
axes[1].bar(lags, pacf_vals, color='coral', alpha=0.75, width=0.6)
axes[1].fill_between([0, n_lags + 1], -conf_bound, conf_bound, alpha=0.12, color='gray', label='IC 95%')
axes[1].axhline(0, color='black', lw=0.5)
axes[1].axvline(5,  color='red',    lw=1.8, ls=':', alpha=0.9, label='lag 5 (semanal)')
axes[1].axvline(21, color='orange', lw=1.8, ls=':', alpha=0.9, label='lag 21 (mensal)')
axes[1].set_title('PACF')
axes[1].set_xlabel('Lag (pregoes)')
axes[1].set_ylabel('Autocorrelacao Parcial')
axes[1].set_xlim(0.5, n_lags + 0.5)
axes[1].legend(fontsize=8)

# Periodograma (log, periodo 2-60 pregoes)
mask_plot = (periods >= 2) & (periods <= 60)
axes[2].semilogy(periods[mask_plot], power_mask[mask_plot], color='mediumseagreen', lw=0.9)
axes[2].axvline(5,  color='red',    lw=1.8, ls=':', alpha=0.9, label='Periodo 5 (semanal)')
axes[2].axvline(21, color='orange', lw=1.8, ls=':', alpha=0.9, label='Periodo 21 (mensal)')
axes[2].set_title('Periodograma (Espectro de Potencia)')
axes[2].set_xlabel('Periodo (pregoes)')
axes[2].set_ylabel('Potencia espectral (escala log)')
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig('Overleaf Latex/figs/bsts_seasonality_diagnostico.png', dpi=150, bbox_inches='tight')
plt.close()
print('Salvo: Overleaf Latex/figs/bsts_seasonality_diagnostico.png')

print('\nConcluido.')

"""
Walk-Forward Cross-Validation para TCC — validação de robustez.

Estrutura: janela expansiva, 6 dobras, teste de 2 anos cada.
Hiperparâmetros fixos (não re-tuning por fold): n_splines=10 (GAM),
sigma_prior=0.01 / sigma_embi=0.1 (BSTS), configuração final dos modelos.

Folds (T_k = início do teste):
  Fold 1: treino 2001-2008, holdout 2009, teste 2010-2011
  Fold 2: treino 2001-2010, holdout 2011, teste 2012-2013
  Fold 3: treino 2001-2012, holdout 2013, teste 2014-2015
  Fold 4: treino 2001-2014, holdout 2015, teste 2016-2017
  Fold 5: treino 2001-2016, holdout 2017, teste 2018-2019
  Fold 6: treino 2001-2018, holdout 2019, teste 2020-2021  ← COVID fold
"""

import json
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from functools import reduce
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pygam import LinearGAM, s

warnings.filterwarnings("ignore")

# ── Constantes ────────────────────────────────────────────────────────────────

FIGS_DIR = os.path.join(os.path.dirname(__file__), "Overleaf Latex", "figs")
WF_DIR   = os.path.join(os.path.dirname(__file__), "walk_forward")

# Datas de início de cada janela de teste (T_k)
ANCHOR_DATES = [
    "2010-01-01",  # Fold 1: teste 2010-2011
    "2012-01-01",  # Fold 2: teste 2012-2013
    "2014-01-01",  # Fold 3: teste 2014-2015
    "2016-01-01",  # Fold 4: teste 2016-2017
    "2018-01-01",  # Fold 5: teste 2018-2019
    "2020-01-01",  # Fold 6: teste 2020-2021 (COVID)
]

GAM_FEATURES = [
    "br_selic_diff",
    "br_dolar_diff",
    "br_ipca_diff",      # G4: IPCA em primeira diferença
    "br_pib_ret_log",
    "ipca_expectativa_diff",
    "embi_brasil_diff",
    "ibov_lag1",         # G8: feature autorregressiva
]

BSTS_REGRESSORS = [
    "br_selic_diff",
    "br_dolar_diff",
    "br_ipca_diff",
    "br_pib_ret_log",
    "ipca_expectativa_diff",
    "embi_brasil_diff",
]

TARGET = "br_ibov_ret_log"


# ── Preparação dos dados ─────────────────────────────────────────────────────

def load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df.sort_index()
    # G4: IPCA em primeira diferença
    if "br_ipca" in df.columns and "br_ipca_diff" not in df.columns:
        df["br_ipca_diff"] = df["br_ipca"].diff()
    # G8: lag-1 do Ibovespa
    df["ibov_lag1"] = df[TARGET].shift(1).fillna(0)
    df = df.dropna(subset=GAM_FEATURES + [TARGET])
    return df


# ── Geração de folds ─────────────────────────────────────────────────────────

def make_folds(df: pd.DataFrame,
               anchor_dates=ANCHOR_DATES,
               holdout_months: int = 12,
               test_years: int = 2) -> list[dict]:
    folds = []
    for i, t_str in enumerate(anchor_dates, start=1):
        t_start  = pd.Timestamp(t_str)
        t_end    = t_start + pd.DateOffset(years=test_years) - pd.DateOffset(days=1)
        h_start  = t_start - pd.DateOffset(months=holdout_months)

        train_mask   = df.index <  h_start
        holdout_mask = (df.index >= h_start) & (df.index < t_start)
        test_mask    = (df.index >= t_start) & (df.index <= t_end)

        n_train = train_mask.sum()
        n_hold  = holdout_mask.sum()
        n_test  = test_mask.sum()

        if n_train < 100 or n_test < 10:
            continue

        folds.append({
            "fold":         i,
            "fold_name":    f"Fold {i} ({t_start.year}–{t_end.year})",
            "test_label":   f"{t_start.year}–{t_end.year}",
            "is_covid":     t_start.year == 2020,
            "train_idx":    df.index[train_mask],
            "holdout_idx":  df.index[holdout_mask],
            "test_idx":     df.index[test_mask],
            "n_train":      n_train,
            "n_holdout":    n_hold,
            "n_test":       n_test,
        })
    return folds


def print_fold_summary(folds: list[dict]):
    print(f"{'Fold':<8} {'Treino':>8} {'Holdout':>8} {'Teste':>8}  Período teste")
    print("-" * 55)
    for f in folds:
        covid = " <- COVID" if f["is_covid"] else ""
        print(f"  {f['fold_name']:<22}  n_train={f['n_train']:>4}  "
              f"n_hold={f['n_holdout']:>3}  n_test={f['n_test']:>3}{covid}")


# ── Métricas ─────────────────────────────────────────────────────────────────

def _metrics(y_true, y_pred, label: str) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"fold_name": label, "mae_test": mae, "rmse_test": rmse, "r2_test": r2}


# ── OLS ───────────────────────────────────────────────────────────────────────

def evaluate_ols(folds: list[dict], df: pd.DataFrame) -> pd.DataFrame:
    ols_features = [c for c in GAM_FEATURES if c != "ibov_lag1"]  # OLS sem lag1 (baseline TCC 1)
    rows = []
    for f in folds:
        X_tr = df.loc[f["train_idx"], ols_features].values
        y_tr = df.loc[f["train_idx"], TARGET].values
        X_te = df.loc[f["test_idx"],  ols_features].values
        y_te = df.loc[f["test_idx"],  TARGET].values

        model = LinearRegression().fit(X_tr, y_tr)
        rows.append({**_metrics(y_te, model.predict(X_te), f["fold_name"]),
                     "modelo": "OLS", "is_covid": f["is_covid"]})

    return pd.DataFrame(rows)


# ── GAM ───────────────────────────────────────────────────────────────────────

def evaluate_gam(folds: list[dict], df: pd.DataFrame,
                 n_splines: int = 10) -> pd.DataFrame:
    rows = []
    n_feat = len(GAM_FEATURES)
    term_list = [s(i, n_splines=n_splines) for i in range(n_feat)]
    terms = reduce(lambda a, b: a + b, term_list)

    for f in folds:
        X_tr = df.loc[f["train_idx"], GAM_FEATURES].values
        y_tr = df.loc[f["train_idx"], TARGET].values
        X_te = df.loc[f["test_idx"],  GAM_FEATURES].values
        y_te = df.loc[f["test_idx"],  TARGET].values

        gam = LinearGAM(terms).fit(X_tr, y_tr)
        rows.append({**_metrics(y_te, gam.predict(X_te), f["fold_name"]),
                     "modelo": "GAM", "is_covid": f["is_covid"]})

    return pd.DataFrame(rows)


# ── BSTS ─────────────────────────────────────────────────────────────────────

def evaluate_bsts(folds: list[dict], df: pd.DataFrame,
                  sigma_opt: float = 0.01,
                  sigma_embi: float = 0.10) -> pd.DataFrame:
    try:
        from orbit.models import DLT
    except ImportError:
        print("  [BSTS] orbit-ml não encontrado — pulando avaliação BSTS.")
        return pd.DataFrame()

    sigma_prior = [sigma_opt] * (len(BSTS_REGRESSORS) - 1) + [sigma_embi]

    rows = []
    for f in folds:
        train_df = df.loc[f["train_idx"], [TARGET] + BSTS_REGRESSORS].copy()
        test_df  = df.loc[f["test_idx"],  [TARGET] + BSTS_REGRESSORS].copy()

        train_df = train_df.reset_index().rename(columns={"index": "date", df.index.name or "index": "date"})
        test_df  = test_df.reset_index().rename(columns={"index": "date", df.index.name or "index": "date"})

        if "date" not in train_df.columns:
            train_df.insert(0, "date", train_df.index)
            test_df.insert(0, "date", test_df.index)

        try:
            model = DLT(
                response_col=TARGET,
                date_col="date",
                regressor_col=BSTS_REGRESSORS,
                regressor_sigma_prior=sigma_prior,
                seasonality=5,
                seed=42,
                estimator="stan-map",
            )
            model.fit(train_df)
            pred = model.predict(test_df)

            pred_col = "prediction" if "prediction" in pred.columns else pred.columns[-1]
            y_pred = pred[pred_col].values
            y_true = test_df[TARGET].values

            rows.append({**_metrics(y_true, y_pred, f["fold_name"]),
                         "modelo": "BSTS", "is_covid": f["is_covid"]})
        except Exception as e:
            print(f"  [BSTS] Fold {f['fold_name']} falhou: {e}")
            rows.append({"fold_name": f["fold_name"], "modelo": "BSTS",
                         "r2_test": np.nan, "mae_test": np.nan,
                         "rmse_test": np.nan, "is_covid": f["is_covid"]})

    return pd.DataFrame(rows)


# ── Agregação e decisão ───────────────────────────────────────────────────────

def aggregate_results(df_results: pd.DataFrame) -> pd.DataFrame:
    agg = (df_results.groupby("modelo")[["r2_test", "mae_test", "rmse_test"]]
           .agg(["mean", "std"])
           .round(4))
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg.reset_index()


def apply_decision(df_results: pd.DataFrame,
                   std_threshold: float = 0.05) -> dict:
    gam_rows = df_results[df_results["modelo"] == "GAM"]
    std_r2   = gam_rows["r2_test"].std()
    mean_r2  = gam_rows["r2_test"].mean()
    ganho    = std_r2 < std_threshold

    decision = {
        "criterion":     f"std(R²_GAM) < {std_threshold}",
        "std_r2_gam":    round(float(std_r2), 4),
        "mean_r2_gam":   round(float(mean_r2), 4),
        "threshold":     std_threshold,
        "ganho":         bool(ganho),
        "outcome":       "ACEITO — WF complementa split único" if ganho
                         else "ROLLBACK — alta variância entre dobras",
        "timestamp":     datetime.now().isoformat(),
    }
    return decision


def save_decision(decision: dict, path: str = None):
    if path is None:
        path = os.path.join(WF_DIR, "decision.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2, ensure_ascii=False)
    print(f"Decisão salva em {path}")


# ── Figura ────────────────────────────────────────────────────────────────────

def plot_r2_per_fold(df_results: pd.DataFrame,
                     out_path: str = None) -> str:
    if out_path is None:
        out_path = os.path.join(FIGS_DIR, "walk_forward_r2_per_fold.png")

    modelos   = df_results["modelo"].unique()
    fold_names = df_results["fold_name"].unique()
    x = np.arange(len(fold_names))
    w = 0.25
    colors = {"OLS": "#6c757d", "GAM": "#0f3460", "BSTS": "#e94560"}

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, modelo in enumerate(modelos):
        sub = df_results[df_results["modelo"] == modelo].set_index("fold_name")
        vals = [sub.loc[fn, "r2_test"] if fn in sub.index else np.nan
                for fn in fold_names]
        offset = (i - len(modelos) / 2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=modelo,
                      color=colors.get(modelo, "#333"),
                      alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.004,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=7.5, color="#333")

    # Linha de referência do split único (GAM)
    ax.axhline(0.3045, color="#0f3460", linestyle="--", linewidth=1,
               alpha=0.5, label="GAM split único (0,3045)")

    # Highlight COVID fold
    covid_idx = [i for i, fn in enumerate(fold_names) if "2020" in fn]
    if covid_idx:
        ax.axvspan(covid_idx[0] - 0.5, covid_idx[0] + 0.5,
                   color="#dc3545", alpha=0.07, label="Fold COVID")

    ax.set_xticks(x)
    ax.set_xticklabels(fold_names, fontsize=9)
    ax.set_ylabel("R² out-of-sample", fontsize=10)
    ax.set_title("Walk-Forward CV — R² por Dobra e Modelo\n(janela expansiva, teste de 2 anos, hiperparâmetros fixos)",
                 fontsize=11, pad=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(bottom=min(-0.05, df_results["r2_test"].min() - 0.05))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figura salva em {out_path}")
    return out_path


# ── Runner principal ──────────────────────────────────────────────────────────

def run_walk_forward(csv_path: str = "br_transformado.csv",
                     std_threshold: float = 0.05,
                     run_bsts: bool = True) -> dict:
    print("=" * 60)
    print("Walk-Forward Cross-Validation — TCC 2")
    print("=" * 60)

    df = load_and_prepare(csv_path)
    print(f"\nDados carregados: {len(df)} observações, "
          f"{df.index.min().date()} a {df.index.max().date()}")

    folds = make_folds(df)
    print(f"\n{len(folds)} dobras geradas:")
    print_fold_summary(folds)

    print("\n--- OLS ---")
    ols_res = evaluate_ols(folds, df)
    print(ols_res[["fold_name", "r2_test", "mae_test"]].to_string(index=False))

    print("\n--- GAM ---")
    gam_res = evaluate_gam(folds, df)
    print(gam_res[["fold_name", "r2_test", "mae_test"]].to_string(index=False))

    if run_bsts:
        print("\n--- BSTS ---")
        bsts_res = evaluate_bsts(folds, df)
        if not bsts_res.empty:
            print(bsts_res[["fold_name", "r2_test", "mae_test"]].to_string(index=False))
    else:
        bsts_res = pd.DataFrame()

    all_res = pd.concat([ols_res, gam_res] +
                        ([bsts_res] if not bsts_res.empty else []),
                        ignore_index=True)

    print("\n--- Agregados (mean ± std) ---")
    agg = aggregate_results(all_res)
    print(agg.to_string(index=False))

    print("\n--- Decisão ---")
    decision = apply_decision(all_res, std_threshold)
    print(f"  std(R²_GAM) = {decision['std_r2_gam']:.4f}  "
          f"(threshold = {std_threshold})")
    print(f"  -> {decision['outcome']}")

    save_decision(decision)
    all_res.to_csv(os.path.join(WF_DIR, "wf_results.csv"), index=False)
    agg.to_csv(os.path.join(WF_DIR, "wf_aggregated.csv"), index=False)
    print("\nResultados salvos em walk_forward/")

    plot_r2_per_fold(all_res)

    return {"results": all_res, "aggregated": agg, "decision": decision}

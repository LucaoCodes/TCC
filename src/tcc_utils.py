"""Módulo compartilhado do TCC: paths, constantes e helpers.

Uso nos notebooks:
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))
    from tcc_utils import *
"""
import pathlib
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
FIGS_DIR     = PROJECT_ROOT / "Overleaf Latex" / "figs"

# ── Constantes ────────────────────────────────────────────────────────────────
DATA_INICIO         = '2000-01-01'
DATA_FIM            = '2024-12-31'
DATA_SPLIT_DATE     = '2020-01-01'
VAL_SPLIT_DATE      = '2018-01-01'   # holdout interno para tuning de hiperparâmetros
COLUNAS_LOG_RETORNO = ['br_ibov', 'br_pib']
TARGET              = 'br_ibov_ret_log'
USE_LOCAL_DATA      = True           # nos notebooks de modelo, sempre carrega do CSV

# Valores ARIMA conhecidos (in-sample, TCC 1 — não recalcular)
ARIMA_ORDEM     = (4, 0, 3)
ARIMA_MAE_BASE  = 0.012225
ARIMA_RMSE_BASE = 0.017085

__all__ = [
    'PROJECT_ROOT', 'DATA_DIR', 'FIGS_DIR',
    'DATA_INICIO', 'DATA_FIM', 'DATA_SPLIT_DATE', 'VAL_SPLIT_DATE',
    'COLUNAS_LOG_RETORNO', 'TARGET', 'USE_LOCAL_DATA',
    'ARIMA_ORDEM', 'ARIMA_MAE_BASE', 'ARIMA_RMSE_BASE',
    'load_transformado', 'split_treino_teste', 'split_xy', 'split_tune_val',
    'metricas', 'registrar_metricas', 'carregar_metricas',
]

# ── Dados ─────────────────────────────────────────────────────────────────────
def load_transformado() -> pd.DataFrame:
    """Carrega data/br_transformado.csv e retorna DataFrame com índice de datas."""
    path = DATA_DIR / 'br_transformado.csv'
    return pd.read_csv(path, index_col=0, parse_dates=True)


def split_treino_teste(df: pd.DataFrame):
    """Divide df em treino (<2020-01-01) e teste (>=2020-01-01)."""
    train = df[df.index < DATA_SPLIT_DATE]
    test  = df[df.index >= DATA_SPLIT_DATE]
    return train, test


def split_xy(df: pd.DataFrame):
    """Separa features X (colunas sem 'ibov') e alvo y (TARGET)."""
    X = df[[c for c in df.columns if 'ibov' not in c]]
    y = df[TARGET]
    return X, y


def split_tune_val(df_train: pd.DataFrame):
    """Sub-divide treino em tune (<2018) e val (2018-2019) para tuning honesto."""
    tune = df_train[df_train.index < VAL_SPLIT_DATE]
    val  = df_train[df_train.index >= VAL_SPLIT_DATE]
    return tune, val


# ── Métricas ──────────────────────────────────────────────────────────────────
def metricas(y_true, y_pred) -> dict:
    """Calcula R², MAE e RMSE. Retorna dict {r2, mae, rmse}."""
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    return {
        'r2':   float(r2_score(y_true, y_pred)),
        'mae':  float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


# ── Persistência de métricas ──────────────────────────────────────────────────
_METRICAS_CSV = DATA_DIR / 'metricas_comparativo.csv'
_COLS = ['modelo', 'r2_base', 'mae_base', 'r2_teste', 'mae_teste', 'rmse_teste', 'extras_json']


def registrar_metricas(
    modelo: str,
    r2_base=None,
    mae_base=None,
    r2_teste=None,
    mae_teste=None,
    rmse_teste=None,
    extras: dict = None,
) -> None:
    """Upsert de uma linha no metricas_comparativo.csv (chave: 'modelo').

    Re-rodar o notebook de um modelo atualiza sua linha — nunca duplica.
    """
    import json as _json
    row = {
        'modelo':      modelo,
        'r2_base':     r2_base,
        'mae_base':    mae_base,
        'r2_teste':    r2_teste,
        'mae_teste':   mae_teste,
        'rmse_teste':  rmse_teste,
        'extras_json': _json.dumps(extras) if extras else None,
    }
    if _METRICAS_CSV.exists():
        df = pd.read_csv(_METRICAS_CSV)
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(columns=_COLS)

    mask = df['modelo'] == modelo
    if mask.any():
        for k, v in row.items():
            df.loc[mask, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df.to_csv(_METRICAS_CSV, index=False)
    print(f"[metricas_comparativo] '{modelo}' → {_METRICAS_CSV.name}")


def carregar_metricas() -> pd.DataFrame:
    """Lê metricas_comparativo.csv. Gera erro informativo se não existir."""
    if not _METRICAS_CSV.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {_METRICAS_CSV}\n"
            "Execute os notebooks 02-06 antes de rodar o comparativo."
        )
    return pd.read_csv(_METRICAS_CSV)

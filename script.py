# ============================================================
# CTMC para estados de servidor: simulação + estimação + análise
# - Gera eventos em tempo contínuo (HH:MM:SS)
# - Estima matriz Q por counts/time
# - Resolve distribuição estacionária (πQ = 0, sum π = 1)
# - Roda N simulações e salva resultados por execução e agregados
# ============================================================

import os
import csv
import math
import numpy as np
import pandas as pd
from datetime import timedelta
from numpy.linalg import lstsq

# -------------------------
# Configurações principais
# -------------------------
STATES = ["Disponível", "Lento", "Erro", "Indisponível"]

# Matriz geradora Q verdadeira (taxas por HORA).
# Fora da diagonal = taxas i->j; diagonal = -soma da linha
Q_TRUE = np.array([
    [-0.10, 0.20, 0.10, 0.25],
    [0.20, -0.27, 0.05, 0.02],
    [0.10, 0.02, -0.13, 0.01],
    [0.05, 0.01, 0.02, -0.08]
], dtype=float)

OUT_DIR = "./resultados_ctmc"
NUM_RUNS = 10
EVENTS_PER_RUN = 10000
SEED_BASE = 12345               # reprodutibilidade (gera seeds 12345, 12346, ...)

os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Utilidades
# -------------------------
def _fmt_hms(total_hours: float) -> str:
    """Converte horas decimais para 'HH:MM:SS' (string)."""
    total_seconds = int(round(total_hours * 3600))
    hh = total_seconds // 3600
    mm = (total_seconds % 3600) // 60
    ss = total_seconds % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"

def _normalize_off_diagonal(row: np.ndarray, i: int) -> np.ndarray:
    """Normaliza taxas off-diagonal de uma linha i de Q para probabilidades de salto."""
    probs = row.copy()
    probs[i] = 0.0
    s = probs.sum()
    if s <= 0:
        # Estado absorvente; permanece
        probs[:] = 0
        probs[i] = 1.0
        return probs
    return probs / s

def save_matrix_csv(path: str, M: np.ndarray, states: list[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["De/Para"] + states)
        for i, s in enumerate(states):
            w.writerow([s] + [f"{x:.6f}" for x in M[i]])

def save_series_csv(path: str, s: pd.Series, col_value: str):
    s = s.copy()
    s.name = col_value
    s.index.name = "Estado"
    s.to_csv(path, float_format="%.6f")

# -------------------------
# Simulação CTMC
# -------------------------
def simulate_ctmc(Q: np.ndarray, states: list[str], n_events: int, seed: int | None = None) -> pd.DataFrame:
    """
    Simula uma CTMC:
      - Mantém-se no estado i por tempo ~ Exp(rate = -Q[i,i]) (em HORAS).
      - Escolhe próximo estado j com prob ∝ Q[i,j] (off-diagonal).
    Retorna DataFrame com:
      timestamp(HH:MM:SS acumulado), from, to, duration_hours, duration_hms
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(states)
    # estado inicial aleatório; você pode fixar se preferir
    current_state = np.random.choice(states)
    clock_hours = 0.0

    rows = []
    for _ in range(n_events):
        i = states.index(current_state)
        exit_rate = -Q[i, i]
        if exit_rate <= 0:
            # estado absorvente; define um tempo grande e se mantém
            sojourn = 1.0  # 1 hora (placeholder)
            next_state = current_state
        else:
            sojourn = np.random.exponential(scale=1.0 / exit_rate)
            probs = _normalize_off_diagonal(Q[i, :], i)
            next_idx = np.random.choice(np.arange(n), p=probs)
            next_state = states[next_idx]

        clock_hours += sojourn
        rows.append({
            "timestamp": _fmt_hms(clock_hours),
            "from": current_state,
            "to": next_state,
            "duration_hours": sojourn,
            "duration_hms": _fmt_hms(sojourn)
        })
        current_state = next_state

    df = pd.DataFrame(rows, columns=["timestamp", "from", "to", "duration_hours", "duration_hms"])
    return df

# -------------------------
# Estimação de Q (counts/time)
# -------------------------
def estimate_q_from_events(df: pd.DataFrame, states: list[str]) -> np.ndarray:
    """
    Estima Q por:
      Q[i,j] = (# transições i->j) / (tempo total em i), i != j
      Q[i,i] = -∑_{j!=i} Q[i,j]
    Exige colunas: 'from', 'to', 'duration_hours'
    """
    n = len(states)
    idx = {s: k for k, s in enumerate(states)}
    counts = np.zeros((n, n), dtype=float)
    dwell = np.zeros(n, dtype=float)

    for _, r in df.iterrows():
        i = idx[r["from"]]
        j = idx[r["to"]]
        dwell[i] += float(r["duration_hours"])
        if i != j:
            counts[i, j] += 1.0

    Q_est = np.zeros((n, n), dtype=float)
    for i in range(n):
        if dwell[i] > 0:
            for j in range(n):
                if i != j:
                    Q_est[i, j] = counts[i, j] / dwell[i]
        Q_est[i, i] = -Q_est[i, :].sum()
    return Q_est

# -------------------------
# Estacionária (πQ = 0, sum π = 1)
# -------------------------
def stationary_distribution(Q: np.ndarray, states: list[str]) -> pd.Series:
    n = Q.shape[0]
    # Monta sistema por mínimos quadrados para maior estabilidade:
    # empilha Q^T e a restrição de soma = 1
    A = np.vstack([Q.T, np.ones((1, n))])
    b = np.zeros(n + 1)
    b[-1] = 1.0
    pi, *_ = lstsq(A, b, rcond=None)
    # limpar ruído numérico leve (negativos ~0, soma ~1)
    pi = np.clip(pi, 0, None)
    if pi.sum() > 0:
        pi = pi / pi.sum()
    return pd.Series(pi, index=states)

# -------------------------
# Estimar Q a partir de LOG REAL (opcional)
# -------------------------
def estimate_q_from_real_csv(path_csv: str, states: list[str]) -> np.ndarray:
    """
    Lê um CSV real com colunas: timestamp, status
    Calcula tempos entre registros consecutivos e conta transições.
    Constrói Q como counts/time (em HORAS).
    """
    df = pd.read_csv(path_csv)
    if not {"timestamp", "status"} <= set(df.columns):
        raise ValueError("CSV deve conter colunas 'timestamp' e 'status'.")

    # ordena por tempo; tenta parsear em datetime ou HH:MM:SS
    try:
        t = pd.to_datetime(df["timestamp"])
        seconds = t.astype("int64") // 10**9
    except Exception:
        # assume HH:MM:SS
        def _hms_to_sec(x: str) -> int:
            hh, mm, ss = map(int, str(x).split(":"))
            return hh * 3600 + mm * 60 + ss
        seconds = df["timestamp"].apply(_hms_to_sec).astype(int)

    df = df.assign(_sec=seconds).sort_values("_sec").reset_index(drop=True)

    # monta eventos: from -> to com duração = delta entre timestamps
    rows = []
    for i in range(len(df) - 1):
        s1 = df.loc[i, "status"]
        s2 = df.loc[i + 1, "status"]
        dt_sec = df.loc[i + 1, "_sec"] - df.loc[i, "_sec"]
        if dt_sec < 0:
            continue
        rows.append({
            "from": s1,
            "to": s2,
            "duration_hours": dt_sec / 3600.0
        })
    events = pd.DataFrame(rows)

    if events.empty:
        raise ValueError("Não foi possível construir eventos a partir do CSV real.")

    return estimate_q_from_events(events, states)

# -------------------------
# Execução principal
# -------------------------
def main():
    # Salva Q verdadeiro
    save_matrix_csv(os.path.join(OUT_DIR, "matriz_q_true.csv"), Q_TRUE, STATES)

    # Roda múltiplas simulações
    all_Q = []
    all_pi = []
    for run in range(NUM_RUNS):
        seed = SEED_BASE + run
        df_events = simulate_ctmc(Q_TRUE, STATES, n_events=EVENTS_PER_RUN, seed=seed)

        # Salvar eventos desta simulação
        df_path = os.path.join(OUT_DIR, f"server_status_events_run{run+1}.csv")
        df_events.to_csv(df_path, index=False)

        # Estimar Q e π
        Q_est = estimate_q_from_events(df_events, STATES)
        pi_est = stationary_distribution(Q_est, STATES)

        # Guardar para agregação
        all_Q.append(Q_est)
        all_pi.append(pi_est.values)

        # Salvar resultados individuais
        save_matrix_csv(os.path.join(OUT_DIR, f"matriz_q_estimada_run{run+1}.csv"), Q_est, STATES)
        save_series_csv(os.path.join(OUT_DIR, f"distribuicao_estacionaria_run{run+1}.csv"), pi_est, "pi")

    # Agregados (média das matrizes e das distribuições)
    Q_mean = np.mean(np.stack(all_Q, axis=0), axis=0)
    pi_mean = np.mean(np.stack(all_pi, axis=0), axis=0)

    save_matrix_csv(os.path.join(OUT_DIR, "matriz_q_estimada_MEDIA.csv"), Q_mean, STATES)
    save_series_csv(os.path.join(OUT_DIR, "distribuicao_estacionaria_MEDIA.csv"),
                    pd.Series(pi_mean, index=STATES), "pi")

    # Impressões de resumo
    print(f"[OK] Salvos resultados em: {os.path.abspath(OUT_DIR)}")
    print("\nQ verdadeira:")
    print(pd.DataFrame(Q_TRUE, index=STATES, columns=STATES).round(6))

    print("\nQ estimada (MÉDIA nas simulações):")
    print(pd.DataFrame(Q_mean, index=STATES, columns=STATES).round(6))

    print("\nDistribuição estacionária média (π):")
    s = pd.Series(pi_mean, index=STATES)
    print((s.apply(lambda x: f"{x:.6f} ({x*100:.2f}%)")).to_string())

    # ------------------------------------------
    # (Opcional) Estimar Q a partir de CSV real:
    # Descomente as 3 linhas abaixo e ajuste o caminho do CSV.
    # csv_real = "./server_status.csv"
    # Q_from_real = estimate_q_from_real_csv(csv_real, STATES)
    # save_matrix_csv(os.path.join(OUT_DIR, "matriz_q_estimada_FROM_REAL.csv"), Q_from_real, STATES)
    # ------------------------------------------

if __name__ == "__main__":
    main()

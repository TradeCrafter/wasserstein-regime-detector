"""
Wasserstein Regime Detector — Desktop App
Baseado em: Horvath, Issa & Muguruza (2021)
"Clustering Market Regimes Using the Wasserstein Distance"

Dependências: pip install matplotlib numpy requests
tkinter vem embutido no Python para Windows/macOS.
"""

import tkinter as tk
from tkinter import ttk, font as tkfont
import threading
import time
import math
import numpy as np
import requests
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# ══════════════════════════════════════════════════════════════════════════════
# PALETA
# ══════════════════════════════════════════════════════════════════════════════

BG       = "#0F172A"   # slate-900
BG2      = "#1E293B"   # slate-800
BG3      = "#334155"   # slate-700
BORDER   = "#475569"   # slate-600
TEXT     = "#F1F5F9"   # slate-100
TEXT2    = "#94A3B8"   # slate-400
EMERALD  = "#10B981"
EMERALD2 = "#064E3B"
RED      = "#EF4444"
RED2     = "#450A0A"
BLUE     = "#3B82F6"
AMBER    = "#F59E0B"
WHITE    = "#FFFFFF"

COINS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT",
    "XRPUSDT","DOGEUSDT","AVAXUSDT","LINKUSDT",
    "ADAUSDT","MATICUSDT","DOTUSDT","LTCUSDT",
    "NEARUSDT","ATOMUSDT","UNIUSDT","AAVEUSDT",
]

TIMEFRAMES = {
    "1h  (recomendado)": "60",
    "4h":                "240",
    "15m":               "15",
    "1d":                "D",
}

# ══════════════════════════════════════════════════════════════════════════════
# BYBIT API
# ══════════════════════════════════════════════════════════════════════════════

def fetch_bybit_klines(symbol: str, interval: str = "60", limit: int = 1000):
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol":   symbol,
        "interval": interval,
        "limit":    str(limit),
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("retCode") != 0:
        raise ValueError(f"Bybit API error: {data.get('retMsg')}")
    rows = data["result"]["list"]
    # rows: [startTime, open, high, low, close, volume, turnover]
    rows = list(reversed(rows))
    closes    = [float(x[4]) for x in rows]
    volumes   = [float(x[5]) for x in rows]
    opens     = [float(x[1]) for x in rows]
    highs     = [float(x[2]) for x in rows]
    lows      = [float(x[3]) for x in rows]
    times     = [int(x[0]) / 1000 for x in rows]
    return {
        "closes": closes,
        "opens":  opens,
        "highs":  highs,
        "lows":   lows,
        "volumes": volumes,
        "times":  times,
    }

# ══════════════════════════════════════════════════════════════════════════════
# WASSERSTEIN MATH
# ══════════════════════════════════════════════════════════════════════════════

def log_returns(closes):
    arr = np.array(closes, dtype=np.float64)
    return np.diff(np.log(arr))

def w1_distance(a: np.ndarray, b: np.ndarray) -> float:
    """W₁ entre duas distribuições empíricas — Equação (21) do paper."""
    n = min(len(a), len(b))
    sa = np.sort(a)[:n]
    sb = np.sort(b)[:n]
    return float(np.mean(np.abs(sa - sb)))

def wasserstein_barycenter(cluster: list, p: int = 1) -> np.ndarray:
    """
    Barycenter de Wasserstein p=1: mediana por quantil — Proposition 2.6.
    Para p=2 usa média por quantil — Remark C.2.
    """
    if not cluster:
        return np.array([])
    target_len = max(len(d) for d in cluster)
    resampled = []
    for d in cluster:
        s = np.sort(d)
        if len(s) == target_len:
            resampled.append(s)
        else:
            xi = np.linspace(0, 1, len(s))
            xo = np.linspace(0, 1, target_len)
            resampled.append(np.interp(xo, xi, s))
    stacked = np.array(resampled)
    return np.median(stacked, axis=0) if p == 1 else np.mean(stacked, axis=0)

def wk_means(distributions: list, k: int = 2, max_iter: int = 60, p: int = 1):
    """
    WK-means completo — Definition 2.7 do paper.
    Inicialização: k-means++ para evitar convergência ruim.
    """
    n = len(distributions)
    if n < k:
        raise ValueError(f"Poucas distribuições ({n}) para k={k}")

    # ── Inicialização k-means++ ──────────────────────────────────────────────
    rng = np.random.default_rng(42)
    first = int(rng.integers(0, n))
    centroids = [distributions[first].copy()]

    while len(centroids) < k:
        dists = np.array([
            min(w1_distance(d, c) for c in centroids)
            for d in distributions
        ])
        probs = dists / dists.sum()
        chosen = int(rng.choice(n, p=probs))
        centroids.append(distributions[chosen].copy())

    # ── Iterações ────────────────────────────────────────────────────────────
    history = []
    for iteration in range(max_iter):
        clusters = [[] for _ in range(k)]
        assignments = []

        for dist in distributions:
            dists_to_c = [w1_distance(dist, c) for c in centroids]
            nearest = int(np.argmin(dists_to_c))
            clusters[nearest].append(dist)
            assignments.append(nearest)

        new_centroids = [
            wasserstein_barycenter(cluster, p) if cluster else centroids[i]
            for i, cluster in enumerate(clusters)
        ]

        # Função de perda: Equação (23) do paper
        loss = sum(w1_distance(centroids[i], new_centroids[i]) for i in range(k))
        history.append(loss)
        centroids = new_centroids

        if loss < 1e-7:
            break

    # ── Identifica bull (menor vol) e bear (maior vol) ───────────────────────
    vols = [float(np.std(c)) for c in centroids]
    bull_idx = int(np.argmin(vols))
    bear_idx = int(np.argmax(vols))

    return {
        "bull": centroids[bull_idx],
        "bear": centroids[bear_idx],
        "bull_vol": vols[bull_idx],
        "bear_vol": vols[bear_idx],
        "clusters": clusters,
        "assignments": assignments,
        "history": history,
        "iterations": len(history),
    }

def sliding_windows(returns: np.ndarray, h1: int, h2: int) -> list:
    """
    Stream lift — Definition 1.2 do paper.
    Gera família de distribuições empíricas com janela deslizante.
    """
    step = h1 - h2
    windows = []
    for i in range(0, len(returns) - h1 + 1, max(step, 1)):
        windows.append(returns[i:i + h1])
    return windows

def classify_regime(returns: np.ndarray, bull_centroid: np.ndarray,
                    bear_centroid: np.ndarray, h1: int):
    """Classifica a janela mais recente."""
    window = returns[-h1:]
    if len(window) < h1:
        return None
    w_bull = w1_distance(window, bull_centroid)
    w_bear = w1_distance(window, bear_centroid)
    total  = w_bull + w_bear or 1.0
    label  = "bull" if w_bull <= w_bear else "bear"
    conf   = (1 - min(w_bull, w_bear) / total) * 100
    return {
        "label":      label,
        "confidence": conf,
        "w_bull":     w_bull,
        "w_bear":     w_bear,
    }

def classify_history(returns: np.ndarray, bull_centroid: np.ndarray,
                     bear_centroid: np.ndarray, h1: int, h2: int):
    """Classifica cada janela do histórico para colorir o gráfico de preços."""
    windows = sliding_windows(returns, h1, h2)
    step = max(h1 - h2, 1)
    labels = []
    for i, w in enumerate(windows):
        wb = w1_distance(w, bull_centroid)
        wd = w1_distance(w, bear_centroid)
        labels.append("bull" if wb <= wd else "bear")
    # Mapeia de volta para cada candle
    candle_labels = ["unknown"] * len(returns)
    for i, lab in enumerate(labels):
        start = i * step
        end   = min(start + h1, len(returns))
        for j in range(start, end):
            candle_labels[j] = lab
    return candle_labels

def annualized_vol(returns: np.ndarray, interval_hours: float = 1.0) -> float:
    factor = math.sqrt(24 / interval_hours * 365)
    return float(np.std(returns) * factor)

def annualized_return(returns: np.ndarray, interval_hours: float = 1.0) -> float:
    factor = 24 / interval_hours * 365
    return float(np.mean(returns) * factor)

# ══════════════════════════════════════════════════════════════════════════════
# MOTOR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class RegimeEngine:
    def __init__(self):
        self.klines    = None
        self.returns   = None
        self.wk        = None
        self.current   = None
        self.history   = None
        self.h1        = 35
        self.h2        = 28
        self.symbol    = ""
        self.interval  = "60"
        self.interval_h = 1.0

    def run(self, symbol: str, interval: str, h1: int, h2: int,
            progress_cb=None):
        self.symbol    = symbol
        self.interval  = interval
        self.h1        = h1
        self.h2        = h2
        self.interval_h = {
            "15": 0.25, "60": 1.0, "240": 4.0, "D": 24.0
        }.get(interval, 1.0)

        def p(msg): 
            if progress_cb: progress_cb(msg)

        p("Buscando klines na Bybit API...")
        self.klines = fetch_bybit_klines(symbol, interval, limit=1000)

        p("Calculando log-retornos...")
        self.returns = log_returns(self.klines["closes"])

        p(f"Gerando distribuições empíricas (h1={h1}, h2={h2})...")
        distributions = sliding_windows(self.returns, h1, h2)
        p(f"{len(distributions)} distribuições geradas. Rodando WK-means...")

        self.wk = wk_means(distributions, k=2, max_iter=80)

        p("Classificando regime atual...")
        self.current = classify_regime(
            self.returns, self.wk["bull"], self.wk["bear"], h1
        )

        p("Colorindo histórico de regimes...")
        self.history = classify_history(
            self.returns, self.wk["bull"], self.wk["bear"], h1, h2
        )

        p("Pronto.")

# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICOS
# ══════════════════════════════════════════════════════════════════════════════

def make_figures(engine: RegimeEngine):
    """Cria a figura matplotlib com todos os subplots."""
    plt.rcParams.update({
        "font.family":      "monospace",
        "axes.facecolor":   BG2,
        "figure.facecolor": BG,
        "axes.edgecolor":   BORDER,
        "axes.labelcolor":  TEXT2,
        "xtick.color":      TEXT2,
        "ytick.color":      TEXT2,
        "text.color":       TEXT,
        "grid.color":       BG3,
        "grid.linewidth":   0.5,
        "axes.grid":        True,
    })

    fig = plt.figure(figsize=(14, 10), facecolor=BG)
    gs  = gridspec.GridSpec(
        3, 3,
        figure=fig,
        hspace=0.42,
        wspace=0.35,
        top=0.93, bottom=0.07,
        left=0.06, right=0.97
    )

    closes  = np.array(engine.klines["closes"])
    returns = engine.returns
    hist    = engine.history
    wk      = engine.wk
    cur     = engine.current
    h1      = engine.h1
    ann_h   = engine.interval_h

    bull_color = EMERALD
    bear_color = RED
    n = len(closes)

    # ── 1. Preço com regime coloring (span top 2 cols) ───────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_title(f"{engine.symbol}  ·  Preço com Regime Coloring", 
                  color=TEXT, fontsize=10, pad=6)

    price_x = np.arange(len(closes))
    ax1.plot(price_x, closes, color=TEXT2, linewidth=0.8, zorder=2)

    # Pinta faixas por regime
    i = 0
    regime_hist_aligned = ["unknown"] + list(hist)  # alinha com closes (retornos têm n-1 pontos)
    while i < len(regime_hist_aligned):
        lab = regime_hist_aligned[i]
        j = i
        while j < len(regime_hist_aligned) and regime_hist_aligned[j] == lab:
            j += 1
        if lab != "unknown":
            col = bull_color if lab == "bull" else bear_color
            ax1.axvspan(i, j, alpha=0.18, color=col, zorder=1)
        i = j

    ax1.set_xlim(0, len(closes))
    ax1.set_ylabel("Preço (USDT)", color=TEXT2, fontsize=8)
    ax1.tick_params(labelsize=7)

    # Legenda
    ax1.legend(handles=[
        mpatches.Patch(color=bull_color, alpha=0.5, label="Bull"),
        mpatches.Patch(color=bear_color, alpha=0.5, label="Bear"),
    ], loc="upper left", fontsize=7, framealpha=0.3)

    # ── 2. Dist W₁ ao longo do tempo (top right) ────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_title("W₁ por Janela", color=TEXT, fontsize=10, pad=6)

    windows = sliding_windows(returns, h1, engine.h2)
    w_bulls = [w1_distance(w, wk["bull"]) for w in windows]
    w_bears = [w1_distance(w, wk["bear"]) for w in windows]
    wx = np.arange(len(w_bulls))

    ax2.plot(wx, w_bulls, color=bull_color, linewidth=1.0, label="W₁ bull")
    ax2.plot(wx, w_bears, color=bear_color, linewidth=1.0, label="W₁ bear")
    ax2.fill_between(wx, w_bulls, w_bears,
        where=[b <= d for b, d in zip(w_bulls, w_bears)],
        alpha=0.15, color=bull_color)
    ax2.fill_between(wx, w_bulls, w_bears,
        where=[b > d for b, d in zip(w_bulls, w_bears)],
        alpha=0.15, color=bear_color)
    ax2.legend(fontsize=7, framealpha=0.3)
    ax2.tick_params(labelsize=7)
    ax2.set_ylabel("Distância W₁", color=TEXT2, fontsize=8)

    # ── 3. Distribuição de retornos — atual vs centroides ───────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.set_title("Distribuição de Retornos: Janela Atual vs Centroides", 
                  color=TEXT, fontsize=10, pad=6)

    cur_window = returns[-h1:]
    bins = np.linspace(
        min(cur_window.min(), wk["bull"].min(), wk["bear"].min()) * 1.3,
        max(cur_window.max(), wk["bull"].max(), wk["bear"].max()) * 1.3,
        40
    )
    ax3.hist(np.sort(wk["bull"]), bins=bins, alpha=0.4, color=bull_color,
             label=f"Centroide Bull (σ={wk['bull_vol']*100:.3f}%)", density=True)
    ax3.hist(np.sort(wk["bear"]), bins=bins, alpha=0.4, color=bear_color,
             label=f"Centroide Bear (σ={wk['bear_vol']*100:.3f}%)", density=True)
    ax3.hist(cur_window, bins=bins, alpha=0.65, color=BLUE,
             label=f"Janela atual (h1={h1})", density=True)
    ax3.axvline(0, color=TEXT2, linewidth=0.5, linestyle="--")
    ax3.legend(fontsize=7, framealpha=0.3)
    ax3.set_xlabel("Log-retorno", color=TEXT2, fontsize=8)
    ax3.set_ylabel("Densidade", color=TEXT2, fontsize=8)
    ax3.tick_params(labelsize=7)

    # ── 4. Convergência do WK-means ─────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_title("Convergência WK-means", color=TEXT, fontsize=10, pad=6)
    hist_loss = wk["history"]
    ax4.plot(hist_loss, color=AMBER, linewidth=1.5)
    ax4.fill_between(range(len(hist_loss)), hist_loss, alpha=0.15, color=AMBER)
    ax4.set_xlabel("Iteração", color=TEXT2, fontsize=8)
    ax4.set_ylabel("Função de Perda", color=TEXT2, fontsize=8)
    ax4.tick_params(labelsize=7)
    ax4.text(
        0.98, 0.95,
        f"Convergiu em\n{wk['iterations']} iterações",
        transform=ax4.transAxes,
        ha="right", va="top", fontsize=7, color=TEXT2
    )

    # ── 5. QQ-plot: janela atual vs centroide vencedor ───────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.set_title("QQ-plot: Atual vs Centroide", color=TEXT, fontsize=10, pad=6)

    winner_centroid = wk["bull"] if cur and cur["label"] == "bull" else wk["bear"]
    winner_color    = bull_color if cur and cur["label"] == "bull" else bear_color
    n_pts = min(len(cur_window), len(winner_centroid))
    q_cur = np.quantile(cur_window, np.linspace(0.01, 0.99, n_pts))
    q_win = np.quantile(winner_centroid, np.linspace(0.01, 0.99, n_pts))
    ax5.scatter(q_win, q_cur, s=6, color=winner_color, alpha=0.6)
    lim = max(abs(q_cur).max(), abs(q_win).max()) * 1.1
    ax5.plot([-lim, lim], [-lim, lim], color=TEXT2, linewidth=0.8, linestyle="--")
    ax5.set_xlabel("Quantis centroide", color=TEXT2, fontsize=8)
    ax5.set_ylabel("Quantis janela atual", color=TEXT2, fontsize=8)
    ax5.tick_params(labelsize=7)

    # ── 6. Histograma de regimes ao longo do tempo ───────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.set_title("Proporção de Regimes (histórico)", color=TEXT, fontsize=10, pad=6)

    valid = [l for l in hist if l != "unknown"]
    n_bull = valid.count("bull")
    n_bear = valid.count("bear")
    total  = n_bull + n_bear or 1

    bars = ax6.bar(
        ["Bull", "Bear"],
        [n_bull / total * 100, n_bear / total * 100],
        color=[bull_color, bear_color],
        width=0.5,
        edgecolor=BORDER
    )
    for bar, val in zip(bars, [n_bull / total * 100, n_bear / total * 100]):
        ax6.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{val:.1f}%",
            ha="center", va="bottom", color=TEXT, fontsize=9, fontweight="bold"
        )
    ax6.set_ylim(0, 105)
    ax6.set_ylabel("% do tempo", color=TEXT2, fontsize=8)
    ax6.tick_params(labelsize=8)

    # ── 7. Volatilidade realizada por regime ─────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.set_title("Volatilidade por Regime (anual.)", color=TEXT, fontsize=10, pad=6)

    bull_vol_ann = annualized_vol(wk["bull"], ann_h) * 100
    bear_vol_ann = annualized_vol(wk["bear"], ann_h) * 100
    cur_vol_ann  = annualized_vol(returns[-h1:], ann_h) * 100

    cats   = ["Bull\ncentroide", "Bear\ncentroide", "Janela\nAtual"]
    vals   = [bull_vol_ann, bear_vol_ann, cur_vol_ann]
    colors = [bull_color, bear_color, BLUE]

    bars2 = ax7.bar(cats, vals, color=colors, width=0.5, edgecolor=BORDER)
    for bar, val in zip(bars2, vals):
        ax7.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center", va="bottom", color=TEXT, fontsize=8, fontweight="bold"
        )
    ax7.set_ylabel("Vol Anualizada (%)", color=TEXT2, fontsize=8)
    ax7.tick_params(labelsize=7)

    # ── Título global ─────────────────────────────────────────────────────────
    regime_label = cur["label"].upper() if cur else "?"
    regime_color = bull_color if (cur and cur["label"] == "bull") else bear_color
    conf_str = f"{cur['confidence']:.1f}%" if cur else "?"

    fig.text(0.5, 0.975,
        f"WASSERSTEIN REGIME DETECTOR  ·  {engine.symbol}  ·  "
        f"Regime Atual: ",
        ha="center", va="top", fontsize=11, color=TEXT, fontweight="bold")
    fig.text(0.72, 0.975,
        f"{regime_label}  ({conf_str} confiança)",
        ha="center", va="top", fontsize=11, color=regime_color, fontweight="bold")

    return fig

# ══════════════════════════════════════════════════════════════════════════════
# INTERFACE TKINTER
# ══════════════════════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Wasserstein Regime Detector — Tradecraft Analytics")
        self.configure(bg=BG)
        self.geometry("1320x820")
        self.minsize(1000, 680)

        self.engine   = RegimeEngine()
        self.canvas   = None
        self.fig      = None
        self._running = False

        self._build_ui()

    # ── Layout ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Painel esquerdo (controles + métricas)
        left = tk.Frame(self, bg=BG, width=240)
        left.pack(side="left", fill="y", padx=(10, 0), pady=10)
        left.pack_propagate(False)

        # Painel direito (gráfico)
        self.right = tk.Frame(self, bg=BG)
        self.right.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self._build_controls(left)
        self._build_metrics(left)
        self._build_placeholder()

    def _build_controls(self, parent):
        self._section(parent, "CONFIGURAÇÃO")

        tk.Label(parent, text="Par:", bg=BG, fg=TEXT2,
                 font=("Courier", 9)).pack(anchor="w", padx=8)
        self.coin_var = tk.StringVar(value="BTCUSDT")
        coin_cb = ttk.Combobox(parent, textvariable=self.coin_var,
                                values=COINS, state="readonly", width=18)
        coin_cb.pack(padx=8, pady=(2, 8), fill="x")
        self._style_combobox(coin_cb)

        tk.Label(parent, text="Timeframe:", bg=BG, fg=TEXT2,
                 font=("Courier", 9)).pack(anchor="w", padx=8)
        self.tf_var = tk.StringVar(value="1h  (recomendado)")
        tf_cb = ttk.Combobox(parent, textvariable=self.tf_var,
                              values=list(TIMEFRAMES.keys()),
                              state="readonly", width=18)
        tf_cb.pack(padx=8, pady=(2, 8), fill="x")
        self._style_combobox(tf_cb)

        # Sliders h1 / h2
        tk.Label(parent, text="Janela h1 (retornos):", bg=BG, fg=TEXT2,
                 font=("Courier", 9)).pack(anchor="w", padx=8)
        self.h1_var = tk.IntVar(value=35)
        h1_frame = tk.Frame(parent, bg=BG)
        h1_frame.pack(fill="x", padx=8, pady=(2, 6))
        self.h1_lbl = tk.Label(h1_frame, text="35", bg=BG, fg=EMERALD,
                                font=("Courier", 9, "bold"), width=3)
        self.h1_lbl.pack(side="right")
        h1_sl = tk.Scale(h1_frame, from_=10, to=100, orient="horizontal",
                         variable=self.h1_var, bg=BG, fg=TEXT2,
                         troughcolor=BG3, highlightthickness=0,
                         command=lambda v: self.h1_lbl.configure(text=v),
                         showvalue=False)
        h1_sl.pack(side="left", fill="x", expand=True)

        tk.Label(parent, text="Overlap h2:", bg=BG, fg=TEXT2,
                 font=("Courier", 9)).pack(anchor="w", padx=8)
        self.h2_var = tk.IntVar(value=28)
        h2_frame = tk.Frame(parent, bg=BG)
        h2_frame.pack(fill="x", padx=8, pady=(2, 10))
        self.h2_lbl = tk.Label(h2_frame, text="28", bg=BG, fg=TEXT2,
                                font=("Courier", 9), width=3)
        self.h2_lbl.pack(side="right")
        h2_sl = tk.Scale(h2_frame, from_=0, to=80, orient="horizontal",
                         variable=self.h2_var, bg=BG, fg=TEXT2,
                         troughcolor=BG3, highlightthickness=0,
                         command=lambda v: self.h2_lbl.configure(text=v),
                         showvalue=False)
        h2_sl.pack(side="left", fill="x", expand=True)

        # Botão Analisar
        self.btn = tk.Button(
            parent, text="▶  ANALISAR",
            bg=EMERALD2, fg=EMERALD,
            activebackground=BG3, activeforeground=EMERALD,
            font=("Courier", 10, "bold"),
            relief="flat", bd=0, cursor="hand2",
            command=self._start_analysis,
            pady=10
        )
        self.btn.pack(fill="x", padx=8, pady=(0, 6))

        # Botão Atualizar (reabre mesma análise)
        self.btn_refresh = tk.Button(
            parent, text="↺  ATUALIZAR",
            bg=BG2, fg=TEXT2,
            activebackground=BG3, activeforeground=TEXT,
            font=("Courier", 9),
            relief="flat", bd=0, cursor="hand2",
            command=self._start_analysis,
            pady=6
        )
        self.btn_refresh.pack(fill="x", padx=8, pady=(0, 10))

        # Status bar
        self.status_var = tk.StringVar(value="Selecione um par e clique em Analisar.")
        tk.Label(parent, textvariable=self.status_var,
                 bg=BG, fg=TEXT2, font=("Courier", 8),
                 wraplength=220, justify="left").pack(padx=8, pady=(0, 4))

        # Progress bar
        self.progress = ttk.Progressbar(parent, mode="indeterminate", length=220)
        self.progress.pack(padx=8, pady=(0, 10))

    def _build_metrics(self, parent):
        self._section(parent, "REGIME ATUAL")

        # Badge de regime
        self.regime_frame = tk.Frame(parent, bg=BG2,
                                      highlightbackground=BORDER,
                                      highlightthickness=1)
        self.regime_frame.pack(fill="x", padx=8, pady=(0, 8))
        self.regime_lbl = tk.Label(
            self.regime_frame, text="—", bg=BG2, fg=TEXT2,
            font=("Courier", 18, "bold"), pady=10
        )
        self.regime_lbl.pack()
        self.conf_lbl = tk.Label(
            self.regime_frame, text="confiança: —", bg=BG2, fg=TEXT2,
            font=("Courier", 9)
        )
        self.conf_lbl.pack(pady=(0, 8))

        self._section(parent, "MÉTRICAS")

        metrics = [
            ("W₁ → Bull",    "w_bull_lbl",  "—"),
            ("W₁ → Bear",    "w_bear_lbl",  "—"),
            ("Vol Bull (a)", "vbull_lbl",   "—"),
            ("Vol Bear (a)", "vbear_lbl",   "—"),
            ("Vol Atual",    "vcur_lbl",    "—"),
            ("Iterações WK", "iter_lbl",    "—"),
            ("Janelas geradas","windows_lbl","—"),
        ]
        for label, attr, default in metrics:
            row = tk.Frame(parent, bg=BG)
            row.pack(fill="x", padx=8, pady=1)
            tk.Label(row, text=label + ":", bg=BG, fg=TEXT2,
                     font=("Courier", 8), anchor="w", width=14).pack(side="left")
            lbl = tk.Label(row, text=default, bg=BG, fg=TEXT,
                           font=("Courier", 8, "bold"), anchor="e")
            lbl.pack(side="right")
            setattr(self, attr, lbl)

        # Footer
        tk.Label(parent, text="\nHorvath, Issa & Muguruza\n(SSRN 3947905, 2021)",
                 bg=BG, fg=BG3, font=("Courier", 7),
                 justify="center").pack(side="bottom", pady=8)

    def _build_placeholder(self):
        self.placeholder = tk.Frame(self.right, bg=BG2,
                                     highlightbackground=BORDER,
                                     highlightthickness=1)
        self.placeholder.pack(fill="both", expand=True)
        tk.Label(
            self.placeholder,
            text="Selecione um par e clique em  ▶ ANALISAR\n\n"
                 "O algoritmo irá:\n"
                 "  1. Buscar 1000 candles na API Bybit\n"
                 "  2. Calcular log-retornos\n"
                 "  3. Gerar distribuições empíricas (sliding window)\n"
                 "  4. Rodar WK-means com Wasserstein barycenter\n"
                 "  5. Classificar o regime atual\n"
                 "  6. Colorir histórico de preços por regime",
            bg=BG2, fg=TEXT2,
            font=("Courier", 10),
            justify="left"
        ).place(relx=0.5, rely=0.5, anchor="center")

    # ── Helpers UI ────────────────────────────────────────────────────────────

    def _section(self, parent, title):
        tk.Label(parent, text=title, bg=BG, fg=TEXT2,
                 font=("Courier", 8), anchor="w",
                 pady=4).pack(fill="x", padx=8)
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=8, pady=(0, 6))

    def _style_combobox(self, cb):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox",
                         fieldbackground=BG2, background=BG2,
                         foreground=TEXT, selectforeground=TEXT,
                         selectbackground=BG3, bordercolor=BORDER,
                         arrowcolor=TEXT2)

    # ── Análise ───────────────────────────────────────────────────────────────

    def _start_analysis(self):
        if self._running:
            return
        self._running = True
        self.btn.configure(state="disabled", text="Analisando...")
        self.progress.start(10)

        def run():
            try:
                symbol   = self.coin_var.get()
                tf_key   = self.tf_var.get()
                interval = TIMEFRAMES.get(tf_key, "60")
                h1       = self.h1_var.get()
                h2       = min(self.h2_var.get(), h1 - 1)

                self.engine.run(
                    symbol, interval, h1, h2,
                    progress_cb=lambda m: self.status_var.set(m)
                )
                self.after(0, self._on_success)
            except Exception as e:
                self.after(0, lambda: self._on_error(str(e)))

        threading.Thread(target=run, daemon=True).start()

    def _on_success(self):
        self.progress.stop()
        self.btn.configure(state="normal", text="▶  ANALISAR")
        self._running = False
        self._update_metrics()
        self._render_charts()

    def _on_error(self, msg):
        self.progress.stop()
        self.btn.configure(state="normal", text="▶  ANALISAR")
        self._running = False
        self.status_var.set(f"Erro: {msg}")

    def _update_metrics(self):
        eng = self.engine
        cur = eng.current
        wk  = eng.wk

        if cur:
            label = cur["label"].upper()
            col   = EMERALD if cur["label"] == "bull" else RED
            bg    = EMERALD2 if cur["label"] == "bull" else RED2

            self.regime_frame.configure(highlightbackground=col)
            self.regime_lbl.configure(
                text=f"{'🟢' if cur['label']=='bull' else '🔴'}  {label}",
                fg=col, bg=bg
            )
            self.regime_frame.configure(bg=bg)
            self.conf_lbl.configure(
                text=f"confiança: {cur['confidence']:.1f}%",
                fg=col, bg=bg
            )
            self.w_bull_lbl.configure(text=f"{cur['w_bull']:.5f}")
            self.w_bear_lbl.configure(text=f"{cur['w_bear']:.5f}")

        if wk:
            ann_h = eng.interval_h
            vbull = annualized_vol(wk["bull"], ann_h) * 100
            vbear = annualized_vol(wk["bear"], ann_h) * 100
            vcur  = annualized_vol(eng.returns[-eng.h1:], ann_h) * 100

            self.vbull_lbl.configure(text=f"{vbull:.1f}%")
            self.vbear_lbl.configure(text=f"{vbear:.1f}%")
            self.vcur_lbl.configure(text=f"{vcur:.1f}%",
                fg=EMERALD if vcur < vbear else RED)
            self.iter_lbl.configure(text=str(wk["iterations"]))

            wins = sliding_windows(eng.returns, eng.h1, eng.h2)
            self.windows_lbl.configure(text=str(len(wins)))

        self.status_var.set(
            f"OK · {eng.symbol} · "
            f"{len(eng.klines['closes'])} candles"
        )

    def _render_charts(self):
        # Remove placeholder e canvas anterior
        if hasattr(self, "placeholder") and self.placeholder.winfo_exists():
            self.placeholder.destroy()
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        if self.fig:
            plt.close(self.fig)

        self.fig = make_figures(self.engine)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = App()
    app.mainloop()

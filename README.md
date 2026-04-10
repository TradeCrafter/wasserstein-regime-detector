<img src="https://raw.githubusercontent.com/TradeCrafter/wasserstein-regime-detector/main/tradecraft-logo.jpg" width="100" height="250">
# Tradecraft Analytics

> Professional-grade crypto trading dashboard powered by the Bybit API.

Tradecraft Analytics is a full-stack SaaS platform built for perpetual futures traders. It combines real-time market data with quantitative research-grade analysis — from order book depth heatmaps to distributional regime detection based on peer-reviewed financial mathematics.

---

## Features

### Market Intelligence
- **Candlestick charts** with RSI, MACD, Bollinger Bands and EMA overlays, multiple timeframes
- **Order Book Heatmap** — live depth visualization with Web Worker rendering and zero-copy Transferable buffers for smooth performance
- **Volume Profile** and MACD histogram panels
- **Trading Signals Panel** — composite score across technical, derivatives and real-time orderbook signals with entry/stop/target levels

### Quantitative Models
- **Wasserstein Regime Detector** — unsupervised market regime classification using the 1-Wasserstein distance between empirical return distributions, based on [Horvath, Issa & Muguruza (2021)](https://ssrn.com/abstract=3947905). Detects bull/bear regimes without parametric assumptions on the return distribution
- **Hidden Markov Model (HMM) Regime Panel** — latent state sequence estimation with Gaussian emission probabilities
- **Kalman Filter Panel** — signal extraction and trend estimation with econophysics-derived signals
- **Fractal Regime Detector** — MFDFA-based multifractal analysis for volatility clustering detection
- **Econophysics Panel** — VPIN (Volume-synchronized Probability of Informed Trading), OFI (Order Flow Imbalance), Local Volatility surface, Depth Correlation

### Derivatives & Forecasting
- **Options Panel** — open interest, put/call ratio and market bias for BTC and ETH (Bybit options)
- **Forecast Panel** — ARIMA-GARCH volatility forecasting with confidence bands
- **Chart Patterns Panel** — automated pattern recognition tied to regime state

### Platform
- **Paper Trading** — full simulated trading environment with position tracking and P&L, no real capital required
- **Price Alerts** — user-configurable alerts stored in Supabase with real-time evaluation
- **Admin Panel** — user approval workflow, MRR tracking, user management
- **Crypto payments** — ETH/BSC/Arbitrum payment detection via on-chain RPC calls

---

## Stack

| Layer | Technology |
|---|---|
| Frontend | React 18 + Vite + Tailwind CSS |
| Charts | Lightweight Charts, Chart.js, D3.js, custom SVG |
| Auth & DB | Supabase (PostgreSQL + Row Level Security) |
| Hosting | Vercel |
| Market Data | Bybit V5 API (REST + WebSocket) |
| Scheduling | Supabase pg_cron + Edge Functions (Deno) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Bybit V5 API                         │
│         REST (klines, OI, funding) + WebSocket          │
└──────────┬──────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│                React Frontend (Vercel)                  │
│                                                         │
│  useMarketData   useTechnicalIndicators   useOptions    │
│  useMarketRegime useEconophysics          useForeccast   │
│  useWassersteinRegime  useTradingSignals               │
│                                                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │Heatmap   │ │Regime    │ │Kalman    │ │Signals   │  │
│  │(Worker)  │ │Detector  │ │Filter    │ │Panel     │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
└──────────┬──────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│                Supabase (PostgreSQL)                    │
│                                                         │
│  users · watchlists · price_alerts · regime_centroids  │
│  api_cache · sentiment_data                             │
│                                                         │
│  Edge Functions:  calibrate-regimes (WK-means)         │
│  pg_cron:         weekly recalibration (Mon 03:00 UTC) │
└─────────────────────────────────────────────────────────┘
```

---

<img src="https://raw.githubusercontent.com/TradeCrafter/wasserstein-regime-detector/main/screen.png">

## Wasserstein Regime Detector

The regime detection module is the most technically sophisticated component of the platform. It implements the **WK-means algorithm** from Horvath, Issa & Muguruza (2021), which frames market regime clustering as a problem on the space of probability measures with finite *p*-th moment, using the *p*-Wasserstein distance as the metric.

### How it works

**Offline calibration** (runs weekly via Supabase pg_cron):

1. Fetch 1000 hourly candles from Bybit for each tracked symbol
2. Compute log-returns: $r_i = \log(S_{i+1}) - \log(S_i)$
3. Apply a sliding window lift with parameters $(h_1, h_2)$ to generate a family of empirical distributions $\mathcal{K} = \{\mu_1, \ldots, \mu_M\}$
4. Run WK-means on $(\mathcal{P}_p(\mathbb{R}), W_p)$ with Wasserstein barycenters as cluster centroids (Proposition 2.6)
5. Persist the bull and bear centroids to `regime_centroids` table

**Real-time classification** (runs in the browser, < 1ms per candle):

Given centroids $\bar{\mu}^{\text{bull}}, \bar{\mu}^{\text{bear}}$ and the current return window $w \in \mathbb{R}^{h_1}$:

$$W_1(\mu, \nu) = \frac{1}{N} \sum_{i=1}^{N} |\alpha_i - \beta_i|$$

where $\alpha_i, \beta_i$ are the sorted order statistics. Regime is assigned to $\arg\min_{c} \, W_1(w, c)$.

### Why Wasserstein over moments?

The Kolmogorov-Smirnov statistic lacks sensitivity for distributional differences in the tails. The KL-divergence requires density estimation and has no natural barycenter. The Wasserstein distance metrizes weak convergence and admits the Wasserstein barycenter as a tractable aggregator — making it the natural choice for clustering empirical distributions on the real line.

On SPY data (2005–2020), the WK-means algorithm correctly identified the Global Financial Crisis, COVID crash, the Eurozone debt crisis (2010), S&P downgrade (2011) and the Chinese market crash (2015) — events that the moment-based MK-means missed entirely.

### Standalone desktop tool

A self-contained Python desktop application is available for offline analysis: [`wasserstein_regime.py`](tools/wasserstein_regime.py)

```bash
pip install matplotlib numpy requests
python wasserstein_regime.py
```

Renders 7 charts: price path with regime coloring, W₁ distance over time, return distribution vs centroids, WK-means convergence, QQ-plot, regime proportion histogram, and annualized volatility by regime.

---

## Getting Started

### Prerequisites

- Node.js 18+
- A Supabase project
- A Bybit account (read-only API key, optional — public endpoints are used by default)

### Local development

```bash
git clone https://github.com/tradecraft-labs/tradecraft-analytics
cd tradecraft-analytics
npm install
```

Create `.env.local`:

```
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key
```

```bash
npm run dev
```

### Supabase setup

Run the migrations in order:

```bash
supabase link --project-ref your-project-ref
supabase db push
```

Deploy the regime calibration Edge Function:

```bash
supabase functions deploy calibrate-regimes --no-verify-jwt
```

Trigger the initial calibration:

```bash
curl -X POST https://your-project.supabase.co/functions/v1/calibrate-regimes \
  -H "Authorization: Bearer YOUR_SERVICE_ROLE_KEY" \
  -H "Content-Type: application/json" \
  -d '{}'
```

---

## Database Schema

```sql
-- User management
users (id, email, status, created_at)

-- Watchlists per user
watchlists (id, user_id, coin_id, created_at)

-- Price alerts
price_alerts (id, user_id, coin_id, target_price, direction, triggered, created_at)

-- Wasserstein regime centroids (updated weekly)
regime_centroids (symbol, bull_centroid float8[], bear_centroid float8[],
                  bull_vol, bear_vol, h1, calibrated_at)

-- API response cache
api_cache (key, data, expires_at)

-- Sentiment data
sentiment_data (coin_id, score, source, created_at)
```

---

## Research References

The quantitative models in Tradecraft Analytics are grounded in academic literature:

| Model | Paper |
|---|---|
| Wasserstein Regime Detector | Horvath, B., Issa, Z., & Muguruza, A. (2021). *Clustering Market Regimes Using the Wasserstein Distance.* SSRN 3947905 |
| Bar Portion Signal (PMM bot) | Stoikov, S. (2024). *The bar portion signal.* |
| VPIN | Easley, D., López de Prado, M. & O'Hara, M. (2012). *Flow toxicity and liquidity in a high-frequency world.* Review of Financial Studies |
| Kalman Filter | Harvey, A. C. (1990). *Forecasting, Structural Time Series Models and the Kalman Filter.* Cambridge University Press |
| HMM Regime Switching | Hamilton, J. D. (1989). *A new approach to the economic analysis of nonstationary time series and the business cycle.* Econometrica |

---

## License

Proprietary. All rights reserved. Tradecraft Labs.

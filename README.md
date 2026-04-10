<img src="https://raw.githubusercontent.com/TradeCrafter/wasserstein-regime-detector/main/tradecraft-logo.jpg" width="200" height="500">

# Tradecraft Analytics

> Dashboard profissional de análise de crypto powered by Bybit API.

Tradecraft Analytics é uma plataforma SaaS full-stack construída para traders de futuros perpétuos. Combina dados de mercado em tempo real com análise de nível acadêmico — desde heatmaps de profundidade de order book até detecção de regimes baseada em matemática financeira publicada em periódicos revisados por pares.

---

## Funcionalidades

### Inteligência de Mercado
- **Gráficos de candlestick** com RSI, MACD, Bollinger Bands e EMAs, múltiplos timeframes
- **Heatmap de Order Book** — visualização de profundidade ao vivo com rendering em Web Worker e buffers Transferable de zero-cópia para performance suave
- **Volume Profile** e painel de histograma MACD
- **Painel de Sinais** — score composto sobre indicadores técnicos, derivativos e order book em tempo real, com níveis de entrada, stop e alvo

### Modelos Quantitativos
- **Wasserstein Regime Detector** — classificação não-supervisionada de regimes de mercado usando a distância 1-Wasserstein entre distribuições empíricas de log-retornos, baseado em [Horvath, Issa & Muguruza (2021)](https://ssrn.com/abstract=3947905). Detecta regimes bull/bear sem premissas paramétricas sobre a distribuição dos retornos
- **Painel HMM (Hidden Markov Model)** — estimação de sequência de estados latentes com probabilidades de emissão gaussianas
- **Painel Kalman Filter** — extração de sinal e estimação de tendência com sinais derivados de econofísica
- **Fractal Regime Detector** — análise multifractal baseada em MFDFA para detecção de clustering de volatilidade
- **Painel de Econofísica** — VPIN (Volume-synchronized Probability of Informed Trading), OFI (Order Flow Imbalance), Superfície de Volatilidade Local, Correlação de Profundidade

### Derivativos & Previsão
- **Painel de Opções** — open interest, put/call ratio e viés de mercado para BTC e ETH (opções Bybit)
- **Painel de Forecast** — previsão de volatilidade ARIMA-GARCH com bandas de confiança
- **Painel de Padrões Gráficos** — reconhecimento automático de padrões vinculado ao estado de regime

### Plataforma
- **Paper Trading** — ambiente completo de trading simulado com rastreamento de posições e P&L, sem capital real
- **Alertas de Preço** — alertas configuráveis pelo usuário armazenados no Supabase com avaliação em tempo real
- **Painel Admin** — fluxo de aprovação de usuários, rastreamento de MRR, gestão de usuários
- **Pagamentos em crypto** — detecção de pagamentos ETH/BSC/Arbitrum via chamadas RPC on-chain

---

## Stack

| Camada | Tecnologia |
|---|---|
| Frontend | React 18 + Vite + Tailwind CSS |
| Gráficos | Lightweight Charts, Chart.js, D3.js, SVG customizado |
| Auth & Banco | Supabase (PostgreSQL + Row Level Security) |
| Hospedagem | Vercel |
| Dados de Mercado | Bybit V5 API (REST + WebSocket) |
| Agendamento | Supabase pg_cron + Edge Functions (Deno) |

---

## Arquitetura

```
┌─────────────────────────────────────────────────────────┐
│                    Bybit V5 API                         │
│         REST (klines, OI, funding) + WebSocket          │
└──────────┬──────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│               Frontend React (Vercel)                   │
│                                                         │
│  useMarketData   useTechnicalIndicators   useOptions    │
│  useMarketRegime useEconophysics          useForecast    │
│  useWassersteinRegime  useTradingSignals               │
│                                                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │Heatmap   │ │Regime    │ │Kalman    │ │Sinais    │  │
│  │(Worker)  │ │Detector  │ │Filter    │ │Trading   │  │
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
│  pg_cron:         recalibração semanal (seg 03:00 UTC) │
└─────────────────────────────────────────────────────────┘
```
<img src="https://raw.githubusercontent.com/TradeCrafter/wasserstein-regime-detector/main/screen.png">
---

## Wasserstein Regime Detector

O módulo de detecção de regimes é o componente mais sofisticado tecnicamente da plataforma. Implementa o **algoritmo WK-means** de Horvath, Issa & Muguruza (2021), que enquadra o problema de clustering de regimes de mercado como um problema no espaço de medidas de probabilidade com momento *p* finito, usando a distância *p*-Wasserstein como métrica.

### Como funciona

**Calibração offline** (roda semanalmente via Supabase pg_cron):

1. Busca 1000 candles de 1 hora na Bybit para cada par rastreado
2. Calcula log-retornos: $r_i = \log(S_{i+1}) - \log(S_i)$
3. Aplica um lifting de janela deslizante com parâmetros $(h_1, h_2)$ para gerar uma família de distribuições empíricas $\mathcal{K} = \{\mu_1, \ldots, \mu_M\}$
4. Roda WK-means em $(\mathcal{P}_p(\mathbb{R}), W_p)$ com baricentros de Wasserstein como centroides de cluster (Proposição 2.6)
5. Persiste os centroides bull e bear na tabela `regime_centroids`

**Classificação em tempo real** (roda no browser, < 1ms por candle):

Dados os centroides $\bar{\mu}^{\text{bull}}, \bar{\mu}^{\text{bear}}$ e a janela de retornos atual $w \in \mathbb{R}^{h_1}$:

$$W_1(\mu, \nu) = \frac{1}{N} \sum_{i=1}^{N} |\alpha_i - \beta_i|$$

onde $\alpha_i, \beta_i$ são as estatísticas de ordem ordenadas. O regime é atribuído a $\arg\min_{c} \, W_1(w, c)$.

### Por que Wasserstein e não momentos?

A estatística de Kolmogorov-Smirnov carece de sensibilidade para diferenças distribucionais nas caudas. A divergência KL exige estimação de densidade e não tem baricentro natural tratável. A distância de Wasserstein metriza convergência fraca e admite o baricentro de Wasserstein como agregador computacionalmente tratável — tornando-a a escolha natural para clustering de distribuições empíricas na reta real.

Em dados do SPY (2005–2020), o algoritmo WK-means identificou corretamente a Crise Financeira Global, o crash do COVID, a crise da dívida da Eurozona (2010), o downgrade do S&P (2011) e o crash da bolsa chinesa (2015) — eventos que o MK-means baseado em momentos não detectou.

### Ferramenta desktop standalone

Uma aplicação desktop Python autossuficiente está disponível para análise offline: [`wasserstein_regime.py`](tools/wasserstein_regime.py)

```bash
pip install matplotlib numpy requests
python wasserstein_regime.py
```

Renderiza 7 gráficos: trajetória de preço com coloração por regime, distância W₁ ao longo do tempo, distribuição de retornos vs centroides, convergência do WK-means, QQ-plot, histograma de proporção de regimes e volatilidade anualizada por regime.

---

## Começando

### Pré-requisitos

- Node.js 18+
- Um projeto Supabase
- Conta Bybit (chave de API somente leitura, opcional — endpoints públicos são usados por padrão)

### Desenvolvimento local

```bash
git clone https://github.com/tradecraft-labs/tradecraft-analytics
cd tradecraft-analytics
npm install
```

Criar `.env.local`:

```
VITE_SUPABASE_URL=https://seu-projeto.supabase.co
VITE_SUPABASE_ANON_KEY=sua-anon-key
```

```bash
npm run dev
```

### Setup do Supabase

Rodar as migrations em ordem:

```bash
supabase link --project-ref seu-project-ref
supabase db push
```

Deploy da Edge Function de calibração de regimes:

```bash
supabase functions deploy calibrate-regimes --no-verify-jwt
```

Disparar a calibração inicial:

```bash
curl -X POST https://seu-projeto.supabase.co/functions/v1/calibrate-regimes \
  -H "Authorization: Bearer SUA_SERVICE_ROLE_KEY" \
  -H "Content-Type: application/json" \
  -d '{}'
```

---

## Schema do Banco

```sql
-- Gestão de usuários
users (id, email, status, created_at)

-- Watchlists por usuário
watchlists (id, user_id, coin_id, created_at)

-- Alertas de preço
price_alerts (id, user_id, coin_id, target_price, direction, triggered, created_at)

-- Centroides de regime Wasserstein (atualizado semanalmente)
regime_centroids (symbol, bull_centroid float8[], bear_centroid float8[],
                  bull_vol, bear_vol, h1, calibrated_at)

-- Cache de respostas de API
api_cache (key, data, expires_at)

-- Dados de sentimento
sentiment_data (coin_id, score, source, created_at)
```

---

## Referências Acadêmicas

Os modelos quantitativos do Tradecraft Analytics são fundamentados em literatura acadêmica:

| Modelo | Paper |
|---|---|
| Wasserstein Regime Detector | Horvath, B., Issa, Z., & Muguruza, A. (2021). *Clustering Market Regimes Using the Wasserstein Distance.* SSRN 3947905 |
| Sinal Bar Portion (bot PMM) | Stoikov, S. (2024). *The bar portion signal.* |
| VPIN | Easley, D., López de Prado, M. & O'Hara, M. (2012). *Flow toxicity and liquidity in a high-frequency world.* Review of Financial Studies |
| Kalman Filter | Harvey, A. C. (1990). *Forecasting, Structural Time Series Models and the Kalman Filter.* Cambridge University Press |
| HMM Regime Switching | Hamilton, J. D. (1989). *A new approach to the economic analysis of nonstationary time series and the business cycle.* Econometrica |

---

## Licença

Proprietário. Todos os direitos reservados. Tradecraft Labs.

# GEX Dashboard

A real-time Gamma Exposure (GEX) dashboard for equities and index options. Streams live options data from Polygon.io, computes dealer gamma positioning across strikes and expirations, and surfaces key levels, alerts, and AI-generated trade ideas.

---

## Architecture

```
gex_dashboard.html   — single-file frontend (Plotly, SSE, REST polling)
gex_server.py        — FastAPI app, alert engine, SSE, background tasks
gex_core.py          — GEX computation, engine state, SQLite persistence
polygon_client.py    — all Polygon HTTP + WebSocket I/O
ai_client.py         — Anthropic API integration, MCP support, trade ideas
```

**Data flow:**
1. On startup, `fast_bootstrap()` fetches the full options chain snapshot from Polygon
2. Spot price seeds from yfinance (free, ~15min delayed) → Polygon 1-min agg bar if available
3. `PolygonOptionsStream` maintains a WebSocket connection to `wss://socket.polygon.io/options` for live OI/gamma updates
4. Background tasks recompute GEX every 5 minutes and push updates to connected clients via SSE
5. Alerts fire when spot crosses key levels (Zero Gamma, walls) and broadcast to all SSE subscribers

---

## Requirements

```bash
pip install fastapi uvicorn[standard] requests websocket-client \
            numpy python-dotenv yfinance anthropic
```

Python 3.10+ required. `zoneinfo` is built-in (3.9+); no timezone packages needed.

---

## Quick Start

```bash
# 1. Set API keys
export POLYGON_API_KEY=your_polygon_key
export ANTHROPIC_API_KEY=your_anthropic_key   # optional — only for trade ideas

# 2. Run
python gex_server.py

# 3. Open browser
open http://localhost:8050
```

---

## Configuration

All settings can be overridden via environment variables or a `.env` file.

### Required

| Variable | Description |
|---|---|
| `POLYGON_API_KEY` | Polygon.io API key. Options Starter plan ($29/mo) is sufficient. |

### Optional — server

| Variable | Default | Description |
|---|---|---|
| `GEX_TICKER` | `SPY` | Default ticker shown on load |
| `GEX_PORT` | `8050` | HTTP port |
| `GEX_MIN_DTE` | `0` | Minimum days-to-expiry for options chain |
| `GEX_MAX_DTE` | `45` | Maximum days-to-expiry for options chain |
| `GEX_REFRESH_SEC` | `300` | Auto-refresh interval in seconds (5 min) |
| `GEX_SNAPSHOT_INT` | `60` | WebSocket snapshot interval in seconds |
| `GEX_DB_PATH` | `gex_snapshots.db` | SQLite database path |
| `GEX_USE_WEBSOCKET` | `true` | Enable live WebSocket options stream |
| `GEX_NEAR_SPOT_PCT` | `0.10` | WS universe: subscribe to contracts within ±10% of spot |

### Optional — spot price polling

| Variable | Default | Description |
|---|---|---|
| `GEX_SPOT_POLL_SEC` | `60` | Outer polling loop interval |
| `GEX_AGG_POLL_SEC` | `30` | Polygon 1-min agg poll interval (market hours) |
| `GEX_WS_STALE_SEC` | `120` | Seconds before WS spot is considered stale and REST poll takes over |

### Optional — AI trade ideas

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required for trade ideas tab |
| `GEX_AI_MODEL` | `claude-sonnet-4-20250514` | Anthropic model to use |
| `GEX_AI_CACHE_TTL` | `600` | Trade ideas cache TTL in seconds |
| `GEX_MCP_SERVERS` | — | JSON array of MCP servers to attach: `[{"name":"exa","url":"https://..."}]` |

---

## Watchlist

Edit the `WATCHLIST` constant in `gex_server.py` (line ~63):

```python
WATCHLIST = ["SPY", "QQQ", "AAPL", "AMZN", "GOOGL", "NVDA", "META", "MSFT", "TSLA", "AVGO"]
```

### Index tickers (SPX, NDX, VIX, RUT)

Index tickers are supported with automatic symbol aliasing:

| Add to WATCHLIST | Polygon aggs | yfinance symbol |
|---|---|---|
| `SPX` | `I:SPX` | `^GSPC` |
| `NDX` | `I:NDX` | `^NDX` |
| `RUT` | `I:RUT` | `^RUT` |
| `VIX` | `I:VIX` | `^VIX` |

No code changes needed — just add the ticker string to `WATCHLIST`. The alias tables in `polygon_client.py` handle routing automatically.

---

## Spot Price Sources

The server resolves spot price using a priority chain on every poll cycle (every 30s during market hours):

1. **Stocks WebSocket** `wss://socket.polygon.io/stocks` — real-time trade prints. Requires a Polygon Stocks subscription (not included in Options Starter). Automatically disabled if auth fails; does not retry.
2. **yfinance** — free, ~15min delayed. Works for all tickers including indices (`^GSPC` etc.). Primary source on Options Starter plan.
3. **Polygon 1-min agg** `/v2/aggs` — ~1min lag. Works if Polygon plan delivers intraday bars.
4. **Polygon prev_close** — fallback seed at startup.

The source currently feeding spot is shown in the dashboard metrics panel as `● WS`, `● AGG`, or `● YF`.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Dashboard HTML |
| `GET` | `/api/gex/{ticker}` | Full GEX state: levels, strikes, bucketed, history |
| `POST` | `/api/gex/{ticker}/refresh` | Trigger full chain re-bootstrap in background |
| `GET` | `/api/gex/{ticker}/stream` | SSE stream: `spot`, `refresh`, `alerts` events |
| `GET` | `/api/gex/ALL/stream` | SSE stream: cross-ticker alert broadcasts |
| `GET` | `/api/spot/{ticker}` | Live spot price + source (lightweight, 5s poll) |
| `GET` | `/api/ohlc/{ticker}` | OHLC candles `?resolution=5&hours=24` |
| `GET` | `/api/alerts` | Live alert feed `?ticker=SPY&limit=50&hours=24` |
| `DELETE` | `/api/alerts/{alert_id}` | Dismiss a single alert |
| `DELETE` | `/api/alerts/ticker/{ticker}` | Clear all alerts for a ticker (CLR button) |
| `GET` | `/api/alerts/history` | Persistent alert log from DB `?ticker=SPY&days=7` |
| `GET` | `/api/watchlist` | Spot + regime snapshot for all watchlist tickers |
| `POST` | `/api/watchlist/{ticker}/load` | Pre-load a watchlist ticker in background |
| `GET` | `/api/trade-ideas/{ticker}` | AI-generated trade ideas (requires Anthropic key) |
| `GET` | `/api/levels-history/{ticker}` | Historical key levels `?days=5` |
| `GET` | `/api/status` | Server health: engines loaded, WS status, DB stats |
| `GET` | `/api/debug/ws/{ticker}` | WebSocket diagnostics: event counts, spot source, WS age |
| `GET` | `/api/debug/spot/{ticker}` | Spot endpoint availability matrix |

---

## Dashboard Views

| View | Description |
|---|---|
| **Profile** | GEX by strike — bar chart with key level overlays |
| **Chart + GEX** | Candlestick price chart (left) + GEX profile (right) |
| **Heatmap** | Strike × expiry GEX grid — color-coded intensity |
| **Matrix** | Tabular strike × expiry values with row/column totals |
| **Buckets** | GEX split into DTE buckets: 0-1d / 2-7d / 8-45d |
| **Research** | Historical key levels + alert log charts |
| **Trade Ideas** | AI analysis with current GEX context |

---

## Alerts

Alerts fire automatically when spot crosses key GEX levels. All alerts are:
- Stored in SQLite (`alerts_history` table)
- Broadcast via SSE to all connected clients
- Deduplicated — same ticker + alert type won't re-fire within 5 minutes
- Filtered to last 24 hours in the feed

| Alert type | Severity | Trigger |
|---|---|---|
| `regime_flip` | CRITICAL | Spot crosses Zero Gamma Level |
| `net_gex_flip` | WARNING | Net GEX polarity changes sign |
| `gamma_wall_near` | WARNING | Spot within 0.5% of Gamma Wall |
| `call_wall_near` | INFO | Spot within 0.5% of Call Wall |
| `call_wall_reject` | WARNING | Spot turns down from Call Wall |
| `call_wall_break` | CRITICAL | Spot breaks above Call Wall |
| `put_wall_near` | WARNING | Spot within 0.5% of Put Wall |
| `put_wall_breach` | CRITICAL | Spot breaks below Put Wall |

---

## Database

SQLite database at `GEX_DB_PATH` (default `gex_snapshots.db`). Three tables:

- **`snapshots`** — GEX-by-strike snapshots every `SNAPSHOT_INT` seconds. Used for the Research tab history charts.
- **`levels_history`** — Key levels (zero gamma, walls) timestamped. Used for research overlay charts.
- **`alerts_history`** — All fired alerts with full detail. Persists across restarts; restored into live feed on startup (last 24h only).

---

## Polygon Plan Notes

| Feature | Options Starter | Options Advanced | Stocks (add-on) |
|---|---|---|---|
| Options chain snapshot | ✓ | ✓ | — |
| Options WebSocket | ✓ (connected, 0 events) | ✓ | — |
| Intraday 1-min agg bars | ✗ (EOD only) | ✓ | ✓ |
| Stocks WebSocket (T.*) | ✗ | ✗ | ✓ |
| Index aggs (I:SPX) | ✓ | ✓ | ✓ |

On Options Starter, spot price is provided by **yfinance** (free, ~15min delay). All GEX computation and alert logic works correctly at this tier.

---

## File Structure

```
gex_server.py        1150 lines   FastAPI app, state cache, SSE, background tasks
gex_core.py           747 lines   GEX engine, computation, SQLite store
polygon_client.py     825 lines   Polygon HTTP + WebSocket clients
ai_client.py          473 lines   Anthropic API, MCP, trade ideas
gex_dashboard.html   1658 lines   Single-file frontend
gex_snapshots.db                  SQLite (auto-created on first run)
```

"""
gex_server.py — GEX Dashboard Server
======================================
python gex_server.py [--ticker SPY] [--port 8050] [--max-dte 45]

All Polygon HTTP calls live in polygon_client.py.
This file owns: FastAPI app, state cache, alert engine, SSE, background tasks.

Env:
    POLYGON_API_KEY, GEX_TICKER, GEX_PORT, GEX_MIN_DTE, GEX_MAX_DTE,
    GEX_REFRESH_SEC, GEX_SNAPSHOT_INT, GEX_DB_PATH, GEX_USE_WEBSOCKET
"""

import argparse
import asyncio
import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

from gex_core import (
    GEXEngine, PolygonOptionsStream, SnapshotStore,
    fast_bootstrap, compute_key_levels, log,
)
from polygon_client import (
    get_spot, get_spot_yfinance, poll_spot_agg, get_ohlc, diagnose_spot_endpoints, _is_market_open,
)
from ai_client_anth import generate_trade_ideas, get_cache_info, load_mcp_servers

load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_TICKER = os.getenv("GEX_TICKER",          "SPY").upper()
PORT           = int(os.getenv("GEX_PORT",         "8050"))
MIN_DTE        = int(os.getenv("GEX_MIN_DTE",      "0"))
MAX_DTE        = int(os.getenv("GEX_MAX_DTE",      "45"))
REFRESH_SEC    = int(os.getenv("GEX_REFRESH_SEC",  "300"))
SNAPSHOT_INT   = int(os.getenv("GEX_SNAPSHOT_INT", "60"))
DB_PATH        = os.getenv("GEX_DB_PATH",          "gex_snapshots.db")
USE_WS         = os.getenv("GEX_USE_WEBSOCKET",    "true").lower() == "true"
NEAR_SPOT_PCT  = float(os.getenv("GEX_NEAR_SPOT_PCT", "0.10"))
SPOT_POLL_SEC  = int(os.getenv("GEX_SPOT_POLL_SEC",   "60"))

WATCHLIST = ["SPY", "QQQ", "AAPL", "AMZN", "GOOGL", "NVDA", "META", "MSFT", "TSLA", "AVGO"]

_store = SnapshotStore(DB_PATH)


# ---------------------------------------------------------------------------
# Alert Engine
# ---------------------------------------------------------------------------

MAX_ALERTS = 200

class AlertEngine:
    """
    Monitors GEX engines for key level crossings and emits structured alerts.

    Alert types:
      regime_flip      — spot crossed Zero Gamma Level
      net_gex_flip     — net GEX polarity changed
      gamma_wall_near  — spot within 0.5% of Gamma Wall
      call_wall_near   — spot approaching Call Wall
      call_wall_reject — rejected at Call Wall
      call_wall_break  — broke above Call Wall (gamma squeeze)
      put_wall_near    — spot approaching Put Wall
      put_wall_breach  — broke below Put Wall (gamma-accelerated downside)
    """

    def __init__(self):
        self.alerts:      deque                = deque(maxlen=MAX_ALERTS)
        self._prev_state: Dict[str, dict]      = {}

    def check(self, ticker: str, engine: GEXEngine, levels: dict) -> List[dict]:
        if not levels or engine.spot_price <= 0:
            return []

        spot     = engine.spot_price
        now_iso  = datetime.utcnow().isoformat() + "Z"
        emitted  = []
        prev     = self._prev_state.get(ticker, {})

        def _alert(atype, severity, msg, extra=None, cooldown_s: int = 300):
            # Dedup: same ticker+type within cooldown window → skip silently
            if self._is_duplicate(ticker, atype, cooldown_s):
                return None
            a = {
                "id":       f"{ticker}-{atype}-{int(time.time())}",
                "time":     now_iso,
                "ticker":   ticker,
                "type":     atype,
                "severity": severity,
                "msg":      msg,
                "spot":     round(spot, 2),
                **(extra or {}),
            }
            self.alerts.appendleft(a)
            emitted.append(a)
            log.info("[ALERT] %s %s — %s (spot=%.2f)", severity, ticker, msg, spot)
            return a

        # ── Regime flip (spot crosses ZGL) ────────────────────────────────────
        zg = levels.get("zero_gamma")
        if zg:
            was_above = prev.get("above_zg")
            is_above  = spot > zg
            if was_above is not None and was_above != is_above:
                direction = "crossed ABOVE" if is_above else "crossed BELOW"
                regime    = "POSITIVE (vol suppression)" if is_above else "NEGATIVE (vol amplification)"
                _alert("regime_flip", "CRITICAL",
                       f"Spot {direction} Zero Gamma ${zg:.2f} → regime now {regime}",
                       {"zero_gamma": zg, "new_regime": levels.get("regime")})

        # ── Net GEX polarity flip ─────────────────────────────────────────────
        prev_regime = prev.get("regime")
        cur_regime  = levels.get("regime")
        if prev_regime and prev_regime != cur_regime:
            label = ("expect range-bound, lower vol" if cur_regime == "positive"
                     else "expect trending, higher vol")
            _alert("net_gex_flip", "WARNING",
                   f"Net GEX flipped to {cur_regime.upper()} (was {prev_regime}) — {label}",
                   {"net_gex_b": levels.get("net_gex_b")})

        # ── Gamma wall proximity ──────────────────────────────────────────────
        gw = levels.get("gamma_wall")
        if gw and abs(spot - gw) / spot <= 0.005 and not prev.get("near_gw"):
            _alert("gamma_wall_near", "WARNING",
                   f"Spot within 0.5% of Gamma Wall ${gw:.2f} — strong magnet/pin risk",
                   {"gamma_wall": gw, "dist_pct": round((spot - gw) / gw * 100, 3)})

        # ── Call wall ─────────────────────────────────────────────────────────
        cw = levels.get("call_wall")
        if cw:
            if abs(spot - cw) / spot <= 0.005 and not prev.get("near_cw"):
                _alert("call_wall_near", "INFO",
                       f"Approaching Call Wall ${cw:.2f} — dealer resistance zone",
                       {"call_wall": cw})
            was_below = prev.get("below_cw")
            is_below  = spot < cw
            if was_below is False and is_below:
                _alert("call_wall_reject", "WARNING",
                       f"Rejected at Call Wall ${cw:.2f} — overhead resistance holding",
                       {"call_wall": cw})
            elif was_below is True and not is_below:
                _alert("call_wall_break", "CRITICAL",
                       f"Broke ABOVE Call Wall ${cw:.2f} — potential gamma squeeze / vol collapse",
                       {"call_wall": cw})

        # ── Put wall ──────────────────────────────────────────────────────────
        pw = levels.get("put_wall")
        if pw:
            if abs(spot - pw) / spot <= 0.005 and not prev.get("near_pw"):
                _alert("put_wall_near", "WARNING",
                       f"Approaching Put Wall ${pw:.2f} — key dealer support / acceleration level",
                       {"put_wall": pw})
            was_above = prev.get("above_pw")
            is_above  = spot > pw
            if was_above is True and not is_above:
                _alert("put_wall_breach", "CRITICAL",
                       f"Breached PUT WALL ${pw:.2f} — gamma-accelerated downside risk",
                       {"put_wall": pw})

        self._prev_state[ticker] = {
            "above_zg": spot > zg   if zg else None,
            "regime":   cur_regime,
            "near_gw":  gw and abs(spot - gw) / spot <= 0.005,
            "near_cw":  cw and abs(spot - cw) / spot <= 0.005,
            "near_pw":  pw and abs(spot - pw) / spot <= 0.005,
            "below_cw": cw and spot < cw,
            "above_pw": pw and spot > pw,
        }
        return emitted

    def get_alerts(self, ticker: Optional[str] = None, limit: int = 50) -> List[dict]:
        alerts = list(self.alerts)
        if ticker:
            alerts = [a for a in alerts if a["ticker"] == ticker]
        return alerts[:limit]

    def delete(self, alert_id: str) -> bool:
        """Remove a single alert by id. Returns True if found and removed."""
        before = len(self.alerts)
        self.alerts = deque(
            (a for a in self.alerts if a["id"] != alert_id),
            maxlen=MAX_ALERTS,
        )
        return len(self.alerts) < before

    def _is_duplicate(self, ticker: str, atype: str, cooldown_s: int = 300) -> bool:
        """
        True if the same ticker+type alert was emitted within cooldown_s seconds.
        Prevents the same near-wall alert firing on every 5-min refresh.
        """
        cutoff = time.time() - cooldown_s
        for a in self.alerts:
            if a["ticker"] == ticker and a["type"] == atype:
                try:
                    ts = datetime.fromisoformat(a["time"].rstrip("Z")).timestamp()
                    if ts > cutoff:
                        return True
                except Exception:
                    pass
        return False


_alert_engine = AlertEngine()


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

@dataclass
class ComputedState:
    """Pre-computed GEX output. Single writer (_update_state_cache), many readers."""
    ticker:           str   = ""
    computed_at:      str   = ""
    spot:             float = 0.0
    contracts:        int   = 0
    snapshots:        int   = 0
    levels:           dict  = dc_field(default_factory=dict)
    levels_by_bucket: dict  = dc_field(default_factory=dict)
    gex_bucketed:     dict  = dc_field(default_factory=dict)
    gamma_flips:      list  = dc_field(default_factory=list)
    bar:              dict  = dc_field(default_factory=dict)
    by_expiry:        dict  = dc_field(default_factory=dict)
    heatmap:          dict  = dc_field(default_factory=dict)
    spot_history:     list  = dc_field(default_factory=list)
    net_history:      list  = dc_field(default_factory=list)
    last_updated:     str   = ""


_state_cache:     Dict[str, ComputedState]        = {}
_engines:         Dict[str, GEXEngine]            = {}
_streams:         Dict[str, PolygonOptionsStream] = {}
_refresh_tasks:   Dict[str, asyncio.Task]         = {}
_refresh_locks:   Dict[str, asyncio.Lock]         = {}
_sse_queues:      Dict[str, Set[asyncio.Queue]]   = defaultdict(set)
_net_gex_history: Dict[str, deque]               = defaultdict(lambda: deque(maxlen=480))
_spot_history:    Dict[str, deque]               = defaultdict(lambda: deque(maxlen=1440))


def _make_engine(ticker: str) -> GEXEngine:
    e = GEXEngine(store=_store, underlying=ticker.upper())
    n = e.restore_from_store()
    if n:
        log.info("[%s] Restored %d snapshots from DB", ticker, n)
    return e


def _run_bootstrap(ticker: str) -> bool:
    ticker = ticker.upper()
    try:
        e = _engines.get(ticker) or _make_engine(ticker)
        _engines[ticker] = e
        fast_bootstrap(e, ticker, MIN_DTE, MAX_DTE)
        # Mark bootstrap price as seed so poll/WS can freely override it
        if e.spot_source in ("", "seed"):
            e.spot_source = "seed"

        # If spot is still 0 after bootstrap (e.g. all Polygon sources failed),
        # force yfinance — essential for index tickers like SPX on Options Starter.
        if e.spot_price <= 0:
            log.warning("[%s] spot=0 after bootstrap — forcing yfinance lookup", ticker)
            from polygon_client import get_spot_yfinance
            spot = get_spot_yfinance(ticker)
            if spot > 0:
                e.update_spot(spot, source="yfinance")
                log.info("[%s] spot recovered via yfinance: %.2f", ticker, spot)

        if e.spot_price <= 0 or not e.contract_data:
            log.error("[%s] Bootstrap produced no usable data (spot=%.2f, contracts=%d)",
                      ticker, e.spot_price, len(e.contract_data))
            return False

        cs = _update_state_cache(e, ticker)
        _net_gex_history[ticker].append({
            "ts": cs.computed_at, "net_gex": cs.levels.get("net_gex_b", 0), "spot": cs.spot,
        })

        new_alerts = _alert_engine.check(ticker, e, cs.levels)
        for a in new_alerts:
            try:
                _store.insert_alert(a)
            except Exception:
                pass

        return True
    except Exception as ex:
        log.error("[%s] Bootstrap error: %s", ticker, ex, exc_info=True)
        return False


async def _ensure_bootstrapped(ticker: str) -> bool:
    """
    Single entry point for all bootstrap calls.

    Guarantees:
      - Only ONE engine is ever created per ticker (no overwrite races)
      - Only ONE bootstrap runs at a time per ticker (lock-protected)
      - Concurrent callers (get_gex + watchlist load) wait behind the lock
        then return immediately once the first caller finishes
      - Safe to call from both foreground (get_gex) and background (watchlist load)
    """
    ticker = ticker.upper()

    # Fast path — already loaded, skip lock entirely
    e = _engines.get(ticker)
    if e and e.contract_data:
        return True

    # Ensure engine exists before acquiring lock
    # setdefault is atomic for dict in CPython — only one engine created
    if ticker not in _engines:
        _engines[ticker] = _make_engine(ticker)

    lock = _refresh_locks.setdefault(ticker, asyncio.Lock())

    # If locked, another caller is bootstrapping — wait then check again
    async with lock:
        e = _engines.get(ticker)
        if e and e.contract_data:
            return True          # done while we waited

        try:
            ok = await asyncio.get_event_loop().run_in_executor(
                None, _run_bootstrap, ticker)
        except asyncio.CancelledError:
            return False

        if ok and ticker not in _refresh_tasks:
            _refresh_tasks[ticker] = asyncio.create_task(_auto_refresh_loop(ticker))
            asyncio.create_task(_spot_poll_loop(ticker))

        return ok


async def _broadcast(ticker: str, payload: dict):
    dead: Set[asyncio.Queue] = set()
    for q in list(_sse_queues.get(ticker, set())):
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            dead.add(q)
    _sse_queues[ticker] -= dead
    for q in list(_sse_queues.get("ALL", set())):
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass


# ---------------------------------------------------------------------------
# Background tasks
# ---------------------------------------------------------------------------

# How stale a WS spot must be before the REST poll is allowed to overwrite it.
# During market hours the WS sends a T event on every trade — 120s means the
# REST poll only kicks in if the WS has been silent for 2 full minutes.
WS_SPOT_STALE_SEC = int(os.getenv("GEX_WS_STALE_SEC", "120"))


# Agg poll interval — tight loop for Polygon 1-min bar
AGG_POLL_SEC = int(os.getenv("GEX_AGG_POLL_SEC", "30"))


def _record_spot(ticker_up: str, price: float) -> None:
    """Append a spot reading to history deque."""
    _spot_history[ticker_up].append({
        "ts":   datetime.utcnow().isoformat() + "Z",
        "spot": round(price, 2),
    })


async def _broadcast_spot(ticker: str, price: float, source: str) -> None:
    """Push a lightweight spot update to all SSE subscribers for this ticker."""
    await _broadcast(ticker.upper(), {
        "event":  "spot",
        "spot":   round(price, 4),
        "source": source,
        "ts":     datetime.utcnow().isoformat() + "Z",
    })


async def _spot_poll_loop(ticker: str):
    """
    Spot price polling loop.  Runs every AGG_POLL_SEC (default 30s).

    Source priority (highest → lowest):
      1. WS T.{ticker} trade event (source="ws")  — set directly by PolygonOptionsStream
         via engine.update_spot(price, source="ws") + engine.ws_spot_ts.
         WS guard: if ws_spot_ts was set within WS_SPOT_STALE_SEC AND the stock
         stream is still marked running, skip REST polling to avoid overwriting.
         IMPORTANT: ws_spot_ts is cleared to 0 by on_auth_failed when the stocks
         WS is unavailable (Options Starter plan) so this guard NEVER fires for
         free-tier users.

      2. yfinance (every tick, ~15min delayed, free, no plan restriction)
         Always tried first during market hours so free-plan users get a live price.

      3. Polygon 1-min agg (tried after yfinance; works on paid Stocks plan)
         ~1min lag.  Returns 0 on Options Starter — treated as a miss, not an error.

    State cache recompute: whenever spot moves >0.1% the GEX bars (which use
    S² in their formula) are recomputed so profile/chart views stay accurate.
    """
    ticker_up = ticker.upper()
    loop      = asyncio.get_event_loop()

    # Seed history from bootstrap value, then wait for first poll
    await asyncio.sleep(3)
    engine = _engines.get(ticker_up)
    if engine and engine.spot_price > 0:
        _record_spot(ticker_up, engine.spot_price)

    prev_spot_for_cache = 0.0  # track when to trigger recompute

    while True:
        sleep_sec = AGG_POLL_SEC if _is_market_open() else 300
        await asyncio.sleep(sleep_sec)
        engine = _engines.get(ticker_up)
        if not engine:
            continue

        # ── WS guard: only skip if WS is genuinely active ─────────────────────
        # ws_spot_ts == 0 means either WS never connected or auth failed and
        # on_auth_failed cleared it.  In that case always run the REST poll.
        if engine.ws_spot_ts > 0:
            ws_age = time.time() - engine.ws_spot_ts
            if ws_age < WS_SPOT_STALE_SEC:
                _record_spot(ticker_up, engine.spot_price)
                log.debug("[%s] spot poll skipped — WS active (%.0fs old)", ticker, ws_age)
                continue

        spot   = 0.0
        source = ""

        # ── yfinance: always first — works on all plans, no API restrictions ──
        if _is_market_open():
            try:
                spot = await loop.run_in_executor(None, get_spot_yfinance, ticker_up)
                if spot > 0:
                    source = "yfinance"
            except Exception as e:
                log.debug("[%s] yfinance error: %s", ticker, e)

        # ── Polygon agg: secondary — ~1min lag, zero on Starter plan ─────────
        if spot <= 0 and _is_market_open():
            try:
                spot = await loop.run_in_executor(None, poll_spot_agg, ticker_up)
                if spot > 0:
                    source = "agg"
            except Exception as e:
                log.debug("[%s] agg poll error: %s", ticker, e)

        # ── Apply update ───────────────────────────────────────────────────────
        if spot > 0:
            engine.update_spot(spot, source=source)
            _record_spot(ticker_up, spot)
            asyncio.create_task(_broadcast_spot(ticker_up, spot, source))

            # Recompute GEX cache if spot moved more than 0.1%
            if prev_spot_for_cache > 0:
                move_pct = abs(spot - prev_spot_for_cache) / prev_spot_for_cache * 100
                if move_pct >= 0.10:
                    try:
                        cs = _update_state_cache(engine, ticker_up)
                        log.debug("[%s] state cache recomputed — spot moved %.2f%%",
                                  ticker, move_pct)
                        # Check for new alerts after spot-driven recompute
                        new_alerts = _alert_engine.check(ticker_up, engine, cs.levels)
                        for a in new_alerts:
                            try: _store.insert_alert(a)
                            except Exception: pass
                        if new_alerts:
                            asyncio.create_task(_broadcast(ticker_up,
                                {"event": "alerts", "alerts": new_alerts}))
                            asyncio.create_task(_broadcast("ALL",
                                {"event": "alerts", "alerts": new_alerts}))
                    except Exception as e:
                        log.debug("[%s] cache recompute error: %s", ticker, e)
            prev_spot_for_cache = spot
        else:
            # Nothing worked — record existing price to keep history continuous
            if engine.spot_price > 0:
                _record_spot(ticker_up, engine.spot_price)
            log.debug("[%s] spot poll: all sources returned 0", ticker)


async def _auto_refresh_loop(ticker: str):
    while True:
        await asyncio.sleep(REFRESH_SEC)
        lock = _refresh_locks.setdefault(ticker, asyncio.Lock())
        if lock.locked():
            continue
        if not _is_market_open() and ticker not in _streams:
            log.debug("[%s] Auto-refresh skipped — market closed", ticker)
            continue
        async with lock:
            log.info("[%s] Auto-refresh", ticker)
            loop = asyncio.get_event_loop()
            ok   = await loop.run_in_executor(None, _run_bootstrap, ticker)
            if ok and ticker in _engines:
                data = _engine_to_dict(_engines[ticker], ticker)
                await _broadcast(ticker, {"event": "refresh", "data": data})
                # Broadcast fresh alerts — new ones from _run_bootstrap check
                fresh_alerts = _alert_engine.get_alerts(ticker, limit=5)
                if fresh_alerts:
                    await _broadcast(ticker, {"event": "alerts", "alerts": fresh_alerts})
                    await _broadcast("ALL",  {"event": "alerts", "alerts": fresh_alerts})


def _start_ws(ticker: str):
    engine = _engines.get(ticker)
    if not engine or not engine.contract_data:
        log.warning("[%s] Cannot start WS — no contracts loaded", ticker)
        return
    import pandas as pd
    rows = [{"ticker": ot, "strike": c["strike"], "type": c["type"],
              "oi": c["oi"], "expiration": None}
            for ot, c in engine.contract_data.items()]
    contracts_df = pd.DataFrame(rows)
    stream = PolygonOptionsStream(
        engine=engine, contracts_df=contracts_df, ticker=ticker,
        snapshot_int=SNAPSHOT_INT, near_spot_pct=NEAR_SPOT_PCT)
    threading.Thread(target=stream.run, name=f"ws-{ticker}", daemon=True).start()
    _streams[ticker] = stream
    log.info("[%s] WebSocket thread started", ticker)


# ---------------------------------------------------------------------------
# State cache
# ---------------------------------------------------------------------------

def _ts_iso(ts) -> str:
    return ts.isoformat() if hasattr(ts, "isoformat") else str(ts)


def _update_state_cache(engine: GEXEngine, ticker: str) -> ComputedState:
    """Single writer: compute everything once, store in _state_cache."""
    ticker  = ticker.upper()
    t0      = time.time()

    gex_map      = engine.compute_gex_by_strike()
    gex_bucketed = engine.compute_gex_bucketed()
    levels       = compute_key_levels(gex_map, engine.spot_price)

    levels_by_bucket = {
        bkt: compute_key_levels(bmap, engine.spot_price)
        for bkt, bmap in gex_bucketed.items() if bmap
    }
    bucketed_net = {
        bkt: round(sum(v for v in bmap.values()) / 1e9, 6)
        for bkt, bmap in gex_bucketed.items()
    }

    sorted_strikes = sorted(gex_map.keys())
    flips: List[float] = []
    if sorted_strikes:
        vals = [gex_map[s] for s in sorted_strikes]
        cum  = list(np.cumsum(vals))
        raw  = [sorted_strikes[i] for i in range(1, len(cum)) if cum[i-1] * cum[i] < 0]
        if raw and engine.spot_price:
            raw.sort(key=lambda x: abs(x - engine.spot_price))
            flips = [round(f, 2) for f in raw[:3]]

    times, hm_strikes, matrix, spots = engine.get_matrix()
    times_iso = [_ts_iso(t) for t in times]
    last_ts   = _ts_iso(engine.history[-1]["timestamp"]) if engine.history else ""

    by_expiry_raw = engine.compute_gex_by_expiry_strike()
    by_expiry = {
        exp: {format(s,'g'): round(v / 1e9, 6) for s, v in smap.items()}
        for exp, smap in by_expiry_raw.items()
    }

    _raw_sh  = list(_spot_history.get(ticker, []))
    spot_history = _raw_sh if _raw_sh else [
        {"ts": _ts_iso(h["timestamp"]), "spot": float(h["spot"])}
        for h in engine.history
    ]
    net_history = list(_net_gex_history.get(ticker, []))

    if _store and levels:
        _store.insert_levels(ticker, datetime.utcnow().isoformat(),
                             engine.spot_price, levels, bucketed_net)

    gex_bucketed_b = {
        bkt: {format(s,'g'): round(v / 1e9, 6) for s, v in bmap.items()}
        for bkt, bmap in gex_bucketed.items()
    }

    cs = ComputedState(
        ticker=ticker, computed_at=datetime.utcnow().isoformat(),
        spot=round(engine.spot_price, 2), contracts=len(engine.contract_data),
        snapshots=len(engine.history), levels=levels,
        levels_by_bucket=levels_by_bucket, gex_bucketed=gex_bucketed_b,
        gamma_flips=flips,
        bar={"strikes": [round(s, 2) for s in sorted_strikes],
             "gex_b":   [round(gex_map[s] / 1e9, 6) for s in sorted_strikes]},
        by_expiry=by_expiry,
        heatmap={"times":   times_iso,
                 "strikes": [round(s, 2) for s in hm_strikes],
                 "matrix":  matrix.tolist() if hasattr(matrix, "tolist") else matrix,
                 "spots":   [float(s) for s in spots]},
        spot_history=spot_history, net_history=net_history, last_updated=last_ts,
    )
    _state_cache[ticker] = cs
    log.debug("[%s] state cache updated in %.2fs", ticker, time.time() - t0)
    return cs


def _state_to_dict(cs: ComputedState) -> dict:
    return {
        "ticker":           cs.ticker,     "spot":             cs.spot,
        "contracts":        cs.contracts,  "snapshots":        cs.snapshots,
        "last_updated":     cs.last_updated, "computed_at":    cs.computed_at,
        "levels":           cs.levels,     "levels_by_bucket": cs.levels_by_bucket,
        "gamma_flips":      cs.gamma_flips, "spot_history":    cs.spot_history,
        "by_expiry":        cs.by_expiry,   "gex_bucketed":    cs.gex_bucketed,
        "bar":              cs.bar,         "heatmap":         cs.heatmap,
        "net_history":      cs.net_history,
    }


def _engine_to_dict(engine: GEXEngine, ticker: str,
                    hours_back: Optional[float] = None) -> dict:
    """
    Return API-ready dict for a ticker.

    GEX computation (strikes, levels, bucketed) comes from _state_cache —
    it is expensive and only rebuilt on bootstrap/refresh.

    Spot price is ALWAYS taken live from engine.spot_price so the dashboard
    reflects the most recent poll/WS update rather than the cached value.
    """
    ticker = ticker.upper()
    cs     = _state_cache.get(ticker) or _update_state_cache(engine, ticker)

    # Patch live spot into the cached state so API always returns current price.
    # Also update levels that depend on spot (dist_to_flip_pct).
    if engine.spot_price > 0 and abs(engine.spot_price - cs.spot) > 0.005:
        cs.spot = engine.spot_price
        if cs.levels and cs.levels.get("zero_gamma"):
            zg = cs.levels["zero_gamma"]
            cs.levels["dist_to_flip_pct"] = round(
                (engine.spot_price - zg) / engine.spot_price * 100, 3)

    d = _state_to_dict(cs)
    # Inject live spot_source for UI
    d["spot_source"] = engine.spot_source
    return d


# ---------------------------------------------------------------------------
# FastAPI app + lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app):
    # ── startup ───────────────────────────────────────────────────────────────
    ticker = DEFAULT_TICKER
    e = _make_engine(ticker)
    _engines[ticker] = e
    loop = asyncio.get_event_loop()
    # Uncomment the next line for one-time endpoint diagnosis, then remove:
    # await loop.run_in_executor(None, diagnose_spot_endpoints, ticker)
    ok = await loop.run_in_executor(None, _run_bootstrap, ticker)
    if ok:
        if USE_WS: _start_ws(ticker)
        _refresh_tasks[ticker] = asyncio.create_task(_auto_refresh_loop(ticker))
        asyncio.create_task(_spot_poll_loop(ticker))
        _restore_alerts_from_db()
    else:
        log.warning("[%s] Startup bootstrap failed", ticker)

    try:
        yield
    except (asyncio.CancelledError, Exception):
        pass  # uvicorn cancels lifespan on shutdown — not an error
    finally:
        # ── shutdown ──────────────────────────────────────────────────────────
        log.info("Shutdown — closing streams and saving state...")
        for task in list(_refresh_tasks.values()):
            if not task.done():
                task.cancel()
        _refresh_tasks.clear()
        for stream in list(_streams.values()):
            try:
                stream.close()
            except Exception:
                pass
        _streams.clear()
        for ticker_k, engine in list(_engines.items()):
            try:
                engine.snapshot()
                log.info("[%s] Final snapshot saved", ticker_k)
            except Exception as e:
                log.debug("[%s] Final snapshot error: %s", ticker_k, e)
        log.info("Shutdown complete.")


app = FastAPI(title="GEX Dashboard", version="4.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    p = Path(__file__).parent / "gex_dashboard.html"
    if not p.exists():
        return HTMLResponse("<h2>gex_dashboard.html not found</h2>", 404)
    return HTMLResponse(p.read_text(encoding="utf-8"))


@app.get("/api/gex/{ticker}")
async def get_gex(ticker: str, hours: Optional[float] = None):
    ticker = ticker.upper()
    if not (_engines.get(ticker) and _engines[ticker].contract_data):
        try:
            ok = await _ensure_bootstrapped(ticker)
        except asyncio.CancelledError:
            return JSONResponse({"error": "server shutting down"}, status_code=503)
        if not ok:
            from fastapi import HTTPException
            raise HTTPException(503, detail=f"Bootstrap failed for {ticker}")
    return _engine_to_dict(_engines[ticker], ticker, hours_back=hours)


@app.post("/api/gex/{ticker}/refresh")
async def trigger_refresh(ticker: str, background_tasks: BackgroundTasks):
    ticker = ticker.upper()
    async def _do():
        lock = _refresh_locks.setdefault(ticker, asyncio.Lock())
        if lock.locked(): return
        async with lock:
            ok = await asyncio.get_event_loop().run_in_executor(None, _run_bootstrap, ticker)
            if ok and ticker in _engines:
                data = _engine_to_dict(_engines[ticker], ticker)
                await _broadcast(ticker, {"event": "refresh", "data": data})
                # Broadcast fresh alerts — new ones from _run_bootstrap check
                fresh_alerts = _alert_engine.get_alerts(ticker, limit=5)
                if fresh_alerts:
                    await _broadcast(ticker, {"event": "alerts", "alerts": fresh_alerts})
                    await _broadcast("ALL",  {"event": "alerts", "alerts": fresh_alerts})
    background_tasks.add_task(_do)
    return {"status": "refreshing", "ticker": ticker}


@app.get("/api/gex/{ticker}/stream")
async def sse_stream(ticker: str, request: Request):
    ticker = ticker.upper()
    queue: asyncio.Queue = asyncio.Queue(maxsize=20)
    _sse_queues[ticker].add(queue)

    async def gen():
        try:
            yield 'data: {"event":"connected"}\n\n'
            while True:
                if await request.is_disconnected(): break
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=30)
                    yield f"data: {json.dumps(payload)}\n\n"
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
        finally:
            _sse_queues[ticker].discard(queue)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no",
                                      "Access-Control-Allow-Origin": "*"})


@app.get("/api/ohlc/{ticker}")
async def get_ohlc_endpoint(ticker: str, resolution: str = "5", hours: float = 6):
    """OHLC candle bars for the price chart. resolution = bar size in minutes."""
    ticker = ticker.upper()
    try:
        res  = int(resolution) if str(resolution).isdigit() else 5
        loop = asyncio.get_event_loop()
        bars = await loop.run_in_executor(None, get_ohlc, ticker, res, hours)
        return {"ticker": ticker, "resolution": res, "bars": bars}
    except Exception as e:
        log.error("[ohlc] %s error: %s", ticker, e)
        return {"ticker": ticker, "resolution": 5, "bars": []}


@app.get("/api/alerts")
async def get_alerts(ticker: Optional[str] = None, limit: int = 50, hours: int = 24):
    t      = ticker.upper() if ticker else None
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    alerts = _alert_engine.get_alerts(ticker=t, limit=limit)
    result = []
    for a in alerts:
        # Normalize time field — ensure Z suffix so JS parses as UTC
        ts_str = a.get("time", "")
        if ts_str and not ts_str.endswith("Z"):
            ts_str += "Z"
        try:
            ts = datetime.fromisoformat(ts_str.rstrip("Z"))
            if ts < cutoff:
                continue          # older than requested window — skip
        except Exception:
            pass                  # malformed ts — include anyway
        result.append({**a, "time": ts_str})
    return {"alerts": result}


@app.delete("/api/alerts/{alert_id}")
async def delete_alert(alert_id: str):
    """Delete a single alert from the live feed (does not remove from DB history)."""
    found = _alert_engine.delete(alert_id)
    if not found:
        from fastapi import HTTPException
        raise HTTPException(404, detail=f"Alert {alert_id!r} not found")
    return {"deleted": alert_id}


@app.delete("/api/alerts/ticker/{ticker}")
async def clear_ticker_alerts(ticker: str):
    """Remove all live alerts for a specific ticker (CLR button)."""
    ticker_up = ticker.upper()
    before = len(_alert_engine.alerts)
    _alert_engine.alerts = deque(
        (a for a in _alert_engine.alerts if a["ticker"] != ticker_up),
        maxlen=MAX_ALERTS,
    )
    removed = before - len(_alert_engine.alerts)
    return {"cleared": removed, "ticker": ticker_up}


@app.get("/api/spot/{ticker}")
async def get_live_spot(ticker: str):
    """Lightweight spot-only endpoint — polled every few seconds by the dashboard."""
    ticker = ticker.upper()
    engine = _engines.get(ticker)
    if not engine:
        from fastapi import HTTPException
        raise HTTPException(404, detail=f"{ticker} not loaded")
    return {
        "ticker":       ticker,
        "spot":         round(engine.spot_price, 4),
        "spot_source":  engine.spot_source,
        "ws_spot_age":  round(time.time() - engine.ws_spot_ts, 1) if engine.ws_spot_ts else None,
        "ts":           datetime.utcnow().isoformat() + "Z",
    }


@app.get("/api/watchlist")
async def get_watchlist():
    """Quick spot + regime snapshot for all watchlist tickers."""
    result = []
    for t in WATCHLIST:
        e = _engines.get(t)
        if e and e.spot_price > 0 and e.contract_data:
            gex_map = e.compute_gex_by_strike()
            levels  = compute_key_levels(gex_map, e.spot_price)
            result.append({
                "ticker":     t,
                "spot":       round(e.spot_price, 2),
                "net_gex_b":  levels.get("net_gex_b", 0),
                "regime":     levels.get("regime", "unknown"),
                "strength":   levels.get("regime_strength", 0),
                "zero_gamma": levels.get("zero_gamma"),
                "gamma_wall": levels.get("gamma_wall"),
                "loaded":     True,
            })
        else:
            result.append({"ticker": t, "loaded": False})
    return {"watchlist": result}


@app.post("/api/watchlist/{ticker}/load")
async def load_watchlist_ticker(ticker: str, background_tasks: BackgroundTasks):
    """Pre-load a watchlist ticker in background."""
    ticker = ticker.upper()
    if ticker not in WATCHLIST:
        from fastapi import HTTPException
        raise HTTPException(400, detail=f"{ticker} not in watchlist")

    async def _do():
        await _ensure_bootstrapped(ticker)

    background_tasks.add_task(_do)
    return {"status": "loading", "ticker": ticker}


# ---------------------------------------------------------------------------
# Trade Idea Engine
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Trade Idea endpoint — logic lives in ai_client.py
# ---------------------------------------------------------------------------

@app.get("/api/trade-ideas/{ticker}")
async def get_trade_ideas(
    ticker:    str,
    force:     bool = False,
    mcp_extra: Optional[str] = None,  # JSON list of {name,url} — merged with GEX_MCP_SERVERS env
):
    """
    AI-powered options trade ideas based on current GEX data.

    Query params:
      force      — bypass 10-min cache and regenerate
      mcp_extra  — JSON string: additional MCP servers for this call
                   e.g. ?mcp_extra=[{"name":"exa","url":"https://exa.mcp.example.com/sse"}]

    MCP servers are also configurable globally via GEX_MCP_SERVERS env var.
    See ai_client.py for full documentation.
    """
    ticker = ticker.upper()
    engine = _engines.get(ticker)
    if not engine or not engine.contract_data:
        from fastapi import HTTPException
        raise HTTPException(404, detail=f"{ticker} not loaded — call /api/gex/{ticker} first")

    data   = _engine_to_dict(engine, ticker)
    alerts = _alert_engine.get_alerts(ticker=ticker, limit=5)

    extra_servers = None
    if mcp_extra:
        try:
            extra_servers = json.loads(mcp_extra)
        except json.JSONDecodeError:
            from fastapi import HTTPException
            raise HTTPException(400, detail="mcp_extra must be valid JSON array")

    try:
        result = await generate_trade_ideas(
            gex_data  = data,
            alerts    = alerts,
            ticker    = ticker,
            force     = force,
            mcp_extra = extra_servers,
        )
        return result
    except ValueError as e:
        from fastapi import HTTPException
        raise HTTPException(500, detail=str(e))
    except Exception as e:
        log.error("[trade-ideas] %s: %s", ticker, e, exc_info=True)
        from fastapi import HTTPException
        raise HTTPException(502, detail=f"AI generation failed: {e}")


# ---------------------------------------------------------------------------
# Utility endpoints
# ---------------------------------------------------------------------------

@app.get("/api/spot/{ticker}")
async def get_spot_history(ticker: str, limit: int = 100):
    """Spot price history + poll diagnostics."""
    ticker = ticker.upper()
    hist   = list(_spot_history.get(ticker, []))
    engine = _engines.get(ticker)
    freq   = None
    if len(hist) >= 2:
        t0   = datetime.fromisoformat(hist[-2]["ts"])
        t1   = datetime.fromisoformat(hist[-1]["ts"])
        freq = round((t1 - t0).total_seconds(), 1)
    return {
        "ticker":          ticker,
        "current_spot":    round(engine.spot_price, 2) if engine else None,
        "poll_interval_s": SPOT_POLL_SEC,
        "data_points":     len(hist),
        "last_interval_s": freq,
        "source":          engine.spot_source if engine else "unknown",
        "ws_active":       ticker in _streams,
        "ws_spot_age_s":   round(time.time() - engine.ws_spot_ts, 1) if engine and engine.ws_spot_ts > 0 else None,
        "market_open":     _is_market_open(),
        "history":         hist[-limit:],
    }


def _restore_alerts_from_db():
    try:
        rows         = _store.load_alerts_history(days=1, limit=100)
        existing_ids = {x["id"] for x in _alert_engine.alerts}
        restored     = 0
        for a in reversed(rows):
            if a.get("id") not in existing_ids:
                # Normalize time to UTC Z format
                ts_str = a.get("time", "")
                if ts_str and not ts_str.endswith("Z"):
                    a = {**a, "time": ts_str + "Z"}
                _alert_engine.alerts.appendleft(a)
                restored += 1
        log.info("Restored %d alerts from DB (last 24h)", restored)
    except Exception as e:
        log.warning("Alert restore: %s", e)


@app.get("/api/levels-history/{ticker}")
async def get_levels_history(ticker: str, days: int = 5):
    """Historical key levels for research."""
    rows = _store.load_levels_history(ticker.upper(), days=days)
    return {"ticker": ticker.upper(), "days": days, "records": len(rows), "history": rows}


@app.get("/api/alerts/history")
async def get_alerts_history_ep(ticker: Optional[str] = None, days: int = 7, limit: int = 200):
    """Persistent alert log from DB."""
    rows = _store.load_alerts_history(
        underlying=ticker.upper() if ticker else None, days=days, limit=limit)
    return {"alerts": rows, "count": len(rows)}


@app.get("/api/state/{ticker}")
async def get_state(ticker: str):
    """Raw ComputedState for debugging."""
    ticker = ticker.upper()
    cs     = _state_cache.get(ticker)
    if not cs:
        from fastapi import HTTPException
        raise HTTPException(404, detail=f"{ticker} not cached")
    age = round((datetime.utcnow() - datetime.fromisoformat(cs.computed_at)).total_seconds(), 1)
    return {
        "ticker": cs.ticker, "computed_at": cs.computed_at, "cache_age_s": age,
        "spot": cs.spot, "contracts": cs.contracts,
        "levels": cs.levels, "levels_by_bucket": cs.levels_by_bucket,
        "gex_bucketed_net": {
            bkt: round(sum(float(v) for v in bmap.values()), 6)
            for bkt, bmap in cs.gex_bucketed.items()
        },
    }


@app.get("/api/debug/spot/{ticker}")
async def debug_spot(ticker: str):
    """Run spot endpoint diagnosis and return current spot. See polygon_client.diagnose_spot_endpoints."""
    ticker = ticker.upper()
    loop   = asyncio.get_event_loop()
    await loop.run_in_executor(None, diagnose_spot_endpoints, ticker)
    spot   = await loop.run_in_executor(None, get_spot, ticker)
    return {"ticker": ticker, "spot": spot, "see_logs": "diagnose output written to server log"}


@app.get("/api/debug/ws/{ticker}")
async def debug_ws(ticker: str):
    """
    Live WebSocket diagnostics — call this to see if T events are arriving.
    Returns event counts, spot source, WS age, and subscription info.
    """
    ticker  = ticker.upper()
    engine  = _engines.get(ticker)
    stream  = _streams.get(ticker)
    client  = getattr(stream, "_client", None) if stream else None

    ws_age = None
    if engine and engine.ws_spot_ts > 0:
        ws_age = round(time.time() - engine.ws_spot_ts, 1)

    stock_stream    = getattr(stream, "_stock_stream", None) if stream else None
    stock_count     = getattr(stock_stream, "_trade_count", 0)

    return {
        "ticker":              ticker,
        "spot":                round(engine.spot_price, 2) if engine else None,
        "spot_source":         engine.spot_source          if engine else None,
        "ws_spot_age_s":       ws_age,
        "options_ws": {
            "connected":       ticker in _streams,
            "subscribed_n":    len(client._subscribed)    if client else 0,
            "msg_counts":      client._msg_counts         if client else {},
            "seen_ev_types":   list(getattr(client, "_seen_evs", set())) if client else [],
        },
        "stocks_ws": {
            "running":         stock_stream is not None,
            "trade_count":     stock_count,
            "live":            stock_count > 0,
        },
        "poll_interval_s":     SPOT_POLL_SEC,
        "ws_stale_thresh_s":   WS_SPOT_STALE_SEC,
        "note":                "OK" if stock_count > 0 else "Waiting for first stock trade event",
    }


@app.get("/api/status")
async def status():
    return {
        "tickers":      list(_engines.keys()),
        "db_path":      DB_PATH,
        "min_dte":      MIN_DTE,
        "max_dte":      MAX_DTE,
        "mcp_servers":  [s["name"] for s in load_mcp_servers()],
        "ai_model":     os.getenv("GEX_AI_MODEL", "claude-sonnet-4-20250514"),
        "trade_cache":  get_cache_info(),
        "engines": {
            t: {"spot": round(e.spot_price, 2), "contracts": len(e.contract_data),
                "snapshots": len(e.history)}
            for t, e in _engines.items()
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GEX Dashboard Server")
    parser.add_argument("--ticker",  default=DEFAULT_TICKER)
    parser.add_argument("--port",    type=int, default=PORT)
    parser.add_argument("--host",    default="0.0.0.0")
    parser.add_argument("--min-dte", type=int, default=MIN_DTE)
    parser.add_argument("--max-dte", type=int, default=MAX_DTE)
    parser.add_argument("--no-ws",   action="store_true")
    args = parser.parse_args()

    os.environ.update({
        "GEX_TICKER":   args.ticker.upper(),
        "GEX_MIN_DTE":  str(args.min_dte),
        "GEX_MAX_DTE":  str(args.max_dte),
    })
    if args.no_ws:
        os.environ["GEX_USE_WEBSOCKET"] = "false"

    api_key = os.getenv("POLYGON_API_KEY", "")
    print(f"\n  GEX Dashboard  →  http://localhost:{args.port}")
    print(f"  Ticker : {args.ticker.upper()}  DTE: {args.min_dte}–{args.max_dte}")
    print(f"  DB     : {DB_PATH}")
    print(f"  API    : {'✓' if api_key else '✗ MISSING'}")
    print(f"  WS     : {'on' if not args.no_ws else 'off'}\n")

    uvicorn.run("gex_server:app", host=args.host, port=args.port,
                reload=False, log_level="info")

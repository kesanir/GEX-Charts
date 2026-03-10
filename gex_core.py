"""
gex_core.py — GEX computation engine
======================================
Pure computation and persistence.  No HTTP calls — all data retrieval in polygon_client.py.

Responsibilities:
  SnapshotStore        — SQLite persistence (snapshots, levels history, alerts)
  GEXEngine            — contract state, GEX-by-strike / bucketed / expiry
  compute_key_levels() — zero gamma, walls, regime, skew
  fast_bootstrap()     — populate engine from options chain snapshot
  PolygonOptionsStream — WebSocket live updates
"""

import json
import logging
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from polygon_client import (
    CONTRACT_SIZE,
    OptionsStreamClient,
    StockTradeStream,
    _is_market_open,
    fetch_contracts_reference,
    get_spot,
    get_spot_yfinance,
    iter_options_snapshot,
)

log = logging.getLogger("gex")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class SnapshotStore:
    def __init__(self, db_path: str):
        self.db_path = str(Path(db_path))
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as con:
            con.execute("""CREATE TABLE IF NOT EXISTS snapshots (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                underlying  TEXT NOT NULL,
                ts          TEXT NOT NULL,
                spot        REAL NOT NULL,
                gex_json    TEXT NOT NULL)""")
            con.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_u_ts ON snapshots(underlying, ts)")

            con.execute("""CREATE TABLE IF NOT EXISTS levels_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                underlying  TEXT NOT NULL,
                ts          TEXT NOT NULL,
                spot        REAL NOT NULL,
                zero_gamma  REAL,
                gamma_wall  REAL,
                call_wall   REAL,
                put_wall    REAL,
                hv_strike   REAL,
                net_gex_b   REAL,
                regime      TEXT,
                strength    REAL,
                gex_skew    REAL,
                dist_flip   REAL,
                net_0_1     REAL,
                net_2_7     REAL,
                net_8_45    REAL)""")
            con.execute("CREATE INDEX IF NOT EXISTS idx_lh_u_ts ON levels_history(underlying, ts)")

            con.execute("""CREATE TABLE IF NOT EXISTS alerts_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id    TEXT UNIQUE,
                underlying  TEXT NOT NULL,
                ts          TEXT NOT NULL,
                alert_type  TEXT NOT NULL,
                severity    TEXT NOT NULL,
                msg         TEXT NOT NULL,
                spot        REAL NOT NULL,
                extra_json  TEXT)""")
            con.execute("CREATE INDEX IF NOT EXISTS idx_ah_u_ts ON alerts_history(underlying, ts)")
            con.commit()

    # ── Levels ────────────────────────────────────────────────────────────────

    def insert_levels(self, underlying: str, ts_iso: str, spot: float,
                      levels: dict, bucketed_net: dict) -> None:
        try:
            with sqlite3.connect(self.db_path) as con:
                con.execute("""INSERT INTO levels_history
                    (underlying,ts,spot,zero_gamma,gamma_wall,call_wall,put_wall,
                     hv_strike,net_gex_b,regime,strength,gex_skew,dist_flip,
                     net_0_1,net_2_7,net_8_45)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                    underlying, ts_iso, float(spot),
                    levels.get("zero_gamma"), levels.get("gamma_wall"),
                    levels.get("call_wall"),  levels.get("put_wall"),
                    levels.get("hv_strike"),  levels.get("net_gex_b"),
                    levels.get("regime"),     levels.get("regime_strength"),
                    levels.get("gex_skew"),   levels.get("dist_to_flip_pct"),
                    bucketed_net.get("0_1", 0.0),
                    bucketed_net.get("2_7", 0.0),
                    bucketed_net.get("8_45", 0.0),
                ))
                con.commit()
        except Exception as e:
            log.warning("insert_levels: %s", e)

    def load_levels_history(self, underlying: str, days: int = 5) -> List[dict]:
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        try:
            with sqlite3.connect(self.db_path) as con:
                rows = con.execute("""
                    SELECT ts,spot,zero_gamma,gamma_wall,call_wall,put_wall,
                           net_gex_b,regime,strength,net_0_1,net_2_7,net_8_45
                    FROM levels_history WHERE underlying=? AND ts>=?
                    ORDER BY ts ASC""",
                    (underlying, cutoff)).fetchall()
            return [
                {"ts": r[0], "spot": r[1], "zero_gamma": r[2], "gamma_wall": r[3],
                 "call_wall": r[4], "put_wall": r[5], "net_gex_b": r[6],
                 "regime": r[7], "strength": r[8],
                 "net_0_1": r[9], "net_2_7": r[10], "net_8_45": r[11]}
                for r in rows
            ]
        except Exception as e:
            log.warning("load_levels_history: %s", e)
            return []

    # ── Alerts ────────────────────────────────────────────────────────────────

    def insert_alert(self, alert: dict) -> None:
        try:
            with sqlite3.connect(self.db_path) as con:
                extra = {k: v for k, v in alert.items()
                         if k not in ("id", "time", "ticker", "type", "severity", "msg", "spot")}
                con.execute("""INSERT OR IGNORE INTO alerts_history
                    (alert_id,underlying,ts,alert_type,severity,msg,spot,extra_json)
                    VALUES(?,?,?,?,?,?,?,?)""", (
                    alert.get("id", ""),   alert.get("ticker", ""),
                    alert.get("time", ""), alert.get("type", ""),
                    alert.get("severity", ""), alert.get("msg", ""),
                    float(alert.get("spot", 0)),
                    json.dumps(extra) if extra else None,
                ))
                con.commit()
        except Exception as e:
            log.warning("insert_alert: %s", e)

    def load_alerts_history(self, underlying: Optional[str] = None,
                            days: int = 7, limit: int = 200) -> List[dict]:
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        try:
            with sqlite3.connect(self.db_path) as con:
                if underlying:
                    rows = con.execute("""
                        SELECT alert_id,underlying,ts,alert_type,severity,msg,spot,extra_json
                        FROM alerts_history WHERE underlying=? AND ts>=?
                        ORDER BY ts DESC LIMIT ?""",
                        (underlying, cutoff, limit)).fetchall()
                else:
                    rows = con.execute("""
                        SELECT alert_id,underlying,ts,alert_type,severity,msg,spot,extra_json
                        FROM alerts_history WHERE ts>=?
                        ORDER BY ts DESC LIMIT ?""",
                        (cutoff, limit)).fetchall()
            return [
                {"id": r[0], "ticker": r[1], "time": r[2], "type": r[3],
                 "severity": r[4], "msg": r[5], "spot": r[6],
                 **(json.loads(r[7]) if r[7] else {})}
                for r in rows
            ]
        except Exception as e:
            log.warning("load_alerts_history: %s", e)
            return []

    # ── Snapshots ─────────────────────────────────────────────────────────────

    def insert_snapshot(self, underlying: str, ts_iso: str, spot: float, gex: dict):
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT INTO snapshots(underlying,ts,spot,gex_json) VALUES(?,?,?,?)",
                (underlying, ts_iso, float(spot),
                 json.dumps({str(k): v for k, v in gex.items()}, separators=(",", ":"))))
            con.commit()

    def load_recent(self, underlying: str, limit: int = 780) -> List[dict]:
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute(
                "SELECT ts,spot,gex_json FROM snapshots "
                "WHERE underlying=? ORDER BY ts DESC LIMIT ?",
                (underlying, int(limit))).fetchall()
        out = []
        for ts, spot, gex_json in reversed(rows):
            raw = json.loads(gex_json)
            gex = {float(k): float(v) for k, v in raw.items()}
            out.append({"timestamp": pd.to_datetime(ts), "spot": float(spot), "gex": gex})
        return out

    def prune_old(self, underlying: str, keep_days: int = 5):
        cutoff = (datetime.utcnow() - timedelta(days=keep_days)).isoformat()
        with sqlite3.connect(self.db_path) as con:
            con.execute("DELETE FROM snapshots WHERE underlying=? AND ts<?", (underlying, cutoff))
            con.commit()


# ---------------------------------------------------------------------------
# GEX Engine
# ---------------------------------------------------------------------------

class GEXEngine:
    def __init__(self, store: Optional[SnapshotStore] = None, underlying: str = ""):
        self.contract_data: Dict[str, Dict[str, Any]] = {}
        self.spot_price:    float                     = 0.0
        self.history:       List[dict]                = []
        self.store          = store
        self.underlying     = underlying
        self.ws_spot_ts:    float                     = 0.0   # epoch of last WS trade update
        self.spot_source:   str                       = ""    # "ws" | "rest" | "seed" | ""

    def update_spot(self, price: float, source: str = "") -> None:
        """
        Update spot price.  source="ws" marks this as a live WS trade —
        REST pollers will not overwrite a WS price until it is stale.
        """
        if price and price > 0:
            self.spot_price = float(price)
            if source:
                self.spot_source = source
            if source == "ws":
                self.ws_spot_ts = time.time()

    def update_contract(self, ticker: str, strike: float, ctype: str,
                        gamma: float, oi: int, expiry=None):
        self.contract_data[ticker] = {
            "strike": float(strike),
            "type":   str(ctype).lower(),
            "gamma":  float(gamma or 0.0),
            "oi":     int(oi or 0),
            "expiry": str(expiry or ""),
        }

    def compute_gex_by_strike(self) -> Dict[float, float]:
        if self.spot_price <= 0:
            return {}
        S       = self.spot_price
        gex_map = defaultdict(float)
        for c in self.contract_data.values():
            raw = c["gamma"] * CONTRACT_SIZE * c["oi"] * (S ** 2) * 0.01
            gex_map[c["strike"]] += raw if c["type"] == "call" else -raw
        return dict(gex_map)

    def compute_gex_by_expiry_strike(self) -> Dict[str, Dict[float, float]]:
        """Returns {expiry_iso: {strike: net_gex}} — strike × expiry matrix."""
        if self.spot_price <= 0:
            return {}
        S   = self.spot_price
        out: Dict[str, Any] = defaultdict(lambda: defaultdict(float))
        for c in self.contract_data.values():
            exp = c.get("expiry", "")
            if not exp:
                continue
            raw = c["gamma"] * CONTRACT_SIZE * c["oi"] * (S ** 2) * 0.01
            out[exp][c["strike"]] += raw if c["type"] == "call" else -raw
        return {k: dict(v) for k, v in sorted(out.items())}

    @staticmethod
    def _dte_bucket(expiry_str: str) -> str:
        """
        Classify expiry into DTE bucket.
          '0_1'  — 0-1 DTE  : intraday/overnight gamma, pin risk
          '2_7'  — 2-7 DTE  : weekly walls, dominant near-term positioning
          '8_45' — 8-45 DTE : structural / monthly walls
          'far'  — >45 DTE  : LEAPS, low gamma per dollar
        """
        try:
            exp = datetime.strptime(str(expiry_str), "%Y-%m-%d").date()
            dte = (exp - date.today()).days
        except Exception:
            return "unknown"
        if dte <= 1:  return "0_1"
        if dte <= 7:  return "2_7"
        if dte <= 45: return "8_45"
        return "far"

    def compute_gex_bucketed(self) -> Dict[str, Dict[float, float]]:
        """
        GEX by strike broken into DTE buckets.
        Returns {bucket: {strike: net_gex_dollars}}.

        Why this matters:
          0_1  — expires today; don't rely on it tomorrow
          2_7  — weekly walls; swing trading horizon
          8_45 — structural; holds for weeks; flip = real regime change
        """
        if self.spot_price <= 0:
            return {}
        S   = self.spot_price
        out = {"0_1": defaultdict(float), "2_7": defaultdict(float),
               "8_45": defaultdict(float), "far": defaultdict(float)}
        for c in self.contract_data.values():
            bucket = self._dte_bucket(c.get("expiry", ""))
            if bucket not in out:
                continue
            raw = c["gamma"] * CONTRACT_SIZE * c["oi"] * (S ** 2) * 0.01
            out[bucket][c["strike"]] += raw if c["type"] == "call" else -raw
        return {k: dict(v) for k, v in out.items() if v}

    def snapshot(self):
        gex = self.compute_gex_by_strike()
        if not gex:
            return
        ts = datetime.utcnow()
        self.history.append({"timestamp": ts, "gex": gex, "spot": self.spot_price})
        if len(self.history) > 780:
            self.history = self.history[-780:]
        if self.store and self.underlying:
            try:
                self.store.insert_snapshot(
                    self.underlying, ts.isoformat(), self.spot_price, gex)
            except Exception as e:
                log.warning("DB snapshot: %s", e)

    def restore_from_store(self, limit: int = 780) -> int:
        if not self.store or not self.underlying:
            return 0
        self.history = self.store.load_recent(self.underlying, limit=limit)
        if self.history:
            self.spot_price = float(self.history[-1]["spot"])
        return len(self.history)

    def get_matrix(self, hours_back: Optional[float] = None
                   ) -> Tuple[list, list, np.ndarray, list]:
        if not self.history:
            return [], [], np.array([]), []
        df = pd.DataFrame(self.history)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if hours_back is not None:
            cutoff = df["timestamp"].max() - timedelta(hours=hours_back)
            df = df[df["timestamp"] >= cutoff]
        if df.empty:
            return [], [], np.array([]), []
        all_strikes = sorted({s for g in df["gex"] for s in g})
        times, spots = df["timestamp"].tolist(), df["spot"].tolist()
        matrix = np.zeros((len(all_strikes), len(times)))
        sidx   = {s: i for i, s in enumerate(all_strikes)}
        for t_idx, row in enumerate(df["gex"]):
            for strike, val in row.items():
                matrix[sidx[strike], t_idx] = val
        return times, all_strikes, matrix / 1e9, spots


# ---------------------------------------------------------------------------
# Quant key levels
# ---------------------------------------------------------------------------

def compute_key_levels(gex_map: Dict[float, float], spot: float) -> Dict[str, Any]:
    """
    Full quant-grade level set from a GEX-by-strike dict.

    zero_gamma       — interpolated strike where cumulative GEX = 0 (regime divider)
    gamma_wall       — strike with highest |GEX| (price magnet)
    call_wall        — strike with highest positive GEX (overhead resistance)
    put_wall         — strike with most negative GEX (downside support/accelerator)
    hv_strike        — highest |GEX| within ±5% of spot (pin target near expiry)
    net_gex_b        — total net GEX $B
    regime           — 'positive' (vol suppression) | 'negative' (vol amplification)
    regime_strength  — net / Σ|GEX|, range −1…+1
    gex_skew         — call_gex / |put_gex|  (>1 = call-heavy)
    dist_to_flip_pct — % distance from spot to zero_gamma
    """
    if not gex_map or spot <= 0:
        return {}

    strikes = sorted(gex_map.keys())
    vals    = [gex_map[s] for s in strikes]
    cum     = list(np.cumsum(vals))
    abs_v   = [abs(v) for v in vals]

    zero_gamma = None
    for i in range(1, len(cum)):
        if cum[i - 1] * cum[i] <= 0:
            s0, s1, c0, c1 = strikes[i-1], strikes[i], cum[i-1], cum[i]
            zero_gamma = (round(s0 + (s1 - s0) * (-c0 / (c1 - c0)), 2)
                          if c1 != c0 else round((s0 + s1) / 2, 2))
            break

    gamma_wall = strikes[int(np.argmax(abs_v))]
    pos_pairs  = [(s, v) for s, v in zip(strikes, vals) if v > 0]
    neg_pairs  = [(s, v) for s, v in zip(strikes, vals) if v < 0]
    call_wall  = max(pos_pairs, key=lambda x: x[1])[0] if pos_pairs else None
    put_wall   = min(neg_pairs, key=lambda x: x[1])[0] if neg_pairs else None

    near      = [(s, abs(v)) for s, v in zip(strikes, vals) if abs(s - spot) / spot <= 0.05]
    hv_strike = max(near, key=lambda x: x[1])[0] if near else gamma_wall

    net_gex   = sum(vals)
    total_abs = sum(abs_v) or 1.0
    regime    = "positive" if net_gex >= 0 else "negative"
    strength  = round(net_gex / total_abs, 4)
    call_gex  = sum(v for v in vals if v > 0)
    put_gex   = abs(sum(v for v in vals if v < 0)) or 1.0
    skew      = round(call_gex / put_gex, 3)
    dist_flip = round((spot - zero_gamma) / spot * 100, 3) if zero_gamma else None

    return {
        "zero_gamma":       zero_gamma,
        "gamma_wall":       round(gamma_wall, 2),
        "call_wall":        round(call_wall, 2)  if call_wall  else None,
        "put_wall":         round(put_wall, 2)   if put_wall   else None,
        "hv_strike":        round(hv_strike, 2),
        "net_gex_b":        round(net_gex / 1e9, 4),
        "regime":           regime,
        "regime_strength":  strength,
        "gex_skew":         skew,
        "dist_to_flip_pct": dist_flip,
    }


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def fast_bootstrap(engine: GEXEngine, underlying_ticker: str,
                   min_dte: int = 0, max_dte: int = 45) -> None:
    """
    Populate engine from /v3/snapshot/options/{ticker} in a single paginated pass.

    Spot seeding strategy:
      1. get_spot() before iteration   — aggs_1min (open) or prev_close (closed)
      2. Inside loop: update from underlying_asset.price only for REAL-TIME/DELAYED
         (won't appear on Options Starter but ready if plan is upgraded)
      3. underlying_asset is read before the DTE filter so all 7000+ skipped
         far-dated pages still contribute their price data
    """
    log.info("[%s] Fast bootstrap DTE %d-%d ...", underlying_ticker, min_dte, max_dte)
    today  = date.today()
    exp_lo = today + timedelta(days=min_dte)
    exp_hi = today + timedelta(days=max_dte)

    # Seed spot before iterating chain
    seed = get_spot(underlying_ticker)
    if seed > 0:
        engine.update_spot(seed)
    got_live_spot = seed > 0

    updated = skipped = 0

    for item in iter_options_snapshot(underlying_ticker):
        det     = item.get("details") or {}
        oticker = det.get("ticker")
        if not oticker:
            continue

        # Extract spot from EVERY item before DTE filter
        ua    = item.get("underlying_asset") or {}
        ua_p  = float(ua.get("price") or 0)
        ua_tf = str(ua.get("timeframe") or "").upper()
        if ua_p > 0:
            if ua_tf in ("REAL-TIME", "DELAYED"):
                engine.update_spot(ua_p)
                got_live_spot = True
            elif ua_tf == "PREVIOUS_CLOSE" and not got_live_spot and not _is_market_open():
                engine.update_spot(ua_p)

        try:
            exp_date = datetime.strptime(det.get("expiration_date", ""), "%Y-%m-%d").date()
        except Exception:
            continue
        if not (exp_lo <= exp_date <= exp_hi):
            skipped += 1
            continue

        ctype  = str(det.get("contract_type", "")).lower()
        strike = float(det.get("strike_price") or 0)
        if ctype not in ("call", "put") or strike <= 0:
            continue

        greeks = item.get("greeks") or {}
        oi     = int(item.get("open_interest") or 0)
        engine.update_contract(
            oticker, strike, ctype,
            float(greeks.get("gamma") or 0), oi,
            expiry=exp_date,
        )
        updated += 1

    if updated == 0:
        log.warning("[%s] Fast bootstrap got 0 contracts — falling back", underlying_ticker)
        _bootstrap_fallback(engine, underlying_ticker, min_dte, max_dte)
        return

    engine.snapshot()
    net = sum(engine.compute_gex_by_strike().values())
    log.info("[%s] Bootstrap done — %d contracts, %d skipped DTE, spot=%.2f, GEX=%.3fB",
             underlying_ticker, updated, skipped, engine.spot_price, net / 1e9)


def _bootstrap_fallback(engine: GEXEngine, underlying_ticker: str,
                        min_dte: int = 0, max_dte: int = 45) -> None:
    """
    Legacy fallback: build contract_meta from reference endpoint, re-paginate snapshot.
    Uses the same iter_options_snapshot generator — no code duplication.
    """
    log.info("[%s] Fallback bootstrap via reference contracts", underlying_ticker)
    raw = fetch_contracts_reference(underlying_ticker, min_dte, max_dte)
    if not raw:
        log.error("[%s] Reference contracts also empty — cannot bootstrap", underlying_ticker)
        return

    contract_meta: Dict[str, dict] = {
        str(c.get("ticker", "")): {
            "strike":     float(c.get("strike_price") or 0),
            "type":       str(c.get("contract_type") or "").lower(),
            "oi":         int(c.get("open_interest") or 0),
            "expiration": str(c.get("expiration_date") or ""),
        }
        for c in raw if c.get("ticker")
    }

    if engine.spot_price <= 0:
        seed = get_spot(underlying_ticker)
        if seed > 0:
            engine.update_spot(seed)

    updated = 0
    for item in iter_options_snapshot(underlying_ticker):
        oticker = (item.get("details") or {}).get("ticker")
        if not oticker:
            continue
        meta = contract_meta.get(oticker)
        if not meta:
            continue
        ua    = item.get("underlying_asset") or {}
        ua_p  = float(ua.get("price") or 0)
        ua_tf = str(ua.get("timeframe") or "").upper()
        if ua_p > 0 and ua_tf in ("REAL-TIME", "DELAYED"):
            engine.update_spot(ua_p)
        greeks = item.get("greeks") or {}
        oi     = int(item.get("open_interest") or meta["oi"])
        engine.update_contract(
            oticker, meta["strike"], meta["type"],
            float(greeks.get("gamma") or 0), oi,
            expiry=meta["expiration"],
        )
        updated += 1

    engine.snapshot()
    log.info("[%s] Fallback done — %d contracts, spot=%.2f",
             underlying_ticker, updated, engine.spot_price)


# ---------------------------------------------------------------------------
# WebSocket live stream  — business logic only, no Polygon protocol code
# ---------------------------------------------------------------------------

class PolygonOptionsStream:
    """
    Manages which options to subscribe to and what to do with incoming data.

    All Polygon-specific concerns (WS URL, auth, reconnect, event parsing)
    are handled by OptionsStreamClient in polygon_client.py.

    This class is pure business logic:
      - Build contract_meta from bootstrapped engine state
      - Select the subscription universe (near-spot, high-OI, DTE-filtered)
      - Update GEXEngine on option quote and equity trade events
      - Trigger periodic snapshots
    """

    def __init__(
        self,
        engine:                  GEXEngine,
        contracts_df:            "pd.DataFrame",
        ticker:                  str,
        snapshot_int:            int             = 60,
        universe_refresh_sec:    int             = 120,
        near_spot_pct:           float           = 0.10,
        max_subscribe_near:      int             = 800,
        max_subscribe_oi_anchors: int            = 200,
        allowed_dte:             Optional[Set[int]] = None,
    ):
        self.engine                   = engine
        self.ticker                   = ticker.upper()
        self.snapshot_int             = int(snapshot_int)
        self.universe_refresh_sec     = int(universe_refresh_sec)
        self.near_spot_pct            = float(near_spot_pct)
        self.max_subscribe_near       = int(max_subscribe_near)
        self.max_subscribe_oi_anchors = int(max_subscribe_oi_anchors)
        self.allowed_dte              = set(allowed_dte) if allowed_dte else {0, 1, 2, 3, 4, 5, 7}
        self._last_snapshot           = time.time()
        self._last_universe_refresh   = 0.0

        # Build contract metadata from bootstrapped engine state
        self.contract_meta: Dict[str, dict] = {}
        for _, row in contracts_df.iterrows():
            t = str(row.get("ticker", ""))
            if not t:
                continue
            self.contract_meta[t] = {
                "strike":     float(row.get("strike", 0)),
                "type":       str(row.get("type", "")).lower(),
                "oi":         int(row.get("oi", 0) or 0),
                "expiration": row.get("expiration"),
            }
            self.engine.update_contract(
                t, self.contract_meta[t]["strike"],
                self.contract_meta[t]["type"],
                0.0, self.contract_meta[t]["oi"],
            )

        # Options WS — handles options quote events
        self._client = OptionsStreamClient(
            ticker       = ticker,
            on_option    = self._handle_option,
            on_trade     = self._handle_trade,   # fallback if plan delivers T events
            on_connected = self._handle_connected,
        )

        # Stocks WS — dedicated T.{ticker} feed for real-time spot price.
        # On Options Starter plan auth_failed fires immediately and permanently
        # stops the reconnect loop.  on_auth_failed clears ws_spot_ts so the
        # _spot_poll_loop WS guard does not permanently block yfinance polling.
        def _on_stocks_auth_failed():
            self.engine.ws_spot_ts = 0.0
            log.info("[%s] Stocks WS auth failed — cleared ws_spot_ts, yfinance poll now active",
                     ticker)

        self._stock_stream = StockTradeStream(
            ticker          = ticker,
            on_trade        = lambda p: self.engine.update_spot(p, source="ws"),
            on_auth_failed  = _on_stocks_auth_failed,
        )

    # ── Business-logic event handlers ─────────────────────────────────────────

    def _handle_option(self, sym: str, gamma: float, oi: int) -> None:
        """Called by OptionsStreamClient on every O event."""
        m = self.contract_meta.get(sym)
        if m:
            self.engine.update_contract(
                sym, m["strike"], m["type"],
                gamma, oi if oi > 0 else m["oi"],
            )
        self._maybe_snapshot()

    def _handle_trade(self, price: float) -> None:
        """Called by OptionsStreamClient on T events for the equity ticker."""
        self.engine.update_spot(price, source="ws")
        self._refresh_universe_if_needed()
        self._maybe_snapshot()

    def _handle_connected(self) -> None:
        """Called by OptionsStreamClient after auth_success + equity sub sent."""
        self._push_universe()

    # ── Universe selection ─────────────────────────────────────────────────────

    def _dte(self, exp) -> int:
        try:
            return (pd.to_datetime(exp).date() - date.today()).days
        except Exception:
            return 9999

    def _select_universe(self) -> Set[str]:
        """
        Choose which option symbols to subscribe to.

        Priority:
          1. Contracts expiring within allowed_dte (default 0-7d)
          2. Among those: near-the-money by spot ± near_spot_pct
          3. Anchor set: top-OI across all DTE to maintain structural walls
        """
        spot     = self.engine.spot_price
        filtered = [(t, m) for t, m in self.contract_meta.items()
                    if self._dte(m["expiration"]) in self.allowed_dte]
        if not filtered:
            filtered = list(self.contract_meta.items())

        if not spot or spot <= 0:
            return set(
                t for t, _ in
                sorted(filtered, key=lambda kv: kv[1]["oi"], reverse=True)
                [:max(self.max_subscribe_near, 400)]
            )

        near     = [(t, m) for t, m in filtered
                    if abs(m["strike"] - spot) / spot <= self.near_spot_pct]
        top_near = [t for t, _ in sorted(near, key=lambda kv: kv[1]["oi"],
                                          reverse=True)[:self.max_subscribe_near]]
        top_oi   = [t for t, _ in sorted(filtered, key=lambda kv: kv[1]["oi"],
                                          reverse=True)[:self.max_subscribe_oi_anchors]]
        return set(top_near + top_oi)

    def _push_universe(self) -> None:
        target = self._select_universe()
        self._client.set_subscriptions(target)
        self._last_universe_refresh = time.time()
        log.info("[%s] WS subscribed %d options + equity", self.ticker, len(target))

    def _refresh_universe_if_needed(self) -> None:
        if time.time() - self._last_universe_refresh >= self.universe_refresh_sec:
            self._push_universe()

    def _maybe_snapshot(self) -> None:
        if time.time() - self._last_snapshot >= self.snapshot_int:
            self.engine.snapshot()
            self._last_snapshot = time.time()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Start both WS connections in parallel threads, then block.

        Thread 1: OptionsStreamClient  — options quotes (O events)
        Thread 2: StockTradeStream     — equity trades (T events) for spot price
        """
        import threading
        stock_thread = threading.Thread(
            target=self._stock_stream.run,
            name=f"ws-stocks-{self.ticker}",
            daemon=True,
        )
        stock_thread.start()
        log.info("[%s] Stock trade stream thread started", self.ticker)
        # Options stream blocks this thread
        self._client.run()

    def close(self) -> None:
        """Stop both WebSocket streams."""
        self._client.close()
        self._stock_stream.close()


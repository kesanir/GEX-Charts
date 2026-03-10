"""
polygon_client.py — Polygon.io API layer
=========================================
Single source of truth for every HTTP call to Polygon.

Provides:
  get_spot(ticker)                  → float
  get_ohlc(ticker, res, hours)      → list[dict]
  iter_options_snapshot(ticker)     → Iterator[dict]   (generator, paginates)
  fetch_contracts_reference(ticker) → list[dict]       (legacy fallback)
  diagnose_spot_endpoints(ticker)   → None             (startup debug)

Confirmed working on Polygon Options Starter:
  ✓  /v2/aggs  (1-min bars, prev close)
  ✓  /v3/snapshot/options/{ticker}
  ✗  /v2/last/trade           (HTTP 403)
  ✗  /v2/snapshot (stocks)    (HTTP 403)
  ✗  options chain ua.price   (empty — no price field on this plan)
"""

import json
import logging
import os
import time
from datetime import datetime, date, timedelta
from typing import Iterator, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

load_dotenv()

log           = logging.getLogger("gex")
API_KEY       = os.getenv("POLYGON_API_KEY", "")
BASE_URL      = "https://api.polygon.io"
CONTRACT_SIZE = 100

if not API_KEY:
    raise ValueError("POLYGON_API_KEY not set in .env")


# ---------------------------------------------------------------------------
# Session — reuses TCP/TLS connections, auto-retries transient errors
# ---------------------------------------------------------------------------

import threading as _threading

_local = _threading.local()

def _make_session() -> requests.Session:
    """Create a new session with retry logic."""
    s     = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=8))
    return s

def _session() -> requests.Session:
    """
    Thread-local session — each thread (executor worker) gets its own
    session so concurrent bootstraps never share a connection pool.
    requests.Session is NOT thread-safe for concurrent use from multiple threads.
    """
    if not hasattr(_local, "session"):
        _local.session = _make_session()
    return _local.session


def _get(path_or_url: str, params: dict = None, timeout: int = 15) -> Optional[dict]:
    """
    Single authenticated GET.  Returns parsed JSON or None on any error.
    Prepends BASE_URL when path_or_url starts with '/'.
    """
    url = (BASE_URL + path_or_url) if path_or_url.startswith("/") else path_or_url
    p   = dict(params or {})
    p.setdefault("apiKey", API_KEY)
    try:
        r = _session().get(url, params=p, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        log.debug("[polygon] HTTP %d → %s", r.status_code, url.split("?")[0])
        return None
    except Exception as e:
        log.debug("[polygon] error: %s", e)
        return None


# ---------------------------------------------------------------------------
# Market hours
# ---------------------------------------------------------------------------

def _et_now():
    """
    Current datetime in US/Eastern timezone.

    Uses UTC offset arithmetic — no zoneinfo/pytz dependency.
    EST = UTC-5, EDT = UTC-4.  DST runs second Sunday of March through
    first Sunday of November.  This matches exchange calendar exactly.
    """
    from datetime import timezone
    utc_now = datetime.now(timezone.utc)
    # Determine DST: second Sun of March → first Sun of November
    yr = utc_now.year
    # Second Sunday of March
    mar1  = date(yr, 3, 1)
    dst_start = mar1 + timedelta(days=(6 - mar1.weekday()) % 7 + 7)  # +2h at 02:00
    # First Sunday of November
    nov1  = date(yr, 11, 1)
    dst_end   = nov1 + timedelta(days=(6 - nov1.weekday()) % 7)      # +1h at 02:00
    d = utc_now.date()
    is_dst = dst_start <= d < dst_end
    offset = timedelta(hours=-4 if is_dst else -5)
    return utc_now + offset


def _is_market_open() -> bool:
    """True during US equity market session Mon–Fri 09:30–16:00 ET."""
    et = _et_now()
    if et.weekday() >= 5:
        return False
    t = et.hour * 60 + et.minute
    return 570 <= t <= 960          # 09:30 – 16:00


def _is_pre_market() -> bool:
    """True during pre-market Mon–Fri 04:00–09:30 ET."""
    et = _et_now()
    if et.weekday() >= 5:
        return False
    t = et.hour * 60 + et.minute
    return 240 <= t < 570           # 04:00 – 09:30


def _last_trading_day() -> date:
    """Most recent weekday (today if Mon–Fri, Friday if weekend)."""
    d = _et_now().date()
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


# ---------------------------------------------------------------------------
# Spot price
# ---------------------------------------------------------------------------

def _agg_close(ticker: str, lookback_ms: int, label: str) -> float:
    """Close price of the most recent 1-min agg bar within the lookback window."""
    from datetime import timezone
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    ago_ms = now_ms - lookback_ms
    d = _get(
        f"/v2/aggs/ticker/{ticker}/range/1/minute/{ago_ms}/{now_ms}",
        {"sort": "desc", "limit": 1, "adjusted": "true"},
    )
    if not d:
        return 0.0
    try:
        return float(((d.get("results") or [{}])[0]).get("c") or 0)
    except Exception:
        return 0.0


def _prev_close(ticker: str) -> float:
    """Official session close from /v2/aggs prev. Confirmed working on Options Starter."""
    d = _get(f"/v2/aggs/ticker/{ticker}/prev", {"adjusted": "true"})
    if not d:
        return 0.0
    try:
        return float(((d.get("results") or [{}])[0]).get("c") or 0)
    except Exception:
        return 0.0


def poll_spot_agg(ticker: str) -> float:
    """
    Fetch the most recent 1-min bar close from Polygon /v2/aggs.

    Uses a 3-minute lookback so we always catch the last completed bar
    (~1 min lag) without pulling unnecessary history.

    Returns 0.0 if the bar is unavailable (plan restriction, market closed,
    or Polygon hasn't finalized the bar yet).

    Called from the dedicated agg-poll loop every 30s during market hours.
    """
    p = _agg_close(ticker.upper(), lookback_ms=3 * 60 * 1000, label=None)
    if p > 0:
        log.info("[spot] %s = %.2f via aggs_1min (poll)", ticker, p)
    return p


def get_spot(ticker: str) -> float:
    """
    Best available REST spot price.

    Hierarchy (market open):
      1. 1-min agg bar  — most current (~1 min lag).  May return 0 on Options Starter
         during market hours if the plan only delivers EOD bars.
      2. prev_close     — always available; seeds the engine until WS overrides it.
         NOTE: _spot_poll_loop will NOT use this to overwrite a live WS price.

    Hierarchy (market closed / pre-market):
      1. Today's daily bar (most recently completed session)
      2. prev_close
      3. Most recent 1-min bar (fallback for pre-market)
    """
    ticker = ticker.upper()

    if _is_market_open():
        # Try increasingly wide windows — 2h first, then full session
        for lookback_min in (120, 390):
            p = _agg_close(ticker, lookback_ms=lookback_min * 60 * 1000, label=None)
            if p > 0:
                log.info("[spot] %s = %.2f via aggs_1min (%dm window)", ticker, p, lookback_min)
                return p
        # Plan restriction: 1-min bars not available intraday on Options Starter.
        # Try yfinance (free, ~15min delayed) before falling back to prev_close.
        p = get_spot_yfinance(ticker)
        if p > 0:
            return p
        p = _prev_close(ticker)
        if p > 0:
            log.info("[spot] %s = %.2f via prev_close (seed only)", ticker, p)
        return p

    else:
        # After close / weekend / holiday.
        # Priority:
        #   1. Today's daily bar  — official session close, finalized ~30min after close
        #   2. prev_close         — only if today's bar not yet available (e.g. right at 4:01pm)
        #   3. aggs_1min fallback — pre-market or holiday (no daily bar yet)
        try:
            import zoneinfo
            today_str = datetime.now(zoneinfo.ZoneInfo("America/New_York")).date().isoformat()
        except ImportError:
            today_str = _last_trading_day().isoformat()

        # Use sort=desc + 5-day window so we always get the most recently
        # completed daily bar, regardless of exactly when Polygon finalizes it.
        # Single-day queries ({today}/{today}) can return empty right after close.
        try:
            from datetime import timezone
            now_ms  = int(datetime.now(timezone.utc).timestamp() * 1000)
            ago_ms  = now_ms - 5 * 24 * 3600 * 1000
        except Exception:
            ago_ms, now_ms = 0, int(__import__('time').time() * 1000)

        d = _get(
            f"/v2/aggs/ticker/{ticker}/range/1/day/{ago_ms}/{now_ms}",
            {"adjusted": "true", "sort": "desc", "limit": 1},
        )
        if d:
            try:
                bar = (d.get("results") or [{}])[0]
                p   = float(bar.get("c") or 0)
                # Only use this bar if it's from today (avoid returning yesterday
                # when today's session hasn't started yet, e.g. pre-market)
                bar_ts_s = (bar.get("t") or 0) / 1000
                import zoneinfo as _zi
                bar_date = datetime.fromtimestamp(
                    bar_ts_s, tz=_zi.ZoneInfo("America/New_York")).date()
                expected = _last_trading_day()
                if p > 0 and bar_date == expected:
                    log.info("[spot] %s = %.2f via aggs_daily (bar=%s)", ticker, p, bar_date)
                    return p
            except Exception:
                pass

        # Daily bar not yet settled (e.g. called within seconds of 4pm) → prev_close
        p = _prev_close(ticker)
        if p > 0:
            return p

        # Pre-market and no prev_close — use most recent 1-min bar
        return _agg_close(ticker, lookback_ms=7 * 24 * 3600 * 1000, label="aggs_1min_recent")


# ---------------------------------------------------------------------------
# OHLC candles
# ---------------------------------------------------------------------------

def get_ohlc(ticker: str, resolution_min: int = 5, hours: float = 6) -> List[dict]:
    """
    OHLC bars for the candlestick chart.

    resolution_min : bar size in minutes — snapped to [1, 2, 5, 15, 30, 60]
    hours          : lookback in hours

    When market is closed the window is anchored to the most recent session open
    (09:30 ET) so "6h" still returns the full afternoon regardless of call time.
    Returns list of {t, o, h, l, c, v} sorted ascending.
    """
    ticker = ticker.upper()
    valid  = [1, 2, 5, 15, 30, 60]
    resolution_min = min(valid, key=lambda x: abs(x - int(resolution_min)))

    from datetime import timezone
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    if _is_market_open():
        ago_ms = int(now_ms - hours * 3600 * 1000)
    else:
        # Anchor to session open of most recent trading day
        d = _last_trading_day()
        try:
            import zoneinfo
            session_open    = datetime(d.year, d.month, d.day, 9, 30,
                                       tzinfo=zoneinfo.ZoneInfo("America/New_York"))
        except ImportError:
            from datetime import timezone as tz
            session_open    = datetime(d.year, d.month, d.day, 14, 30, tzinfo=tz.utc)
        session_open_ms = int(session_open.timestamp() * 1000)
        user_ago_ms     = int(now_ms - hours * 3600 * 1000)
        ago_ms          = max(user_ago_ms, session_open_ms)

    d = _get(
        f"/v2/aggs/ticker/{ticker}/range/{resolution_min}/minute/{ago_ms}/{now_ms}",
        {"sort": "asc", "limit": 500, "adjusted": "true"},
    )
    if not d:
        return []

    bars = [
        {"t": b["t"], "o": b["o"], "h": b["h"],
         "l": b["l"], "c": b["c"], "v": round(b.get("v", 0), 0)}
        for b in (d.get("results") or [])
        if all(k in b for k in ("t", "o", "h", "l", "c"))
    ]
    log.debug("[ohlc] %s %dmin × %d bars", ticker, resolution_min, len(bars))
    return bars


# ---------------------------------------------------------------------------
# Options chain pagination — generator
# ---------------------------------------------------------------------------

def iter_options_snapshot(ticker: str, limit: int = 250) -> Iterator[dict]:
    """
    Paginate /v3/snapshot/options/{ticker}, yield one contract dict at a time.

    Uses the shared session so all pages share one TCP/TLS connection —
    SPY has ~52 pages; connection reuse saves ~200ms × 52 ≈ 10 seconds vs bare requests.
    """
    url    = f"{BASE_URL}/v3/snapshot/options/{ticker.upper()}"
    params = {"limit": limit}
    page   = 0

    while url:
        d = _get(url, params)
        if d is None:
            log.error("[options_snapshot] page %d failed for %s", page + 1, ticker)
            return
        page += 1
        yield from d.get("results", [])
        url    = d.get("next_url")
        params = {}          # next_url already contains all query params
        if url:
            time.sleep(0.2)  # gentle pacing; session reuse already cuts latency


# ---------------------------------------------------------------------------
# Reference contracts (legacy fallback)
# ---------------------------------------------------------------------------

def fetch_contracts_reference(ticker: str, min_dte: int = 0, max_dte: int = 45) -> list:
    """
    Fetch option contract metadata from /v3/reference/options/contracts.
    Only called as a fallback when iter_options_snapshot yields 0 contracts.
    """
    today    = date.today()
    exp_from = (today + timedelta(days=min_dte)).isoformat()
    exp_to   = (today + timedelta(days=max_dte)).isoformat()
    url      = f"{BASE_URL}/v3/reference/options/contracts"
    params   = {
        "underlying_ticker": ticker,
        "expiration_date.gte": exp_from,
        "expiration_date.lte": exp_to,
        "limit": 1000,
    }
    out = []
    while url:
        d = _get(url, params)
        if not d:
            break
        out.extend(d.get("results", []))
        url    = d.get("next_url")
        params = {}
    log.info("[contracts_ref] %s: %d contracts", ticker, len(out))
    return out


# ---------------------------------------------------------------------------
# WebSocket stream client  — ALL Polygon WS concerns live here
# ---------------------------------------------------------------------------

from typing import Callable, Set as _Set

# Silence the websocket-client library's own status messages
# ("Websocket connected", "--- request header ---", etc.) which clutter logs.
import logging as _logging
_logging.getLogger("websocket").setLevel(_logging.WARNING)


class OptionsStreamClient:
    """
    Manages the raw WebSocket connection to the Polygon options stream.

    Owns everything that is Polygon-specific about the connection:
      - WS URL and authentication protocol
      - connect / auth / subscribe / unsubscribe command formatting
      - parse Polygon event envelope  (ev: "status" / "O" / "T")
      - auto-reconnect with exponential backoff (5s → 10s → 20s … cap 120s)
      - ping keepalive

    Dispatches three typed callbacks — callers supply business logic:
      on_option(symbol, gamma, oi)  called on every "O" event
      on_trade(price)               called on "T" events for the equity ticker
      on_connected()                called after auth_success + equity sub sent

    All subscription management (which symbols, when to resub) is the
    caller's responsibility via set_subscriptions(symbols).
    """

    WS_URL = "wss://socket.polygon.io/options"

    def __init__(
        self,
        ticker:       str,
        on_option:    Callable[[str, float, int], None],
        on_trade:     Callable[[float], None],
        on_connected: Callable[[], None],
    ):
        self.ticker        = ticker.upper()
        self._on_option    = on_option
        self._on_trade     = on_trade
        self._on_connected = on_connected
        self._ws            = None
        self._subscribed:   _Set[str]     = set()
        self._running       = False
        self._msg_counts:   dict          = {"O": 0, "T": 0}
        self._seen_evs:     _Set[str]     = set()

    # ── Public API ────────────────────────────────────────────────────────────

    def set_subscriptions(self, symbols: _Set[str]) -> None:
        """
        Diff current subscriptions against desired set and send subscribe /
        unsubscribe commands.  Safe to call before connection is established
        (queued until auth_success via _pending).
        """
        add = symbols - self._subscribed
        rem = self._subscribed - symbols
        if add:
            self._send("subscribe",   ",".join(f"O.{s}" for s in add))
        if rem:
            self._send("unsubscribe", ",".join(f"O.{s}" for s in rem))
        self._subscribed = set(symbols)

    def close(self) -> None:
        """Stop the reconnect loop and close the connection."""
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

    def run(self) -> None:
        """
        Blocking.  Connects, authenticates, then dispatches events.
        Reconnects with exponential backoff on any disconnect or error.
        """
        import websocket as _wslib
        _wslib.enableTrace(False)
        self._running = True
        delay = 5
        while self._running:
            log.info("[ws:%s] connecting…", self.ticker)
            try:
                self._ws = _wslib.WebSocketApp(
                    self.WS_URL,
                    on_open    = self._handle_open,
                    on_message = self._handle_message,
                    on_error   = self._handle_error,
                    on_close   = self._handle_close,
                )
                self._ws.run_forever(ping_interval=25, ping_timeout=10)
            except Exception as exc:
                log.error("[ws:%s] run_forever error: %s", self.ticker, exc)
            if not self._running:
                break
            log.info("[ws:%s] reconnecting in %ds…", self.ticker, delay)
            time.sleep(delay)
            delay = min(delay * 2, 120)
        log.info("[ws:%s] stream stopped", self.ticker)

    # ── Internal WS handlers — protocol only, no business logic ──────────────

    def _send(self, action: str, params: str) -> None:
        if not params or not self._ws:
            return
        try:
            self._ws.send(json.dumps({"action": action, "params": params}))
        except Exception as exc:
            log.debug("[ws:%s] send error: %s", self.ticker, exc)

    def _handle_open(self, ws) -> None:
        ws.send(json.dumps({"action": "auth", "params": API_KEY}))

    def _handle_message(self, ws, message: str) -> None:
        try:
            for msg in json.loads(message):
                ev = msg.get("ev")

                if ev == "status":
                    status = msg.get("status", "")
                    if status == "auth_success":
                        log.info("[ws:%s] authenticated", self.ticker)
                        self._send("subscribe", f"T.{self.ticker}")
                        self._on_connected()
                    elif status == "auth_failed":
                        log.error("[ws:%s] auth failed — check POLYGON_API_KEY", self.ticker)
                        ws.close()
                    elif status == "success":
                        # Subscription confirmed — log first few so we can verify T.* landed
                        params = msg.get("message", "")
                        if "T." in params:
                            log.info("[ws:%s] equity trade sub confirmed: %s", self.ticker, params)
                    elif status not in ("connected",):
                        log.debug("[ws:%s] status: %s — %s",
                                  self.ticker, status, msg.get("message", ""))

                elif ev == "O":
                    sym    = msg.get("sym", "")
                    greeks = msg.get("greeks") or {}
                    gamma  = float(greeks.get("gamma") or 0)
                    oi     = int(msg.get("oi") or 0)
                    self._on_option(sym, gamma, oi)
                    self._msg_counts["O"] += 1

                elif ev == "T":
                    sym   = msg.get("sym", "")
                    price = float(msg.get("p") or 0)
                    self._msg_counts["T"] += 1
                    if sym == self.ticker and price > 0:
                        # Log first trade and then every 100th to confirm liveness
                        if self._msg_counts["T"] <= 3 or self._msg_counts["T"] % 100 == 0:
                            log.info("[ws:%s] T event → %s @ %.2f (count=%d)",
                                     self.ticker, sym, price, self._msg_counts["T"])
                        self._on_trade(price)
                    elif sym != self.ticker:
                        # T event for a different symbol — shouldn't happen on options feed
                        log.debug("[ws:%s] T event for unexpected sym=%s (expected %s)",
                                  self.ticker, sym, self.ticker)

                else:
                    # Log unknown event types once so we know what the feed sends
                    if ev and ev not in self._seen_evs:
                        self._seen_evs.add(ev)
                        log.debug("[ws:%s] new event type seen: ev=%s", self.ticker, ev)

        except Exception as exc:
            log.error("[ws:%s] message parse error: %s", self.ticker, exc)

    def _handle_error(self, ws, error) -> None:
        log.error("[ws:%s] error: %s", self.ticker, error)

    def _handle_close(self, ws, code, msg) -> None:
        log.warning("[ws:%s] closed (code=%s msg=%s)", self.ticker, code, msg)
        self._subscribed.clear()       # reset so we re-subscribe on reconnect
        self._msg_counts = {"O": 0, "T": 0}  # fresh counts per connection


# ---------------------------------------------------------------------------
# yfinance spot price — free real-time fallback when Polygon plan is options-only
# ---------------------------------------------------------------------------

def get_spot_yfinance(ticker: str) -> float:
    """
    Fetch current spot price via yfinance (no API key required).
    Uses fast_info.last_price which is a real-time delayed quote (~15 min).
    Falls back to info["regularMarketPrice"] if fast_info is unavailable.

    Only called when Polygon 1-min bars return 0 during market hours
    (plan restriction on Options Starter).
    """
    try:
        import yfinance as yf
        t = yf.Ticker(ticker.upper())
        p = getattr(t.fast_info, "last_price", None)
        if p and p > 0:
            log.info("[spot] %s = %.2f via yfinance", ticker, float(p))
            return float(p)
        # fallback to .info (slower but more complete)
        p = t.info.get("regularMarketPrice") or t.info.get("currentPrice")
        if p and p > 0:
            log.info("[spot] %s = %.2f via yfinance.info", ticker, float(p))
            return float(p)
    except ImportError:
        log.debug("[spot] yfinance not installed — pip install yfinance")
    except Exception as e:
        log.debug("[spot] yfinance error for %s: %s", ticker, e)
    return 0.0


# ---------------------------------------------------------------------------
# Stock trade stream — real-time spot via wss://socket.polygon.io/stocks
# ---------------------------------------------------------------------------

class StockTradeStream:
    """
    Minimal WebSocket client for wss://socket.polygon.io/stocks.

    Subscribes to T.{ticker} (trades) only — one connection per underlying.
    Used solely to get real-time spot price when the options WS doesn't
    deliver equity trade events (e.g. Polygon Options Starter plan).

    Callbacks:
        on_trade(price: float)  — called on every confirmed trade print
        on_connected()          — called after auth_success

    Reconnects with exponential backoff identical to OptionsStreamClient.
    """

    WS_URL = "wss://socket.polygon.io/stocks"

    def __init__(
        self,
        ticker:          str,
        on_trade:        Callable[[float], None],
        on_connected:    Callable[[], None]        = None,
        on_auth_failed:  Callable[[], None]        = None,
    ):
        self.ticker           = ticker.upper()
        self._on_trade        = on_trade
        self._on_connected    = on_connected or (lambda: None)
        self._on_auth_failed  = on_auth_failed or (lambda: None)
        self._ws              = None
        self._running         = False
        self._trade_count     = 0

    def close(self) -> None:
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

    def run(self) -> None:
        """Blocking. Connects and delivers T.{ticker} events via on_trade()."""
        import websocket as _wslib
        self._running = True
        delay = 5
        while self._running:
            log.info("[ws-stocks:%s] connecting…", self.ticker)
            try:
                self._ws = _wslib.WebSocketApp(
                    self.WS_URL,
                    on_open    = self._handle_open,
                    on_message = self._handle_message,
                    on_error   = self._handle_error,
                    on_close   = self._handle_close,
                )
                self._ws.run_forever(ping_interval=25, ping_timeout=10)
            except Exception as exc:
                log.error("[ws-stocks:%s] run_forever error: %s", self.ticker, exc)
            if not self._running:
                break
            log.info("[ws-stocks:%s] reconnecting in %ds…", self.ticker, delay)
            time.sleep(delay)
            delay = min(delay * 2, 120)
        log.info("[ws-stocks:%s] stream stopped", self.ticker)

    def _send(self, action: str, params: str) -> None:
        if self._ws:
            try:
                self._ws.send(json.dumps({"action": action, "params": params}))
            except Exception:
                pass

    def _handle_open(self, ws) -> None:
        ws.send(json.dumps({"action": "auth", "params": API_KEY}))

    def _handle_message(self, ws, message: str) -> None:
        try:
            for msg in json.loads(message):
                ev     = msg.get("ev")
                status = msg.get("status", "")

                if ev == "status":
                    if status == "auth_success":
                        log.info("[ws-stocks:%s] authenticated — subscribing T.%s",
                                 self.ticker, self.ticker)
                        self._send("subscribe", f"T.{self.ticker}")
                        self._on_connected()
                    elif status == "auth_failed":
                        log.warning("[ws-stocks:%s] auth failed — "
                                    "stocks WS requires a Stocks subscription "
                                    "(Options Starter plan does not include it). "
                                    "Spot price will be provided by yfinance poll.", self.ticker)
                        self._running = False      # stop reconnect loop permanently
                        self._on_auth_failed()     # notify caller to clear ws_spot_ts
                        ws.close()
                    elif status == "success":
                        log.info("[ws-stocks:%s] subscription confirmed: %s",
                                 self.ticker, msg.get("message", ""))

                elif ev == "T" and msg.get("sym") == self.ticker:
                    price = float(msg.get("p") or msg.get("c") or 0)
                    if price > 0:
                        self._trade_count += 1
                        if self._trade_count <= 3 or self._trade_count % 500 == 0:
                            log.info("[ws-stocks:%s] trade @ %.2f (count=%d)",
                                     self.ticker, price, self._trade_count)
                        self._on_trade(price)

        except Exception as exc:
            log.error("[ws-stocks:%s] message parse error: %s", self.ticker, exc)

    def _handle_error(self, ws, error) -> None:
        log.error("[ws-stocks:%s] error: %s", self.ticker, error)

    def _handle_close(self, ws, code, msg) -> None:
        log.warning("[ws-stocks:%s] closed (code=%s)", self.ticker, code)


# ---------------------------------------------------------------------------
# Startup diagnostic — remove call from server once spot is confirmed working
# ---------------------------------------------------------------------------

def diagnose_spot_endpoints(ticker: str) -> None:
    """Log full raw responses from every price endpoint for plan-tier debugging."""
    ticker = ticker.upper()
    log.info("=" * 60)
    log.info("SPOT DIAGNOSIS for %s", ticker)
    log.info("=" * 60)

    from datetime import timezone
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    ago_ms = now_ms - 30 * 60 * 1000

    tests = [
        ("options_chain_1item",
         f"/v3/snapshot/options/{ticker}", {"limit": 1}),
        ("last_trade",
         f"/v2/last/trade/{ticker}", {}),
        ("stock_snapshot",
         f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}", {}),
        ("aggs_1min",
         f"/v2/aggs/ticker/{ticker}/range/1/minute/{ago_ms}/{now_ms}",
         {"sort": "desc", "limit": 1, "adjusted": "true"}),
        ("prev_close",
         f"/v2/aggs/ticker/{ticker}/prev", {"adjusted": "true"}),
    ]

    for label, path, params in tests:
        url = (BASE_URL + path) if path.startswith("/") else path
        try:
            r = _session().get(url, params={**params, "apiKey": API_KEY}, timeout=10)
            d = r.json()
            if label == "options_chain_1item":
                ua = ((d.get("results") or [{}])[0]).get("underlying_asset") or {}
                log.info("[diag] %-24s HTTP=%d | underlying_asset=%s",
                         label, r.status_code, json.dumps(ua))
            elif label == "stock_snapshot":
                t = d.get("ticker") or {}
                log.info("[diag] %-24s HTTP=%d | min=%s lastTrade=%s day=%s prevDay=%s",
                         label, r.status_code,
                         t.get("min"), t.get("lastTrade"), t.get("day"), t.get("prevDay"))
            else:
                log.info("[diag] %-24s HTTP=%d | %s",
                         label, r.status_code, json.dumps(d)[:200])
        except Exception as e:
            log.info("[diag] %-24s ERROR: %s", label, e)

    log.info("=" * 60)

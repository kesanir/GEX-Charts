"""
ai_client.py — Anthropic API client
======================================
All Claude API interaction lives here.  gex_server.py imports one function: generate_trade_ideas().

Features
--------
  Multi-MCP support   — configure any number of MCP servers via env or at call time
  Tool-use loop       — handles Claude calling MCP tools across multiple turns automatically
  Prompt builder      — builds the GEX trade-idea prompt from engine data + alerts
  JSON extraction     — robust parsing from text, tool_result, or raw JSON blocks
  10-min cache        — stored here, not in the server, so the server stays stateless

MCP Configuration (env vars)
-----------------------------
  GEX_MCP_SERVERS   — JSON list of {name, url} objects to always include
                       e.g. '[{"name":"brave","url":"https://brave.mcp.example.com"}]'

  GEX_MCP_EXTRA     — same format; merged at runtime (useful for per-request overrides)

Example .env
------------
  GEX_MCP_SERVERS=[{"name":"exa","url":"https://exa.mcp.example.com/sse"},{"name":"polygon","url":"https://polygon.mcp.example.com"}]

Built-in optional MCPs (uncomment or extend in MCP_DEFAULTS below):
  - exa web search  : live news, earnings, macro context
  - polygon mcp     : if/when Polygon exposes an MCP endpoint
  - any custom tool : add {name, url} to MCP_DEFAULTS
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("gex")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_MODEL   = os.getenv("GEX_AI_MODEL", "claude-sonnet-4-20250514")
MAX_TOKENS        = int(os.getenv("GEX_AI_MAX_TOKENS", "4096"))
CACHE_TTL_SEC     = int(os.getenv("GEX_AI_CACHE_TTL",  "600"))   # 10 min default
MAX_TOOL_TURNS    = int(os.getenv("GEX_AI_MAX_TURNS",  "5"))      # max tool-use rounds

_ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
if not _ANTHROPIC_API_KEY:
    log.warning("[ai_client] ANTHROPIC_API_KEY not set — trade ideas will return 401")

# MCP servers always included — extend this list or override via env
MCP_DEFAULTS: List[Dict[str, str]] = [
    # {"name": "exa",     "url": "https://exa.mcp.example.com/sse"},
    # {"name": "polygon", "url": "https://polygon.mcp.example.com/sse"},
]

_trade_cache: Dict[str, dict] = {}


# ---------------------------------------------------------------------------
# MCP helpers
# ---------------------------------------------------------------------------

def load_mcp_servers(extra: Optional[List[dict]] = None) -> List[dict]:
    """
    Build the final MCP server list from three sources (merged, deduplicated by name):
      1. MCP_DEFAULTS (hardcoded above)
      2. GEX_MCP_SERVERS env var (JSON list)
      3. GEX_MCP_EXTRA env var (JSON list — useful for per-deployment overrides)
      4. extra arg (per-call additions, e.g. from an API endpoint query param)

    Each entry must have at least {name, url}.
    The Anthropic API also accepts {name, url, authorization_token} for protected MCPs.

    Example GEX_MCP_SERVERS value:
      [{"name":"brave","url":"https://brave.mcp.example.com/sse"},
       {"name":"exa",  "url":"https://exa.mcp.example.com/sse","authorization_token":"tok_..."}]
    """
    servers: List[dict] = list(MCP_DEFAULTS)

    for env_var in ("GEX_MCP_SERVERS", "GEX_MCP_EXTRA"):
        raw = os.getenv(env_var, "").strip()
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    servers.extend(parsed)
            except json.JSONDecodeError:
                log.warning("[ai_client] Could not parse %s: %s", env_var, raw[:80])

    if extra:
        servers.extend(extra)

    # Deduplicate by name — last definition wins
    seen: Dict[str, dict] = {}
    for s in servers:
        name = s.get("name", "")
        if name:
            seen[name] = s
    result = list(seen.values())

    if result:
        log.debug("[ai_client] MCP servers: %s", [s["name"] for s in result])
    return result


def _mcp_entries(servers: List[dict]) -> List[dict]:
    """Convert {name, url, ?authorization_token} → Anthropic mcp_servers format."""
    out = []
    for s in servers:
        entry: dict = {"type": "url", "name": s["name"], "url": s["url"]}
        if s.get("authorization_token"):
            entry["authorization_token"] = s["authorization_token"]
        if s.get("headers"):
            entry["headers"] = s["headers"]
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# API call + tool-use loop
# ---------------------------------------------------------------------------

async def _call_claude(
    messages:    List[dict],
    system:      Optional[str]  = None,
    mcp_servers: List[dict]     = (),
    max_tokens:  int            = MAX_TOKENS,
    timeout:     float          = 60.0,
) -> List[dict]:
    """
    Single call to /v1/messages.
    Returns the full content block list from the response.
    Raises httpx.HTTPStatusError on non-200.
    """
    body: Dict[str, Any] = {
        "model":      ANTHROPIC_MODEL,
        "max_tokens": max_tokens,
        "messages":   messages,
    }
    if system:
        body["system"] = system
    if mcp_servers:
        body["mcp_servers"] = _mcp_entries(mcp_servers)

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            ANTHROPIC_API_URL,
            headers={
                "Content-Type":      "application/json",
                "x-api-key":         os.getenv("ANTHROPIC_API_KEY", ""),
                "anthropic-version": ANTHROPIC_VERSION,
                "anthropic-beta":    "mcp-client-2025-04-04",
            },
            json=body,
        )

    if resp.status_code != 200:
        body = resp.text[:400]
        log.error("[ai_client] HTTP %d: %s", resp.status_code, body)
        # Surface friendly messages for known billing/auth errors
        try:
            err_msg = resp.json().get("error", {}).get("message", body)
        except Exception:
            err_msg = body
        if resp.status_code in (400, 402) and "credit" in err_msg.lower():
            raise ValueError(
                "Anthropic API credits required. Claude.ai Pro ≠ API credits. "
                "Add credits at console.anthropic.com/settings/billing"
            )
        if resp.status_code == 401:
            raise ValueError("ANTHROPIC_API_KEY missing or invalid — check your .env")
        resp.raise_for_status()

    return resp.json().get("content", [])


async def _run_tool_loop(
    messages:    List[dict],
    system:      Optional[str] = None,
    mcp_servers: List[dict]    = (),
) -> str:
    """
    Multi-turn tool-use loop.

    Claude may call MCP tools across several turns before producing a final text response.
    This handles the full pattern:
      user → assistant (tool_use blocks) → user (tool_result blocks) → assistant (text) → …

    Returns the concatenated final text from the last assistant turn that contains no tool_use.
    """
    history = list(messages)

    for turn in range(MAX_TOOL_TURNS + 1):
        content = await _call_claude(history, system=system, mcp_servers=mcp_servers)

        # Separate tool calls from text
        tool_use_blocks = [b for b in content if b.get("type") == "tool_use"]
        text_blocks     = [b for b in content if b.get("type") == "text"]

        if not tool_use_blocks:
            # No tools called — final response
            return "\n".join(b.get("text", "") for b in text_blocks).strip()

        if turn >= MAX_TOOL_TURNS:
            log.warning("[ai_client] Reached max tool turns (%d) — returning partial text", MAX_TOOL_TURNS)
            return "\n".join(b.get("text", "") for b in text_blocks).strip()

        # Append assistant turn to history
        history.append({"role": "assistant", "content": content})

        # Build tool_result blocks — MCP results are auto-resolved by the API;
        # for non-MCP tool_use (rare in this setup) we return a placeholder.
        tool_results = []
        for tb in tool_use_blocks:
            log.debug("[ai_client] Tool call: %s(%s)", tb.get("name"), str(tb.get("input", {}))[:80])
            # MCP tool results come back in the next API call automatically.
            # We add a stub here only if the API needs explicit tool_result messages.
            tool_results.append({
                "type":       "tool_result",
                "tool_use_id": tb["id"],
                "content":    "",   # MCP results injected by Anthropic infra
            })

        if tool_results:
            history.append({"role": "user", "content": tool_results})

    return ""   # unreachable


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_trade_prompt(gex_data: dict, alerts: List[dict]) -> str:
    """
    Build the GEX trade idea prompt from engine snapshot + recent alerts.
    Structured so Claude can also use MCP tools (e.g. web search for macro context)
    before generating trade ideas.
    """
    lvl    = gex_data.get("levels", {})
    spot   = gex_data.get("spot", 0)
    ticker = gex_data.get("ticker", "")

    alert_lines = (
        [f"  [{a['severity']}] {a['type']}: {a['msg']}" for a in alerts[:5]]
        if alerts else ["  None"]
    )

    return f"""You are a professional options trader and quantitative analyst specializing in \
gamma exposure (GEX) based strategies.

Analyze the following real-time GEX data for {ticker} and generate 2-4 specific, actionable \
options trade ideas for swing trading (1-21 days).

If you have access to web search or news tools, use them to enrich the analysis with:
  - Recent macro events or Fed commentary that could affect vol regime
  - Earnings dates within the option DTE window
  - Any sector-specific news for {ticker}
Incorporate this context into your trade rationale. If no tools are available, proceed with GEX data alone.

== CURRENT GEX DATA ==
Ticker:          {ticker}
Spot Price:      ${spot:.2f}
Net GEX:         {lvl.get('net_gex_b', 0):.4f}B
Regime:          {lvl.get('regime', 'unknown').upper()} (strength: {lvl.get('regime_strength', 0)*100:.1f}%)
GEX Skew:        {lvl.get('gex_skew', 0):.3f} ({'call-heavy' if (lvl.get('gex_skew') or 0) > 1 else 'put-heavy'})

Key Levels:
  Zero Gamma Level: ${lvl.get('zero_gamma') or 'N/A'}  ← regime flip here
  Gamma Wall:       ${lvl.get('gamma_wall') or 'N/A'}  ← max |GEX| magnet
  Call Wall:        ${lvl.get('call_wall') or 'N/A'}  ← dealer resistance
  Put Wall:         ${lvl.get('put_wall') or 'N/A'}  ← dealer support/accelerator
  HV Strike:        ${lvl.get('hv_strike') or 'N/A'}  ← pinning target
  Dist to ZGL:      {(lvl.get('dist_to_flip_pct') or 0):.2f}% {'above' if (lvl.get('dist_to_flip_pct') or 0) > 0 else 'below'}

Contracts loaded: {gex_data.get('contracts', 0)}

Recent Alerts:
{chr(10).join(alert_lines)}

== GEX STRATEGY FRAMEWORK ==
Positive GEX regime: Dealers long gamma → sell rallies/buy dips → mean-reverting, low vol
  → favor premium selling (iron condors, credit spreads, short straddles near gamma wall)
Negative GEX regime: Dealers short gamma → buy rallies/sell dips → trending, high vol
  → favor directional debit spreads, long options

Zero Gamma Level (ZGL) is the most critical level:
  - Spot above ZGL = positive regime (vol suppression) → range trades
  - Spot below ZGL = negative regime (vol amplification) → directional trades
  - ZGL cross = regime change → highest priority signal

Gamma Wall = price magnet into expiry → pin risk trades
Put Wall breach = gamma-accelerated downside → aggressive puts
Call Wall break = gamma squeeze → aggressive calls

== REQUIRED OUTPUT FORMAT ==
After any tool use, respond with ONLY a JSON array — no markdown fences, no explanation.
Each trade idea must follow this exact schema:

[
  {{
    "id": 1,
    "strategy": "Bull Call Spread",
    "direction": "bullish",
    "confidence": "high",
    "setup": "One sentence: why this trade exists given the GEX data",
    "entry_condition": "Specific price/level condition to enter",
    "macro_context": "Any relevant macro/news context used (empty string if none)",
    "legs": [
      {{"action": "BUY",  "type": "CALL", "strike": 0, "expiry_dte": 7, "note": ""}},
      {{"action": "SELL", "type": "CALL", "strike": 0, "expiry_dte": 7, "note": ""}}
    ],
    "target_credit_debit": "debit $X.XX",
    "max_profit": "$X.XX at expiry",
    "max_loss": "$X.XX",
    "stop_loss": "Close if spot breaks below $X",
    "profit_target": "Close at 50% of max profit or spot reaches $X",
    "gex_rationale": "Specific GEX level or regime driving this trade",
    "risk_reward": "1:2.5",
    "tags": ["regime_play", "gamma_wall"]
  }}
]

Strikes MUST be anchored to actual GEX levels provided.
DTE should match regime: positive → 5-14d, negative → 14-30d.
Include at least one neutral/premium-selling idea if regime_strength > 0.5.
Tags from: regime_play, gamma_wall, zgl_bounce, put_wall_support, call_wall_resistance,
           gamma_squeeze, vol_crush, pin_risk, regime_flip, directional.
Confidence: high / medium / low.
"""


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def _extract_json(raw: str) -> list:
    """
    Robustly extract a JSON array from Claude's response text.
    Handles markdown fences, leading text, and trailing commentary.
    """
    text = raw.strip()

    # Strip markdown fences
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            try:
                result = json.loads(part)
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                continue

    # Direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Find first [ ... ] block
    start = text.find("[")
    end   = text.rfind("]")
    if start != -1 and end > start:
        try:
            result = json.loads(text[start:end + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON array from response: {text[:200]}")


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

async def generate_trade_ideas(
    gex_data:    dict,
    alerts:      List[dict],
    ticker:      str,
    force:       bool             = False,
    mcp_extra:   Optional[List[dict]] = None,
) -> dict:
    """
    Generate AI-powered trade ideas for a ticker from its current GEX snapshot.

    Parameters
    ----------
    gex_data   : dict from _engine_to_dict()
    alerts     : recent alerts from AlertEngine
    ticker     : e.g. "SPY"
    force      : bypass cache
    mcp_extra  : additional MCP servers for this specific call
                 e.g. [{"name": "exa", "url": "https://exa.mcp.example.com/sse"}]
                 Merged with GEX_MCP_SERVERS env var and MCP_DEFAULTS.

    Returns
    -------
    dict with keys: ticker, spot, regime, generated_at, ideas, cached, cache_age_s,
                    levels, mcp_servers_used
    """
    ticker    = ticker.upper()
    cache_key = f"{ticker}_trade_ideas"
    now       = time.time()

    # ── Cache check ───────────────────────────────────────────────────────────
    cached = _trade_cache.get(cache_key)
    if cached and not force and now - cached["ts"] < CACHE_TTL_SEC:
        log.info("[trade-ideas] Cache hit for %s (age=%ds)", ticker, int(now - cached["ts"]))
        return {**cached["result"], "cached": True, "cache_age_s": int(now - cached["ts"])}

    # ── MCP setup ─────────────────────────────────────────────────────────────
    mcp_servers = load_mcp_servers(extra=mcp_extra)

    # ── Build prompt and call Claude ──────────────────────────────────────────
    prompt = build_trade_prompt(gex_data, alerts)
    log.info("[trade-ideas] Generating for %s | model=%s | mcps=%s",
             ticker, ANTHROPIC_MODEL, [s["name"] for s in mcp_servers] or "none")

    raw_text = await _run_tool_loop(
        messages    = [{"role": "user", "content": prompt}],
        mcp_servers = mcp_servers,
    )

    if not raw_text:
        raise ValueError("Claude returned empty response")

    # ── Parse JSON ────────────────────────────────────────────────────────────
    ideas = _extract_json(raw_text)
    log.info("[trade-ideas] Got %d ideas for %s", len(ideas), ticker)

    result = {
        "ticker":           ticker,
        "spot":             gex_data.get("spot"),
        "regime":           (gex_data.get("levels") or {}).get("regime"),
        "generated_at":     datetime.utcnow().isoformat(),
        "ideas":            ideas,
        "cached":           False,
        "cache_age_s":      0,
        "levels":           gex_data.get("levels", {}),
        "mcp_servers_used": [s["name"] for s in mcp_servers],
        "model":            ANTHROPIC_MODEL,
    }
    _trade_cache[cache_key] = {"ts": now, "result": result}
    return result


def get_cache_info() -> Dict[str, Any]:
    """Return cache metadata — useful for /api/status."""
    now = time.time()
    return {
        key: {
            "age_s":   int(now - v["ts"]),
            "expires": max(0, int(CACHE_TTL_SEC - (now - v["ts"]))),
        }
        for key, v in _trade_cache.items()
    }

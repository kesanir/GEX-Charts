"""
ai_client_grok.py — xAI Grok API client
=======================================
Adapted from Anthropic version for Grok (xAI API).
All Grok API interaction lives here. gex_server.py imports: generate_trade_ideas().

Features
--------
  - MCP support via local proxy functions (calls your remote MCP URLs)
  - Tool-use loop (handles Grok calling tools across multiple turns)
  - Prompt builder (same as Anthropic version)
  - JSON extraction (robust parsing)
  - 10-min cache

MCP Configuration (env vars)
----------------------------
  GEX_MCP_SERVERS   — JSON list of {name, url} objects
  GEX_MCP_EXTRA     — merged at runtime

Example .env
------------
  GEX_MCP_SERVERS=[{"name":"brave","url":"https://brave.mcp.example.com/sse"}]
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
log = logging.getLogger("gex")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

XAI_API_KEY       = os.getenv("XAI_API_KEY", "")
if not XAI_API_KEY:
    log.warning("[ai_client_grok] XAI_API_KEY not set — trade ideas will return errors")

BASE_URL          = "https://api.x.ai/v1"
GROK_MODEL        = os.getenv("GEX_AI_MODEL", "grok-4-1-fast-reasoning")
MAX_TOKENS        = int(os.getenv("GEX_AI_MAX_TOKENS", "8192"))
CACHE_TTL_SEC     = int(os.getenv("GEX_AI_CACHE_TTL", "600"))   # 10 min
MAX_TOOL_TURNS    = int(os.getenv("GEX_AI_MAX_TURNS", "5"))

_client = AsyncOpenAI(
    api_key=XAI_API_KEY,
    base_url=BASE_URL,
)

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
    Also accepts {name, url, authorization_token} for protected MCPs.
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
                log.warning("[ai_client_grok] Could not parse %s: %s", env_var, raw[:80])

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
        log.debug("[ai_client_grok] MCP servers: %s", [s["name"] for s in result])
    return result


async def call_mcp_proxy(name: str, input_data: dict, servers: List[dict]) -> dict:
    """Proxy call to remote MCP server via HTTP."""
    server = next((s for s in servers if s["name"] == name), None)
    if not server:
        raise ValueError(f"MCP server '{name}' not found")

    url = server["url"]
    headers = {"Content-Type": "application/json"}
    if server.get("authorization_token"):
        headers["Authorization"] = f"Bearer {server['authorization_token']}"

    async with httpx.AsyncClient(timeout=240.0) as client:
        resp = await client.post(url, json=input_data, headers=headers)
        resp.raise_for_status()
        return resp.json()


# Define tool schemas (OpenAI-style) — one per MCP server
def get_mcp_tools(mcp_servers: List[dict]) -> List[dict]:
    tools = []
    for server in mcp_servers:
        name = server["name"]
        tools.append({
            "type": "function",
            "function": {
                "name": f"mcp_{name}",
                "description": f"Call remote MCP tool '{name}' (e.g. search, data fetch)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Main query or input"},
                        "params": {
                            "type": "object",
                            "description": "Additional parameters (dict)",
                            "additionalProperties": True,
                        },
                    },
                    "required": ["query"],
                },
            },
        })
    return tools


# ---------------------------------------------------------------------------
# Tool execution dispatcher
# ---------------------------------------------------------------------------

async def execute_tool(tool_call, mcp_servers: List[dict]) -> dict:
    func_name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    if func_name.startswith("mcp_"):
        mcp_name = func_name[4:]  # strip prefix
        result = await call_mcp_proxy(mcp_name, args, mcp_servers)
        return {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": func_name,
            "content": json.dumps(result),
        }
    else:
        raise ValueError(f"Unknown tool: {func_name}")


# ---------------------------------------------------------------------------
# API call + tool loop
# ---------------------------------------------------------------------------

async def _call_grok(
    messages:    List[dict],
    tools:       Optional[List[dict]] = None,
    max_tokens:  int                  = MAX_TOKENS,
    temperature: float                = 0.4,
    timeout:     float                = 60.0,
) -> dict:
    """
    Single call to /v1/chat/completions.
    Returns the full message from the first choice.
    Raises httpx.HTTPStatusError on non-200.
    """
    body: Dict[str, Any] = {
        "model":      GROK_MODEL,
        "messages":   messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{BASE_URL}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {XAI_API_KEY}",
            },
            json=body,
        )

    if resp.status_code != 200:
        body_text = resp.text[:400]
        log.error("[ai_client_grok] HTTP %d: %s", resp.status_code, body_text)
        # Surface friendly messages for known errors
        try:
            err_msg = resp.json().get("error", {}).get("message", body_text)
        except Exception:
            err_msg = body_text
        if resp.status_code == 401:
            raise ValueError("XAI_API_KEY missing or invalid — check your .env")
        resp.raise_for_status()

    return resp.json()["choices"][0]["message"]


async def _run_tool_loop(
    messages:    List[dict],
    mcp_servers: List[dict]     = (),
) -> str:
    """
    Multi-turn tool-use loop for Grok (OpenAI-compatible).

    Grok may call tools across several turns before producing a final text response.
    This handles the full pattern:
      user → assistant (tool_calls) → tool responses → assistant (text) → …

    Returns the final content from the last assistant turn that contains no tool_calls.
    """
    history = list(messages)
    tools = get_mcp_tools(mcp_servers) if mcp_servers else None

    for turn in range(MAX_TOOL_TURNS + 1):
        message = await _call_grok(history, tools=tools)

        # Append assistant message to history
        history.append(message)

        if not message.get("tool_calls"):
            # No tools called — final response
            return message.get("content", "").strip()

        if turn >= MAX_TOOL_TURNS:
            log.warning("[ai_client_grok] Reached max tool turns (%d) — returning partial text", MAX_TOOL_TURNS)
            return message.get("content", "").strip()

        # Execute tools and add responses
        tool_results = []
        for tool_call in message["tool_calls"]:
            log.debug("[ai_client_grok] Tool call: %s(%s)", tool_call["function"]["name"], tool_call["function"]["arguments"][:80])
            result_msg = await execute_tool(tool_call, mcp_servers)
            tool_results.append(result_msg)

        history.extend(tool_results)

    return ""  # unreachable


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_trade_prompt(gex_data: dict, alerts: List[dict]) -> str:
    """
    Build the GEX trade idea prompt from engine snapshot + recent alerts.
    Structured so Grok can also use MCP tools (e.g. web search for macro context)
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
    "setup": "One sentence: why this trade exists given given the GEX data",
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
    Robustly extract a JSON array from Grok's response text.
    Handles markdown fences, leading text, and trailing commentary.
    Returns [] on failure for safety.
    """
    if not raw or raw.strip() == "":
        log.warning("[extract_json] Grok returned empty or whitespace response")
        return []

    log.debug("[extract_json] Raw Grok output (first 1000 chars):\n%s", raw[:1000])

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

    log.error("[extract_json] Could not extract JSON array from response: %s", raw[:200])
    return []  # Safe fallback


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

    # ── Build prompt and call Grok ────────────────────────────────────────────
    prompt = build_trade_prompt(gex_data, alerts)
    log.info("[trade-ideas] Generating for %s | model=%s | mcps=%s",
             ticker, GROK_MODEL, [s["name"] for s in mcp_servers] or "none")

    raw_text = await _run_tool_loop(
        messages    = [{"role": "user", "content": prompt}],
        mcp_servers = mcp_servers,
    )

    log.info("[trade-ideas] Raw Grok response length: %d chars | first 400: %s",
             len(raw_text), raw_text[:400])

    if not raw_text:
        raise ValueError("Grok returned empty response")

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
        "model":            GROK_MODEL,
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
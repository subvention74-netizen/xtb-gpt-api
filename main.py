import io
import math
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import feedparser
import pandas as pd
import pdfplumber
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from cachetools import TTLCache, cached
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

APP_NAME = "XTB France Market Tools"
APP_VERSION = "1.0.0"

XTB_HTML_URL = "https://www.xtb.com/fr/specificites-des-instruments"
XTB_DOCS_URL = "https://www.xtb.com/fr/specificites-des-instruments/documents"
XTB_EQUITY_PDF = "https://www.xtb.com/fr/equity-table.pdf"
XTB_OMI_PDF = "https://www.xtb.com/fr/specification_table_omi-fr.pdf"
XTB_FX_COMMO_PDF = "https://www.xtb.com/fr/fichiers/table-des-spe-cifications-forex-matie-res-premie-res-indices.pdf"

USER_AGENT = {
    "User-Agent": "Mozilla/5.0 (compatible; XTB-France-Action/1.0; +https://example.com)"
}

# 12h cache for XTB instrument catalogs
xtb_catalog_cache = TTLCache(maxsize=8, ttl=60 * 60 * 12)

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "Free market data helper for a Custom GPT. "
        "Primary source for XTB France availability is XTB official instrument documentation."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return float(value)
    except Exception:
        return None


def normalize_symbol(symbol: str) -> str:
    return re.sub(r"\s+", "", (symbol or "").strip()).upper()


def normalize_text(text: str) -> str:
    text = (text or "").upper()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_pdf_text(url: str) -> str:
    r = requests.get(url, headers=USER_AGENT, timeout=60)
    r.raise_for_status()
    with pdfplumber.open(io.BytesIO(r.content)) as pdf:
        pages = []
        for page in pdf.pages:
            txt = page.extract_text() or ""
            if txt:
                pages.append(txt)
        return "\n".join(pages)


def fetch_html_text(url: str) -> str:
    r = requests.get(url, headers=USER_AGENT, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    return soup.get_text(" ", strip=True)


@cached(xtb_catalog_cache)
def build_xtb_catalog() -> Dict[str, Any]:
    """
    Builds a searchable text catalog from official XTB France sources.
    Priority:
      1) official XTB web page
      2) official XTB PDFs
    """
    sources = []
    combined_text_parts = []

    for label, url in [
        ("xtb_html", XTB_HTML_URL),
        ("xtb_equity_pdf", XTB_EQUITY_PDF),
        ("xtb_omi_pdf", XTB_OMI_PDF),
        ("xtb_fx_pdf", XTB_FX_COMMO_PDF),
    ]:
        try:
            if url.endswith(".pdf"):
                txt = parse_pdf_text(url)
            else:
                txt = fetch_html_text(url)
            combined_text_parts.append(txt)
            sources.append({"label": label, "url": url, "status": "ok"})
        except Exception as exc:
            sources.append({"label": label, "url": url, "status": f"error: {exc}"})

    combined = "\n".join(combined_text_parts)
    return {
        "built_at_utc": utc_now_iso(),
        "text": combined,
        "sources": sources,
    }


def find_xtb_matches(symbol: str, text: str) -> List[Dict[str, str]]:
    """
    Returns a few matching lines around the symbol.
    """
    norm_symbol = normalize_symbol(symbol)
    matches = []
    for raw_line in text.splitlines():
        line = normalize_text(raw_line)
        if not line:
            continue

        # exact symbol token hit
        if re.search(rf"(?<![A-Z0-9]){re.escape(norm_symbol)}(?![A-Z0-9])", line):
            matches.append({"match_type": "symbol_exact", "line": raw_line.strip()})
            continue

        # fallback: common market suffix normalization, e.g. AAPL.US <-> AAPL
        base = norm_symbol.split(".")[0]
        if len(base) >= 2 and re.search(rf"(?<![A-Z0-9]){re.escape(base)}(\.[A-Z]{{2,4}})?(?![A-Z0-9])", line):
            matches.append({"match_type": "symbol_base", "line": raw_line.strip()})

    # deduplicate and cap
    seen = set()
    uniq = []
    for m in matches:
        key = (m["match_type"], m["line"])
        if key not in seen:
            seen.add(key)
            uniq.append(m)
        if len(uniq) >= 10:
            break
    return uniq


def map_interval(interval: str) -> str:
    allowed = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}
    if interval not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported interval: {interval}")
    return "60m" if interval == "1h" else interval


def period_for_interval(interval: str) -> str:
    # yfinance constraints:
    # 1m only last ~7d, intraday under 60d
    if interval == "1m":
        return "7d"
    return "60d"


def build_quote(ticker_obj: yf.Ticker, symbol: str) -> Dict[str, Any]:
    info = {}
    fast_info = {}
    try:
        fast_info = dict(ticker_obj.fast_info or {})
    except Exception:
        fast_info = {}
    try:
        info = ticker_obj.info or {}
    except Exception:
        info = {}

    last_price = (
        safe_float(fast_info.get("lastPrice"))
        or safe_float(info.get("currentPrice"))
        or safe_float(info.get("regularMarketPrice"))
    )

    previous_close = (
        safe_float(fast_info.get("previousClose"))
        or safe_float(info.get("previousClose"))
        or safe_float(info.get("regularMarketPreviousClose"))
    )

    bid = safe_float(info.get("bid"))
    ask = safe_float(info.get("ask"))
    currency = info.get("currency") or fast_info.get("currency")
    exchange = info.get("exchange") or info.get("fullExchangeName")
    short_name = info.get("shortName") or info.get("longName")

    market_time = info.get("regularMarketTime")
    if isinstance(market_time, (int, float)):
        market_timestamp = datetime.fromtimestamp(market_time, tz=timezone.utc).isoformat()
    else:
        market_timestamp = utc_now_iso()

    return {
        "symbol": symbol,
        "name": short_name,
        "exchange": exchange,
        "currency": currency,
        "last": last_price,
        "bid": bid,
        "ask": ask,
        "previous_close": previous_close,
        "timestamp_utc": market_timestamp,
        "data_source": "yfinance",
        "notes": [
            "Free source; values may be delayed depending on venue/instrument.",
            "Bid/ask can be missing on some instruments.",
        ],
    }


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "server_time_utc": utc_now_iso(),
        "endpoints": [
            "/check_xtb_instrument",
            "/get_realtime_quote",
            "/get_intraday_bars",
            "/get_market_news",
            "/health",
            "/privacy",
        ],
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "server_time_utc": utc_now_iso()}


@app.get("/privacy", include_in_schema=False)
def privacy() -> Dict[str, Any]:
    return {
        "message": (
            "This demo service does not require authentication and does not intentionally store personal data. "
            "Server logs may contain IP address, timestamps, and requested symbols for operational debugging."
        ),
        "contact": "replace-with-your-email@example.com",
        "updated_at_utc": utc_now_iso(),
    }


@app.get("/check_xtb_instrument")
def check_xtb_instrument(
    symbol: str = Query(..., description="Instrument symbol, e.g. AAPL.US, EURUSD, EU50, GOLD")
) -> Dict[str, Any]:
    symbol = normalize_symbol(symbol)
    catalog = build_xtb_catalog()
    text = catalog["text"]
    matches = find_xtb_matches(symbol, text)

    # heuristic status
    available = len(matches) > 0

    return {
        "symbol": symbol,
        "available_on_xtb_france": available,
        "match_count": len(matches),
        "matches": matches,
        "checked_at_utc": utc_now_iso(),
        "primary_sources": [
            XTB_HTML_URL,
            XTB_EQUITY_PDF,
            XTB_OMI_PDF,
            XTB_FX_COMMO_PDF,
        ],
        "source_fetch_report": catalog["sources"],
        "notes": [
            "Primary verification is based on official XTB France sources.",
            "If the exact symbol naming differs between your data feed and XTB naming, verify manually in XTB as final confirmation.",
        ],
    }


@app.get("/get_realtime_quote")
def get_realtime_quote(
    symbol: str = Query(..., description="Yahoo Finance symbol, e.g. AAPL, MSFT, EURUSD=X, ^GSPC, GC=F")
) -> Dict[str, Any]:
    symbol = symbol.strip()
    try:
        ticker_obj = yf.Ticker(symbol)
        payload = build_quote(ticker_obj, symbol)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to fetch quote for {symbol}: {exc}") from exc

    if payload["last"] is None and payload["bid"] is None and payload["ask"] is None:
        raise HTTPException(status_code=404, detail=f"No quote data available for {symbol}")

    return payload


@app.get("/get_intraday_bars")
def get_intraday_bars(
    symbol: str = Query(..., description="Yahoo Finance symbol"),
    interval: str = Query("5m", description="1m, 2m, 5m, 15m, 30m, 60m, 1h"),
    limit: int = Query(100, ge=10, le=1000, description="Max number of bars to return"),
) -> Dict[str, Any]:
    symbol = symbol.strip()
    yf_interval = map_interval(interval)
    period = period_for_interval(interval)

    try:
        ticker_obj = yf.Ticker(symbol)
        df = ticker_obj.history(period=period, interval=yf_interval, auto_adjust=False, prepost=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to fetch bars for {symbol}: {exc}") from exc

    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No intraday bars available for {symbol} at {interval}")

    df = df.tail(limit).copy()
    df = df.reset_index()

    time_col = None
    for candidate in ("Datetime", "Date"):
        if candidate in df.columns:
            time_col = candidate
            break
    if time_col is None:
        raise HTTPException(status_code=500, detail="Unexpected bar format returned by upstream source")

    bars = []
    for _, row in df.iterrows():
        dt_val = row[time_col]
        if hasattr(dt_val, "to_pydatetime"):
            dt_val = dt_val.to_pydatetime()
        if isinstance(dt_val, datetime):
            if dt_val.tzinfo is None:
                dt_iso = dt_val.replace(tzinfo=timezone.utc).isoformat()
            else:
                dt_iso = dt_val.astimezone(timezone.utc).isoformat()
        else:
            dt_iso = str(dt_val)

        bars.append(
            {
                "timestamp_utc": dt_iso,
                "open": safe_float(row.get("Open")),
                "high": safe_float(row.get("High")),
                "low": safe_float(row.get("Low")),
                "close": safe_float(row.get("Close")),
                "volume": safe_float(row.get("Volume")),
            }
        )

    return {
        "symbol": symbol,
        "interval": interval,
        "period_used": period,
        "count": len(bars),
        "bars": bars,
        "data_source": "yfinance",
        "fetched_at_utc": utc_now_iso(),
        "notes": [
            "Free source; intraday history depth depends on interval and upstream constraints.",
        ],
    }


@app.get("/get_market_news")
def get_market_news(
    symbol: str = Query(..., description="Yahoo Finance symbol or company query"),
    limit: int = Query(8, ge=1, le=20),
) -> Dict[str, Any]:
    symbol = symbol.strip()
    ticker_obj = yf.Ticker(symbol)

    company_name = None
    try:
        info = ticker_obj.info or {}
        company_name = info.get("shortName") or info.get("longName")
    except Exception:
        company_name = None

    query = company_name or symbol
    rss_url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}%20when:7d&hl=fr&gl=FR&ceid=FR:fr"
    feed = feedparser.parse(rss_url)

    items = []
    for entry in feed.entries[:limit]:
        published = entry.get("published", "")
        items.append(
            {
                "title": entry.get("title"),
                "link": entry.get("link"),
                "published": published,
                "source": (entry.get("source") or {}).get("title"),
                "summary": entry.get("summary", "")[:500],
            }
        )

    return {
        "symbol": symbol,
        "query_used": query,
        "count": len(items),
        "items": items,
        "data_source": "Google News RSS",
        "fetched_at_utc": utc_now_iso(),
        "notes": [
            "News is free and query-based; relevance should be reviewed before acting.",
        ],
    }


# Optional convenience endpoint to reduce GPT tool chatter
@app.get("/get_market_packet")
def get_market_packet(
    xtb_symbol: str = Query(..., description="XTB symbol to check, e.g. AAPL.US or EURUSD"),
    data_symbol: str = Query(..., description="Yahoo Finance symbol, e.g. AAPL or EURUSD=X"),
    interval: str = Query("5m"),
    bars_limit: int = Query(100, ge=10, le=1000),
    news_limit: int = Query(8, ge=1, le=20),
) -> Dict[str, Any]:
    xtb = check_xtb_instrument(xtb_symbol)
    quote = get_realtime_quote(data_symbol)
    bars = get_intraday_bars(data_symbol, interval, bars_limit)
    news = get_market_news(data_symbol, news_limit)
    return {
        "xtb_check": xtb,
        "quote": quote,
        "bars": bars,
        "news": news,
        "fetched_at_utc": utc_now_iso(),
        "notes": [
            "Use this endpoint only if you prefer a single action call.",
        ],
    }

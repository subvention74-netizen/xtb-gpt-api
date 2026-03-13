from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import yfinance as yf
import httpx
from bs4 import BeautifulSoup
import feedparser
from datetime import datetime, timezone

app = FastAPI(
    title="XTB GPT Custom Action API",
    version="1.0.0",
    description="API gratuite pour vérifier la disponibilité XTB France, récupérer un prix, des bougies intraday et des news."
)

XTB_REFERENCE_URL = "https://www.xtb.com/fr/specificites-des-instruments"


class XTBInstrumentResponse(BaseModel):
    symbol: str
    xtb_france_available: Optional[bool]
    source_checked: str
    matched_text: Optional[str] = None
    note: str


class QuoteResponse(BaseModel):
    symbol: str
    source: str
    last_price: Optional[float]
    bid: Optional[float]
    ask: Optional[float]
    currency: Optional[str]
    exchange: Optional[str]
    timestamp_utc: str
    note: str


class BarItem(BaseModel):
    timestamp: str
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    close: Optional[float]
    volume: Optional[float]


class BarsResponse(BaseModel):
    symbol: str
    interval: str
    source: str
    bars: List[BarItem]
    note: str


class NewsItem(BaseModel):
    title: str
    link: str
    published: Optional[str] = None
    source: Optional[str] = None


class NewsResponse(BaseModel):
    symbol: str
    source: str
    items: List[NewsItem]
    note: str


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@app.get("/")
def root():
    return {
        "message": "API XTB GPT active",
        "docs": "/docs"
    }


@app.get("/check_xtb_instrument", response_model=XTBInstrumentResponse, operation_id="check_xtb_instrument")
async def check_xtb_instrument(symbol: str = Query(..., description="Ticker ou symbole, ex: AAPL")):
    symbol_upper = symbol.upper().strip()

    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            r = await client.get(XTB_REFERENCE_URL)
            text = r.text
            page_text = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)

        found = symbol_upper in page_text.upper()

        return XTBInstrumentResponse(
            symbol=symbol_upper,
            xtb_france_available=found,
            source_checked=XTB_REFERENCE_URL,
            matched_text=symbol_upper if found else None,
            note=(
                "Vérification basée prioritairement sur la page officielle XTB France. "
                "Si le symbole n'est pas trouvé, cela ne prouve pas forcément l'absence de l'instrument : "
                "la page peut être structurée différemment ou renvoyer vers des documents externes."
            )
        )
    except Exception as e:
        return XTBInstrumentResponse(
            symbol=symbol_upper,
            xtb_france_available=None,
            source_checked=XTB_REFERENCE_URL,
            matched_text=None,
            note=f"Impossible de confirmer automatiquement la disponibilité XTB France : {e}"
        )


@app.get("/get_realtime_quote", response_model=QuoteResponse, operation_id="get_realtime_quote")
async def get_realtime_quote(symbol: str = Query(..., description="Ticker ou symbole, ex: AAPL")):
    symbol_upper = symbol.upper().strip()

    try:
        ticker = yf.Ticker(symbol_upper)
        info = ticker.fast_info if hasattr(ticker, "fast_info") else {}
        hist = ticker.history(period="1d", interval="1m")

        last_price = None
        bid = None
        ask = None
        currency = None
        exchange = None

        try:
            if hist is not None and not hist.empty:
                last_price = float(hist["Close"].dropna().iloc[-1])
        except Exception:
            pass

        try:
            bid = float(info.get("bid")) if info.get("bid") is not None else None
        except Exception:
            pass

        try:
            ask = float(info.get("ask")) if info.get("ask") is not None else None
        except Exception:
            pass

        try:
            currency = info.get("currency")
        except Exception:
            pass

        try:
            exchange = info.get("exchange")
        except Exception:
            pass

        return QuoteResponse(
            symbol=symbol_upper,
            source="yfinance / Yahoo Finance",
            last_price=last_price,
            bid=bid,
            ask=ask,
            currency=currency,
            exchange=exchange,
            timestamp_utc=now_utc_iso(),
            note="Source gratuite. Selon l'instrument, les données peuvent être retardées, incomplètes ou indisponibles."
        )

    except Exception as e:
        return QuoteResponse(
            symbol=symbol_upper,
            source="yfinance / Yahoo Finance",
            last_price=None,
            bid=None,
            ask=None,
            currency=None,
            exchange=None,
            timestamp_utc=now_utc_iso(),
            note=f"Erreur de récupération du prix : {e}"
        )


@app.get("/get_intraday_bars", response_model=BarsResponse, operation_id="get_intraday_bars")
async def get_intraday_bars(
    symbol: str = Query(..., description="Ticker ou symbole, ex: AAPL"),
    interval: str = Query(..., description="1m, 5m ou 1h")
):
    symbol_upper = symbol.upper().strip()

    interval_map = {
        "1m": ("1m", "1d"),
        "5m": ("5m", "5d"),
        "1h": ("60m", "1mo"),
    }

    if interval not in interval_map:
        return BarsResponse(
            symbol=symbol_upper,
            interval=interval,
            source="yfinance / Yahoo Finance",
            bars=[],
            note="Intervalle invalide. Utiliser uniquement 1m, 5m ou 1h."
        )

    yf_interval, period = interval_map[interval]

    try:
        ticker = yf.Ticker(symbol_upper)
        hist = ticker.history(period=period, interval=yf_interval)

        bars: List[BarItem] = []

        if hist is not None and not hist.empty:
            hist = hist.tail(100)
            for idx, row in hist.iterrows():
                bars.append(
                    BarItem(
                        timestamp=str(idx),
                        open=float(row["Open"]) if row["Open"] == row["Open"] else None,
                        high=float(row["High"]) if row["High"] == row["High"] else None,
                        low=float(row["Low"]) if row["Low"] == row["Low"] else None,
                        close=float(row["Close"]) if row["Close"] == row["Close"] else None,
                        volume=float(row["Volume"]) if row["Volume"] == row["Volume"] else None,
                    )
                )

        return BarsResponse(
            symbol=symbol_upper,
            interval=interval,
            source="yfinance / Yahoo Finance",
            bars=bars,
            note="Source gratuite. Les bougies peuvent être retardées ou indisponibles selon l'actif."
        )
    except Exception as e:
        return BarsResponse(
            symbol=symbol_upper,
            interval=interval,
            source="yfinance / Yahoo Finance",
            bars=[],
            note=f"Erreur de récupération des bougies : {e}"
        )


@app.get("/get_market_news", response_model=NewsResponse, operation_id="get_market_news")
async def get_market_news(symbol: str = Query(..., description="Ticker ou symbole, ex: AAPL")):
    symbol_upper = symbol.upper().strip()

    rss_url = f"https://news.google.com/rss/search?q={symbol_upper}%20stock&hl=fr&gl=FR&ceid=FR:fr"

    try:
        feed = feedparser.parse(rss_url)
        items: List[NewsItem] = []

        for entry in feed.entries[:10]:
            items.append(
                NewsItem(
                    title=entry.get("title", ""),
                    link=entry.get("link", ""),
                    published=entry.get("published", None),
                    source=getattr(entry, "source", {}).get("title") if getattr(entry, "source", None) else None
                )
            )

        return NewsResponse(
            symbol=symbol_upper,
            source="Google News RSS",
            items=items,
            note="News récentes issues d'un flux RSS gratuit. Vérifier la source primaire avant décision."
        )
    except Exception as e:
        return NewsResponse(
            symbol=symbol_upper,
            source="Google News RSS",
            items=[],
            note=f"Erreur de récupération des news : {e}"
        )

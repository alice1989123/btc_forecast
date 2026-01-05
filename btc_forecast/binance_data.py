# btc_forecast/binance_data.py
import os
from typing import Any, List, Optional, Union
from dotenv import load_dotenv
from binance.client import Client

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")


_client = Client(API_KEY, API_SECRET)

StartArg = Union[int, str, None]

def get_binance_data(coin: str, start: StartArg, end: StartArg, interval: str) -> List[List[Any]]:
    """
    start/end can be:
      - int milliseconds since epoch
      - date string accepted by Binance ("1 Jan, 2026") etc.
      - None
    """
    if not interval:
        raise ValueError("interval is required.")
    if not coin:
        raise ValueError("coin is required.")

    # If someone accidentally passes seconds, convert to ms
    def _coerce_ms(x: StartArg) -> StartArg:
        if isinstance(x, int) and x < 10_000_000_000:  # < ~Sat Nov 20 2286 in seconds
            return x * 1000
        return x

    start = _coerce_ms(start)
    end = _coerce_ms(end)

    return _client.get_historical_klines(
        symbol=coin,
        interval=interval,
        start_str=start,
        end_str=end,
    )

def get_tickers():
    return _client.get_all_tickers()

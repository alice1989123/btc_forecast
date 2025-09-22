# common/rolling.py
from __future__ import annotations
import logging
import pandas as pd

log = logging.getLogger(__name__)

def parse_window(raw) -> int | pd.Timedelta:
    """
    Accepts 30, "30", "30D", "45min", pd.Timedelta.
    Returns an int (periods) or a Timedelta for time-based rolling.
    """
    if isinstance(raw, pd.Timedelta):
        return raw
    if isinstance(raw, (int, float)):
        return int(raw)
    s = str(raw).strip()
    if s.isdigit():
        return int(s)
    # allow strings like "30D", "45min", "2H"
    try:
        return pd.to_timedelta(s)
    except Exception:
        # fallback: safest is int if numeric-ish, else raise
        raise ValueError(f"Unrecognized window value: {raw!r}")

def window_for_index(idx: pd.Index, win: int | pd.Timedelta) -> int | pd.Timedelta:
    """
    If idx is DatetimeIndex and win is int -> convert to Timedelta using median step.
    Else return win as-is.
    """
    if isinstance(idx, pd.DatetimeIndex) and isinstance(win, int) and len(idx) >= 2:
        step = (idx[1:] - idx[:-1]).median()
        if pd.notna(step) and step > pd.Timedelta(0):
            return step * win
    return win

def resolve_rolling_window(idx: pd.Index, raw_window) -> int | pd.Timedelta:
    """
    One-shot resolver: parse raw_window and adapt to index.
    Also logs details at DEBUG (no prints).
    """
    win = parse_window(raw_window)
    final = window_for_index(idx, win)
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            "resolve_rolling_window: raw=%r(%s) parsed=%r(%s) final=%r(%s) index=%s",
            raw_window, type(raw_window).__name__,
            win, type(win).__name__,
            final, type(final).__name__,
            type(idx).__name__,
        )
    return final

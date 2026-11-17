import math

import pandas as pd


def num(x, digits: int = 4) -> float | None:
    """A rounded float, or None for missing values, so tool output stays compact."""
    if x is None:
        return None
    try:
        value = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) or math.isinf(value) else round(value, digits)


def iso(d) -> str | None:
    """ISO date string for a date, datetime or pandas timestamp."""
    if d is None or pd.isna(d):
        return None
    return pd.Timestamp(d).date().isoformat()

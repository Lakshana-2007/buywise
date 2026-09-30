"""
Deterministic currency conversion.

Uses only the fixed, dated exchange rates provided in exchange_rates.csv.
NO live API calls. NO invented rates.

Strategy:
1. Find the rate row where rate_date <= event_date (closest prior month).
2. If no prior rate exists, take the earliest available rate.
3. Support multi-hop conversion via USD as a bridge currency when a
   direct rate is unavailable (e.g. EUR -> ZAR via EUR->USD->ZAR).
4. If a rate genuinely does not exist, raise ConversionError — never
   silently return 1.0 or invent a rate.

This module is pure function — no I/O, no side effects.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from buywise.core.models import ExchangeRate


class ConversionError(Exception):
    """Raised when a required exchange rate cannot be found."""


def _find_rate(
    rates: list[ExchangeRate],
    from_currency: str,
    to_currency: str,
    on_date: date,
) -> Optional[float]:
    """
    Find the exchange rate for from_currency -> to_currency on or before on_date.
    Returns the rate value, or None if no matching rate is found.
    """
    if from_currency == to_currency:
        return 1.0

    # Filter to matching pairs
    matching = [
        r for r in rates
        if r.from_currency == from_currency and r.to_currency == to_currency
        and r.rate_date <= on_date
    ]
    if not matching:
        # Fall back to the earliest available rate for this pair
        all_pair = [
            r for r in rates
            if r.from_currency == from_currency and r.to_currency == to_currency
        ]
        if all_pair:
            return min(all_pair, key=lambda r: r.rate_date).rate
        return None

    # Take the most recent rate on or before on_date
    return max(matching, key=lambda r: r.rate_date).rate


def convert(
    amount: float,
    from_currency: str,
    to_currency: str,
    on_date: date,
    rates: list[ExchangeRate],
) -> float:
    """
    Convert `amount` from `from_currency` to `to_currency` using
    the provided exchange rates for `on_date`.

    Raises ConversionError if the conversion cannot be performed.
    """
    if from_currency == to_currency:
        return round(amount, 2)

    # Try direct rate
    rate = _find_rate(rates, from_currency, to_currency, on_date)
    if rate is not None:
        return round(amount * rate, 2)

    # Try inverse rate
    inv_rate = _find_rate(rates, to_currency, from_currency, on_date)
    if inv_rate is not None and inv_rate != 0:
        return round(amount / inv_rate, 2)

    # Try bridge via USD
    for bridge in ("USD", "EUR"):
        if bridge == from_currency or bridge == to_currency:
            continue
        r1 = _find_rate(rates, from_currency, bridge, on_date)
        r2 = _find_rate(rates, bridge, to_currency, on_date)
        if r1 is not None and r2 is not None:
            return round(amount * r1 * r2, 2)

        # Bridge with inverse legs
        r1i = _find_rate(rates, bridge, from_currency, on_date)
        r2i = _find_rate(rates, bridge, to_currency, on_date)
        if r1i is not None and r1i != 0 and r2i is not None:
            return round((amount / r1i) * r2i, 2)

        r1 = _find_rate(rates, from_currency, bridge, on_date)
        r2i = _find_rate(rates, to_currency, bridge, on_date)
        if r1 is not None and r2i is not None and r2i != 0:
            return round((amount * r1) / r2i, 2)

    raise ConversionError(
        f"Cannot convert {from_currency} -> {to_currency} on {on_date}: "
        f"no direct or bridge rate found in provided exchange_rates."
    )


def needs_conversion(event_currency: str, home_currency: str) -> bool:
    return event_currency != home_currency

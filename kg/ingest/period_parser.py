"""Parse period strings from column names / sheet metadata → Period nodes."""
from __future__ import annotations

import re
from dataclasses import dataclass

# e.g. "52 W bis 29/03/26", "4 W bis 29/03/26", "13 W bis 29/03/26"
_PERIOD_RE = re.compile(
    r"(?P<window>\d+)\s*W\s*bis\s*(?P<end_date>\d{1,2}/\d{1,2}/\d{2,4})",
    re.IGNORECASE,
)

# e.g. "YTD 2025", "MAT 2025"
_NAMED_RE = re.compile(r"(?P<type>YTD|MAT)\s*(?P<year>\d{4})", re.IGNORECASE)


@dataclass
class ParsedPeriod:
    label: str        # canonical string used as node id
    window_weeks: int | None = None
    end_date: str | None = None
    period_type: str | None = None  # "rolling" | "YTD" | "MAT" | "unknown"
    raw: str = ""


def parse_period(raw: str) -> ParsedPeriod | None:
    """Return a ParsedPeriod if raw looks like a period string, else None."""
    m = _PERIOD_RE.search(raw)
    if m:
        weeks = int(m.group("window"))
        end = m.group("end_date")
        label = f"{weeks}W_bis_{end.replace('/', '-')}"
        return ParsedPeriod(
            label=label,
            window_weeks=weeks,
            end_date=end,
            period_type="rolling",
            raw=raw,
        )

    m2 = _NAMED_RE.search(raw)
    if m2:
        ptype = m2.group("type").upper()
        year = m2.group("year")
        label = f"{ptype}_{year}"
        return ParsedPeriod(
            label=label,
            period_type=ptype,
            raw=raw,
        )

    return None


def extract_periods_from_columns(columns: list[str]) -> list[ParsedPeriod]:
    """Scan all column names and return unique parsed periods."""
    seen: dict[str, ParsedPeriod] = {}
    for col in columns:
        p = parse_period(col)
        if p and p.label not in seen:
            seen[p.label] = p
    return list(seen.values())

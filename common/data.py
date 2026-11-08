"""Shared loader/cleaner for the SBA FOIA 7(a) FY2010-FY2019 file.

Every cleaning decision is made here once so both assignments stay consistent.
"""

from pathlib import Path
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "foia-7a-fy2010-fy2019-asof-260331.csv"

DATE_COLS = ["approvaldate", "firstdisbursementdate", "paidinfulldate", "chargeoffdate"]


def load_raw(path: Path = RAW) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, dtype={"bankzip": str, "borrzip": str})
    df.columns = [c.strip().lower() for c in df.columns]
    for c in DATE_COLS:
        df[c] = pd.to_datetime(df[c], format="%m/%d/%Y", errors="coerce")
    df["loanstatus"] = df["loanstatus"].str.strip()
    df["bankname"] = df["bankname"].str.strip()
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Decisions:
    1. Drop exact duplicate rows.
    2. Drop CANCLD (cancelled before funding: no credit outcome).
    3. Drop COMMIT (not yet disbursed).
    """
    n0 = len(df)
    df = df.drop_duplicates().copy()
    df = df[~df["loanstatus"].isin(["CANCLD", "COMMIT"])].copy()
    df.attrs["rows_dropped"] = n0 - len(df)
    return df
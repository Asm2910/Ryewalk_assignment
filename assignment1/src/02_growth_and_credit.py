"""Step 2 - Growth (Q2) and credit quality (Q3) for the identified SBLCs.

Metric definitions
  Volume    : loans approved and gross approval $ per approval FY.
  Share     : lender / all 7(a) loans in the file (market share) and lender / SBLC pool.
  Gain/loss : market share FY2017-19 vs FY2010-12 (3-yr windows smooth lumpy years).
  Bad loan  : status CHGOFF. Distress = CHGOFF + LIQUID + PURCH(NOT C/O) + CLSLN (sensitivity).
  Fair credit:
    (a) cumulative charge-off rate, compared with the all-7(a) rate for the same FY x size band
        (observed/expected) -> controls vintage age and ticket size;
    (b) 60-month charge-off rate (all vintages have >=6.5y of history by Mar-2026). 36m was tested
        and captured only ~1/4 of charge-offs (median time to charge-off is ~56 months);
    (c) charge-off $ / approved $ (severity).
Run: python assignment1/src/02_growth_and_credit.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.data import load_raw, clean

OUT = ROOT / "assignment1" / "outputs"
SBLC_TIERS = ["1-Confirmed (data + external)", "2-Likely (data only)"]


def main():
    df = clean(load_raw())
    cls = pd.read_csv(OUT / "sblc_classification.csv")
    sblcs = cls[cls.tier.isin(SBLC_TIERS)].bankname.tolist()
    df["is_sblc"] = df.bankname.isin(sblcs)
    df["bad"] = (df.loanstatus == "CHGOFF").astype(int)
    df["bad60"] = ((df.loanstatus == "CHGOFF") &
                   ((df.chargeoffdate - df.approvaldate).dt.days <= 60 * 30.4)).astype(int)
    df["distress"] = df.loanstatus.isin(["CHGOFF", "LIQUID", "PURCH(NOT C/O)", "CLSLN"]).astype(int)
    df["size_band"] = pd.cut(df.grossapproval, [0, 50e3, 150e3, 350e3, 1e6, 5.1e6],
                             labels=["<=50k", "50-150k", "150-350k", "350k-1M", ">1M"])

    # ---- Q2 volume & share
    yr = df[df.is_sblc].groupby(["bankname", "approvalfy"]).agg(loans=("bad", "size"), gross=("grossapproval", "sum")).reset_index()
    tot = df.groupby("approvalfy").agg(mkt_loans=("bad", "size"), mkt_gross=("grossapproval", "sum"))
    yr = yr.merge(tot, left_on="approvalfy", right_index=True)
    yr["share_loans_bps"] = yr.loans / yr.mkt_loans * 1e4
    yr["share_gross_bps"] = yr.gross / yr.mkt_gross * 1e4
    yr.to_csv(OUT / "sblc_by_year.csv", index=False)

    def window(lo, hi):
        w = df[df.approvalfy.between(lo, hi)]
        return (w[w.is_sblc].groupby("bankname").grossapproval.agg(["size", "sum"])
                .rename(columns={"size": f"loans_{lo}_{hi}", "sum": f"gross_{lo}_{hi}"}), w.grossapproval.sum(), len(w))
    a, a_g, a_n = window(2010, 2012); b, b_g, b_n = window(2017, 2019)
    g = a.join(b, how="outer").fillna(0)
    g["mkt_share_gross_early_bps"] = g.gross_2010_2012 / a_g * 1e4
    g["mkt_share_gross_late_bps"] = g.gross_2017_2019 / b_g * 1e4
    g["share_change_bps"] = g.mkt_share_gross_late_bps - g.mkt_share_gross_early_bps
    g["gross_cagr_early_to_late"] = (g.gross_2017_2019 / g.gross_2010_2012.replace(0, np.nan)) ** (1 / 7) - 1
    g["pool_share_late"] = g.gross_2017_2019 / g.gross_2017_2019.sum()
    g = g.sort_values("gross_2017_2019", ascending=False)
    g.to_csv(OUT / "sblc_growth.csv"); print(g.round(2).to_string())

    # ---- Q3 credit
    exp_rate = df.groupby(["approvalfy", "size_band"], observed=True).bad60.mean().rename("exp60")
    df = df.join(exp_rate, on=["approvalfy", "size_band"])
    df = df.join(df.groupby(["approvalfy", "size_band"], observed=True).bad.mean().rename("exp_ever"),
                 on=["approvalfy", "size_band"])
    s = df[df.is_sblc]
    c = s.groupby("bankname").agg(loans=("bad", "size"), gross_usd_m=("grossapproval", lambda x: x.sum() / 1e6),
                                  co_rate=("bad", "mean"), co60_rate=("bad60", "mean"), distress_rate=("distress", "mean"),
                                  exp60=("exp60", "mean"), exp_ever=("exp_ever", "mean"),
                                  co_amt=("grosschargeoffamount", "sum"), gross=("grossapproval", "sum"))
    c["co_dollar_rate"] = c.co_amt / c.gross
    c["obs_to_exp_60m"] = c.co60_rate / c.exp60
    c["obs_to_exp_ever"] = c.co_rate / c.exp_ever
    z = 1.96; p = c.co60_rate; n = c.loans          # Wilson CI on the 60m rate
    c["ci_lo"] = (p + z**2/(2*n) - z*np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / (1 + z**2/n)
    c["ci_hi"] = (p + z**2/(2*n) + z*np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / (1 + z**2/n)
    mk = df.groupby("is_sblc").agg(co60=("bad60", "mean"), co=("bad", "mean"))
    print("\nMarket benchmark:\n", mk.round(4)); print("Overall 7(a): 60m CO", round(df.bad60.mean(), 4))
    c = c.sort_values("gross_usd_m", ascending=False); c.to_csv(OUT / "sblc_credit.csv"); print(c.round(3).to_string())

    # growth vs credit: was growth bought with credit quality?
    lv = (s.assign(period=np.where(s.approvalfy <= 2014, "FY10-14", "FY15-19"))
            .groupby(["bankname", "period"])
            .agg(loans=("bad", "size"), gross=("grossapproval", "sum"), co60=("bad60", "mean"), exp60=("exp60", "mean"))
            .reset_index())
    lv["obs_to_exp"] = lv.co60 / lv.exp60
    lv.to_csv(OUT / "sblc_period_credit.csv", index=False); print("\n", lv.round(3).to_string())
    df.groupby("approvalfy").bad60.mean().to_csv(OUT / "market_co60_by_fy.csv")


if __name__ == "__main__":
    main()
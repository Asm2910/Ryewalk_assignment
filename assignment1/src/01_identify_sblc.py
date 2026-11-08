"""Step 1 - Identify SBLCs.

The FOIA file has no lender-type field. Evidence used (all derivable from the file):
  E1 GATE   : no FDIC cert and no NCUA charter -> not a regulated depository
  E2 name   : SBLC / small-business-lending vocabulary
  E3 footprint: number of distinct borrower states (SBLCs are national, CDFIs/CDCs regional)
  E4 program: share of loans under the Community Advantage pilot (mission lenders, not SBLCs)
  E5 product: commercial 7(a) profile (PLP / larger tickets)
  E6 scale  : enough loans to judge
  E7 external: named as an SBA-licensed SBLC in public sources
  NEG name  : CDC / economic development / BIDCO / farm credit / govt -> penalty

Output: outputs/lender_features.csv, outputs/sblc_classification.csv
Run:  python assignment1/src/01_identify_sblc.py
"""
import re, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.data import load_raw, clean

OUT = ROOT / "assignment1" / "outputs"; OUT.mkdir(exist_ok=True)

NAME_POS = re.compile(r"SBLC|small business (lend|loan|financ)|SBA lending|business loan", re.I)
NAME_NEG = re.compile(r"develop|economic|BIDCO|farm|AgSouth|AgChoice|AgCredit|Ag Credit|ACA\b|FLCA|"
                      r"certified|community|initiative|foundation|fund\b|FDIC|Federal Deposit|National Credit Union|"
                      r"\bSBA\b|credit union|\bFCU\b|\bCU\b|housing|enterprise|entrepreneur|trade finance|"
                      r"CDC\b|CDFI|\bcorp(oration)? of\b", re.I)
NAME_BANK = re.compile(r"\bbank\b|trust company|savings|national association", re.I)

# External corroboration (E7): lenders named as SBA-licensed SBLCs in public sources, matched by
# name fragment. Sources: SBA/CU Times 2003 SBLC roster (Newtek, Loan Source, BLC ...),
# Coleman Report 2015 (Readycap / CenterStone SBLC licences), Funderintel roster of long-time
# holders (Readycap, Harvest, Lendistry, VelocitySBA, Exos, CenterStone, First Western SBLC, CRF,
# Grow America Fund). Newtek later converted to a bank (post-2019, outside the data window).
# VERIFY against SBA's official roster before presenting.
EXTERNAL = ["newtek small business", "readycap", "harvest small business", "lendistry", "velocitysba",
            "exos small business", "centerstone sba", "first western sblc", "crf small business",
            "grow america fund", "loan source incorporated", "capitalspring sblc", "business loan center"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("bankname")
    f = pd.DataFrame({
        "loans": g.size(),
        "gross_usd_m": g.grossapproval.sum() / 1e6,
        "median_loan": g.grossapproval.median(),
        "n_borrower_states": g.borrstate.nunique(),
        "has_fdic": g.bankfdicnumber.apply(lambda s: s.notna().any()),
        "has_ncua": g.bankncuanumber.apply(lambda s: s.notna().any()),
        "ca_share": g.processingmethod.apply(lambda s: s.eq("Community Advantage Initiative").mean()),
        "plp_share": g.processingmethod.apply(lambda s: s.isin(["Preferred Lenders Program", "Preferred Lenders with EWCP"]).mean()),
        "first_fy": g.approvalfy.min(), "last_fy": g.approvalfy.max(),
        "n_locationids": g.locationid.nunique(),
    })
    return f.reset_index()


def score(r) -> tuple[int, list[str]]:
    why, s = [], 0
    if r.has_fdic or r.has_ncua or re.search(r"Federal Deposit|National Credit Union|\bSBA CLSC", r.bankname, re.I):
        return -99, ["regulated depository / government entity -> excluded"]
    why.append("E1 no FDIC/NCUA id")
    if re.search(r"SBLC", r.bankname, re.I): s += 3; why.append("E2 name contains SBLC (+3)")
    elif NAME_POS.search(r.bankname): s += 2; why.append("E2 name has small-business-lending vocabulary (+2)")
    if NAME_NEG.search(r.bankname): s -= 2; why.append("NEG name looks CDC/dev/farm/govt/credit-union (-2)")
    if NAME_BANK.search(r.bankname): s -= 1; why.append("NEG name looks like a bank (-1)")
    if r.n_borrower_states >= 10: s += 2; why.append(f"E3 national footprint, {r.n_borrower_states} states (+2)")
    elif r.n_borrower_states >= 5: s += 1; why.append(f"E3 multi-state, {r.n_borrower_states} states (+1)")
    else: why.append(f"E3 regional, {r.n_borrower_states} states (0)")
    if r.ca_share >= 0.5: s -= 2; why.append(f"E4 {r.ca_share:.0%} Community Advantage = mission lender (-2)")
    else: s += 1; why.append("E4 not Community-Advantage-based (+1)")
    if r.plp_share >= 0.5 or r.median_loan >= 150_000: s += 1; why.append("E5 commercial PLP/large-ticket profile (+1)")
    if r.loans >= 100: s += 1; why.append(f"E6 scale {r.loans} loans (+1)")
    return s, why


def main():
    df = clean(load_raw())
    f = build_features(df)
    res = f.apply(score, axis=1, result_type="expand"); f["score"], f["evidence"] = res[0], res[1].map("; ".join)
    f["external_match"] = f.bankname.str.lower().map(lambda n: any(k in n for k in EXTERNAL))

    def tier(r):
        if r.score == -99: return "Not SBLC (depository)"
        if r.external_match and r.score >= 3: return "1-Confirmed (data + external)"
        if r.score >= 5: return "2-Likely (data only)"
        if r.score >= 3: return "3-Possible (weak)"
        return "4-Not SBLC"
    f["tier"] = f.apply(tier, axis=1)

    f.sort_values(["score", "loans"], ascending=False).to_csv(OUT / "lender_features.csv", index=False)
    c = f[f.score > -99].sort_values(["score", "loans"], ascending=False)
    c.to_csv(OUT / "sblc_classification.csv", index=False)
    print(c[["bankname", "loans", "n_borrower_states", "ca_share", "plp_share", "median_loan", "score", "tier"]].head(40).round(2).to_string())


if __name__ == "__main__":
    main()
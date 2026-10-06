"""Load the client data pack and clean raw claim records.

The same `clean_claims` is used for the training batch and for single records sent to the API,
so a claim is cleaned identically wherever it comes from.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AUTO_APPROVE_LIMIT_INR, POLICY_CHANGE

# Raw claim columns, as in test_unlabelled.csv and the API request
YES_NO = {"Y": 1, "N": 0}

CLAIM_COLUMNS = ["claim_id", "submitted_at", "partner_id", "sku", "product_serial", "days_since_purchase",
                 "claim_amount_inr", "photo_attached", "partner_inspected", "claim_description",
                 "inspector_note", "customer_prior_claims", "source"]


@dataclass(frozen=True)
class DataPack:
    train: pd.DataFrame
    test: pd.DataFrame
    partners: pd.DataFrame
    products: pd.DataFrame
    sample_submission: pd.DataFrame


def find_file(data_dir: Path, suffix: str) -> Path:
    """File names in the pack carry a UUID prefix (e.g. '18ec...-train.csv'), so match on the suffix."""
    matches = list(Path(data_dir).glob(f"*{suffix}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one file ending in '{suffix}' in {data_dir}, found {len(matches)}.")
    return matches[0]


def load_pack(data_dir: Path) -> DataPack:
    """Read the five CSVs of the client data pack, with dates parsed."""
    read = lambda suffix, **kw: pd.read_csv(find_file(data_dir, suffix), **kw)
    return DataPack(
        train=read("train.csv", parse_dates=["submitted_at"]),
        test=read("test_unlabelled.csv", parse_dates=["submitted_at"]),
        partners=read("partners.csv", parse_dates=["onboarded_date"]),
        products=read("products.csv"),
        sample_submission=read("sample_submission.csv"),
    )


def _require(ok: pd.Series, problem: str, values: pd.Series) -> None:
    """Fail with the offending values named, instead of letting NaNs reach the model."""
    if not ok.all():
        raise ValueError(f"{problem}: {sorted(values[~ok].astype(str).unique())[:5]}")


def serial_format(serial: str) -> str:
    """How the partner typed the serial. Kept as a feature: typing habits differ by outlet."""
    if serial.startswith(" "):
        return "leading_space"
    if "-" in serial:
        return "hyphen"
    if serial[:2].islower():
        return "lowercase"
    return "clean"


def clean_claims(claims: pd.DataFrame, partners: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Normalise raw claims and join partner and product reference data."""
    out = claims.copy()
    out["submitted_at"] = pd.to_datetime(out["submitted_at"])

    raw_serial = out["product_serial"].astype(str)
    out["serial_format"] = raw_serial.map(serial_format)
    out["product_serial"] = raw_serial.str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)

    for col in ["photo_attached", "partner_inspected"]:
        _require(out[col].isin(YES_NO), f"{col} must be Y or N", out[col])
        out[col] = out[col].map(YES_NO)

    # Keep only the standard fault phrase. A few partner-entered descriptions carry appended free text
    # (including text addressed to automated analysis tools), which is not part of the fault.
    out["claim_description"] = (out["claim_description"].str.lower()
                                .str.split(r"[.;]", regex=True).str[0].str.strip()
                                .replace({"display blank": "display not working"}))

    out["has_inspector_note"] = out["inspector_note"].notna().astype(int)
    out["inspector_note"] = out["inspector_note"].fillna("no note")

    # Policy regime: from POLICY_CHANGE, claims under the limit skip inspection
    out["after_may_change"] = (out["submitted_at"] >= POLICY_CHANGE).astype(int)
    out["small_claim"] = (out["claim_amount_inr"] < AUTO_APPROVE_LIMIT_INR).astype(int)

    _require(out["partner_id"].isin(partners["partner_id"]), "partner_id not in partners.csv", out["partner_id"])
    _require(out["sku"].isin(products["sku"]), "sku not in products.csv", out["sku"])
    return (out.merge(partners, on="partner_id", how="left", validate="many_to_one")
               .merge(products, on="sku", how="left", validate="many_to_one"))


def prepare_history(pack: DataPack) -> pd.DataFrame:
    """Every known claim, cleaned: train (including open investigations) plus test.

    Re-submitted claims appear more than once in train with a later `submitted_at`; the first
    submission is when the claim was made, so it is the one kept. Open investigations stay in
    (with a blank label) because they still count towards a partner's claim volume.
    """
    train = pack.train.sort_values("submitted_at").drop_duplicates("claim_id", keep="first")
    claims = pd.concat([train.assign(dataset="train"), pack.test.assign(dataset="test", is_fraud=np.nan)],
                       ignore_index=True)
    return clean_claims(claims, pack.partners, pack.products)

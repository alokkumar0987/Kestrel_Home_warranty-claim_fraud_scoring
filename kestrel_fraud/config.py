"""Paths, business rules and modelling choices, kept in one place.

Business rules cite the client's ops policy (Data/*-ops-policy.pdf). Modelling choices cite the
analysis notebook (notebooks/Kestrel_Home.ipynb) where each one was tested.
"""
import os
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- paths
# Environment variables let the service run from a different working directory or data location.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("KESTREL_DATA_DIR", ROOT / "Data"))
ARTIFACT_PATH = Path(os.environ.get("KESTREL_ARTIFACT", ROOT / "artifacts" / "model.joblib"))
OUTPUT_DIR = ROOT / "outputs"
EVIDENCE_DIR = ROOT / "evidence"

# --------------------------------------------------------------------------- business rules
POLICY_CHANGE = pd.Timestamp("2026-05-01")   # policy §5: claims under the limit auto-approved from this date
AUTO_APPROVE_LIMIT_INR = 2_000               # policy §5
REVIEWS_PER_MONTH = 40                       # policy §5: investigation desk capacity
GOODWILL_COST_INR = 380                      # policy §4: cost of holding a genuine claim
CONTACT_COST_INR = 260                       # policy §4: blended cost of a service contact
LOCAL_TZ = "Asia/Kolkata"                    # README.txt: submitted_at is IST

# --------------------------------------------------------------------------- modelling choices
NEAR_LIMIT_FLOOR_INR = 1_800   # "just under the limit" band: [1,800, 2,000), notebook §6.4
NEW_PARTNER_DAYS = 365         # first-year partner, notebook §5 and the hypothesis notebook
DAYS_PER_MONTH = 30.44         # converts warranty months to days
LABEL_LAG_DAYS = 30            # an investigation outcome is assumed known 30 days after the claim, notebook §6
POST_MAY_WEIGHT = 3            # training weight of post-policy claims, chosen in notebook §7.3
PRIOR_FRAUD_RATE = 0.0127      # overall fraud rate in labelled training claims
PRIOR_WEIGHT = 20              # pseudo-claims used to smooth a partner's fraud rate towards PRIOR_FRAUD_RATE
LOGISTIC_C = 0.1               # regularisation strength, notebook §7

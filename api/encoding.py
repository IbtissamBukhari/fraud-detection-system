"""Translate raw categorical values into the codes the model was trained on.

The encoding cell in notebook 04 was executed twice during training. The first
run label-encoded the raw strings (for example "visa" -> 4); the second run
re-encoded those integer codes as strings ("4" -> some new index). The model was
trained on the result of the second pass, and only the second-pass encoders were
saved to models/encoders.pkl.

Those saved encoders therefore know values like "0", "1", "10" but not "visa" or
"gmail.com". Passing raw values straight to them silently turned every input
into "unknown", so the user's choices had no effect on the prediction.

This module reproduces the first pass from the known category lists (the same
lists the first LabelEncoder saw, sorted alphabetically with "unknown" added)
and then applies the saved second-pass encoder, giving exactly the codes the
model expects.
"""

from typing import Dict, List

# Category values present in the IEEE-CIS training data. Missing values were
# filled with "unknown" in notebook 01, so "unknown" is a real training value.
RAW_CATEGORIES: Dict[str, List[str]] = {
    "ProductCD": ["C", "H", "R", "S", "W"],
    "card4": ["american express", "discover", "mastercard", "unknown", "visa"],
    "card6": ["charge card", "credit", "debit", "debit or credit", "unknown"],
    "P_emaildomain": [
        "aim.com", "anonymous.com", "aol.com", "att.net", "bellsouth.net",
        "cableone.net", "centurylink.net", "cfl.rr.com", "charter.net",
        "comcast.net", "cox.net", "earthlink.net", "embarqmail.com",
        "frontier.com", "frontiernet.net", "gmail", "gmail.com", "gmx.de",
        "hotmail.co.uk", "hotmail.com", "hotmail.de", "hotmail.es",
        "hotmail.fr", "icloud.com", "juno.com", "live.com", "live.com.mx",
        "live.fr", "mac.com", "mail.com", "me.com", "msn.com", "netzero.com",
        "netzero.net", "optonline.net", "outlook.com", "outlook.es",
        "prodigy.net.mx", "protonmail.com", "ptd.net", "q.com",
        "roadrunner.com", "rocketmail.com", "sbcglobal.net", "sc.rr.com",
        "servicios-ta.com", "suddenlink.net", "twc.com", "unknown",
        "verizon.net", "web.de", "windstream.net", "yahoo.co.jp",
        "yahoo.co.uk", "yahoo.com", "yahoo.com.mx", "yahoo.de", "yahoo.es",
        "yahoo.fr", "ymail.com",
    ],
}

# Feature-engineering rules from notebook 03
HIGH_RISK_EMAIL_DOMAINS = {"mail.com", "outlook.com", "live.com.mx", "hotmail.com"}


def first_pass_classes(field: str) -> List[str]:
    """Classes of the first (unsaved) LabelEncoder: sorted values plus 'unknown'."""
    return sorted(set(RAW_CATEGORIES[field]) | {"unknown"})


def normalise(field: str, value: str) -> str:
    """Lower-case and trim free-text values; map anything unseen to 'unknown'."""
    value = str(value).strip()
    if field != "ProductCD":
        value = value.lower()
    return value if value in RAW_CATEGORIES[field] else "unknown"


def encode(field: str, value: str, encoders: dict) -> int:
    """Return the model-ready code for a raw categorical value."""
    raw = normalise(field, value)
    code = str(first_pass_classes(field).index(raw))
    if code not in set(encoders[field].classes_):
        # e.g. ProductCD "unknown": never seen in training, so use the
        # second-pass "unknown" class the notebook added as a fallback
        code = "unknown"
    return int(encoders[field].transform([code])[0])


def high_risk_email(domain: str) -> int:
    return int(normalise("P_emaildomain", domain) in HIGH_RISK_EMAIL_DOMAINS)


def high_risk_card(card4: str, card6: str) -> int:
    return int(normalise("card4", card4) == "discover" and normalise("card6", card6) == "credit")

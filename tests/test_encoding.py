"""Checks that raw categorical values reach the model as distinct, correct codes."""
import joblib
import pytest

from api.encoding import (
    RAW_CATEGORIES,
    encode,
    first_pass_classes,
    high_risk_card,
    high_risk_email,
)
from tests.conftest import ROOT

ENCODERS = joblib.load(ROOT / "models" / "encoders.pkl")


@pytest.mark.parametrize("field", list(RAW_CATEGORIES))
def test_category_lists_match_saved_encoders(field):
    # Every first-pass code seen in training must exist in the saved encoder
    saved = set(ENCODERS[field].classes_) - {"unknown"}
    first_pass = {str(i) for i in range(len(first_pass_classes(field)))}
    assert saved <= first_pass
    # Only codes that never occurred in training (e.g. ProductCD "unknown") may be absent
    for code in first_pass - saved:
        assert first_pass_classes(field)[int(code)] == "unknown"


@pytest.mark.parametrize("field", ["ProductCD", "card4", "card6", "P_emaildomain"])
def test_known_values_get_distinct_codes(field):
    values = [v for v in RAW_CATEGORIES[field] if v != "unknown"]
    codes = [encode(field, v, ENCODERS) for v in values]
    assert len(set(codes)) == len(values)


def test_unseen_values_fall_back_to_unknown():
    assert encode("P_emaildomain", "not-a-real-domain.xyz", ENCODERS) == encode(
        "P_emaildomain", "unknown", ENCODERS
    )


def test_input_is_normalised():
    assert encode("card4", "  VISA ", ENCODERS) == encode("card4", "visa", ENCODERS)
    assert encode("P_emaildomain", "Gmail.com", ENCODERS) == encode("P_emaildomain", "gmail.com", ENCODERS)


def test_engineered_flags():
    assert high_risk_email("mail.com") == 1
    assert high_risk_email("gmail.com") == 0
    assert high_risk_card("discover", "credit") == 1
    assert high_risk_card("visa", "credit") == 0

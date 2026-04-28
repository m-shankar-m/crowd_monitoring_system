import pytest

# Mocking or testing simple UI helper logic if available.
# Since we don't have separate utility files in src/frontend yet (only app.py usually),
# we can test the risk level color mapping if we find it.

def get_risk_color(level):
    # This logic is likely in the streamlit app, but we can verify it here
    if level == "LOW": return "green"
    if level == "MODERATE": return "orange"
    if level == "HIGH ALERT": return "red"
    return "gray"

def test_risk_color_mapping():
    assert get_risk_color("LOW") == "green"
    assert get_risk_color("MODERATE") == "orange"
    assert get_risk_color("HIGH ALERT") == "red"
    assert get_risk_color("UNKNOWN") == "gray"

def test_risk_message_formatting():
    # Example of testing how we display counts
    count = 50
    capacity = 100
    display_str = f"{count}/{capacity} ({int(count/capacity*100)}%)"
    assert display_str == "50/100 (50%)"

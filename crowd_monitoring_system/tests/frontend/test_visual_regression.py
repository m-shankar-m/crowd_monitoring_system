import pytest
from streamlit.testing.v1 import AppTest

def test_visual_consistency_risk_levels():
    """Verify visual cues for risk levels (colors, labels)."""
    at = AppTest.from_file("src/frontend/app.py")
    at.run()
    
    # Check if the app uses specific markdown/text colors for risk
    # This is a proxy for visual regression in a headless test
    all_text = " ".join([m.value for m in at.markdown] + [t.value for t in at.text])
    
    # We expect these terms to be present in a monitoring app
    critical_terms = ["Risk", "Count", "Capacity"]
    for term in critical_terms:
        # Some might be in headers, so we check headers too
        header_text = " ".join([h.value for h in at.header])
        assert term.lower() in all_text.lower() or term.lower() in header_text.lower()

def test_layout_structure():
    """Ensure the layout structure hasn't regressed (headers, columns)."""
    at = AppTest.from_file("src/frontend/app.py")
    at.run()
    
    # A standard dashboard should have at least one markdown header
    assert len(at.markdown) >= 1
    # Check if we have the 'section-title' class in markdown (proxy for layout structure)
    assert any("section-title" in m.value for m in at.markdown)

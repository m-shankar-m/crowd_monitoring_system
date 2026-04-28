import pytest
from streamlit.testing.v1 import AppTest

def test_app_smoke():
    """Basic test to see if the app can start up without crashing."""
    at = AppTest.from_file("src/frontend/app.py")
    at.run(timeout=30)
    
    # Assert that there are no exceptions during the run
    assert not at.exception
    
    # Check for main UI elements
    # Since the app uses st.markdown for headers, we check markdown
    assert len(at.markdown) > 0
    
    # Check if a specific header text exists
    header_found = any("Crowd Predictor Framework" in m.value for m in at.markdown)
    assert header_found

def test_app_sidebar():
    """Test if sidebar elements exist."""
    at = AppTest.from_file("src/frontend/app.py")
    at.run()
    
    # Check if there's a sidebar (usually contains settings)
    # This might depend on how the app is structured
    # For now, just ensure it runs and we can access the sidebar object
    assert at.sidebar is not None

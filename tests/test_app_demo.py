"""End-to-end check of the Streamlit demo with the real trained models (skipped if none exist).

Runs the app headless with streamlit's AppTest: loads the models, analyses one review, and checks
that the hybrid probability, the autoencoder error and the LIME words are shown without errors.
Slow (loads SBERT), so it is marked `slow`: run with  python -m pytest -m slow
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HAS_MODELS = (ROOT / "checkpoints/hybrid__salminen__seed42/meta.json").exists() or (ROOT / "app/model/hybrid/meta.json").exists()


@pytest.mark.slow
@pytest.mark.skipif(not HAS_MODELS, reason="no trained hybrid model")
def test_single_review_analysis():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=600)
    at.run()
    assert not at.exception, at.exception
    at.text_area[0].input("Love this! Well made, sturdy, and very comfortable. I love it! Very pretty")
    at.button[0].click()
    at.slider[0].set_value(100)          # fewer LIME samples keeps the test fast on CPU
    at.run()
    assert not at.exception, at.exception
    labels = [m.label for m in at.metric]
    assert "P(fake) - hybrid" in labels and "AE reconstruction error" in labels
    assert at.dataframe, "LIME word table missing"

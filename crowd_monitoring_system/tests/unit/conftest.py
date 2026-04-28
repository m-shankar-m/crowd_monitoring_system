"""
conftest.py  —  Project-wide pytest configuration
==================================================
Ensures ALL tests run in isolation from the real production dataset.

How it works
------------
1.  `autouse=True` means every test automatically gets the `isolate_dataset` fixture
    applied — you don't need to add it to each test manually.

2.  Before the test runs, the real crowd_data.csv is backed up to a temp file.

3.  After the test finishes (pass OR fail), the backup is restored and any
    test-generated rows are wiped out.

4.  If the real file didn't exist yet, it is simply deleted after the test.
"""

import pytest
import os
import shutil
import tempfile

REAL_CSV = "data/crowd_data.csv"


@pytest.fixture(autouse=True)
def isolate_dataset():
    """
    Backs up the real crowd_data.csv before each test and restores it after.
    Fake/test data written during the test is automatically discarded.
    """
    backup_path = None

    if os.path.exists(REAL_CSV):
        # Back up the real data to a temp file
        fd, backup_path = tempfile.mkstemp(suffix=".csv", prefix="crowd_backup_")
        os.close(fd)
        shutil.copy2(REAL_CSV, backup_path)

    yield  # ← test runs here

    # ── Teardown: restore real data ──────────────────────────────────────────
    if backup_path and os.path.exists(backup_path):
        shutil.copy2(backup_path, REAL_CSV)   # restore original
        os.remove(backup_path)                # delete temp backup
    elif os.path.exists(REAL_CSV):
        # Real file didn't exist before the test; remove what the test created
        os.remove(REAL_CSV)

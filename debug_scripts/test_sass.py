import pytest
from pathlib import Path
from tinygrad.helpers import getenv
test_dir = Path(__file__). parents[1] / "test"

fail_fast = getenv("FAIL_FAST", 0)
tests = [
  "test_uops.py",
  "test_dtype_alu.py",
  "test_dtype.py",
]
pytest.main((["-x"] if fail_fast else []) + [test_dir / t for t in tests])

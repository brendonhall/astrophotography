import os
import sys

# Make scripts/ importable (astrolib, pcc) from tests.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

"""
Intelligent Phishing & Malicious URL Classifier (Link Inspector)
Root entrypoint wrapper for Streamlit Community Cloud and local execution.
Delegates execution to dashboard/app.py.
"""
import os
import sys
from pathlib import Path

# Locate dashboard directory
BASE_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = BASE_DIR / "dashboard"

if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

# Ensure dashboard app script exists
app_script = DASHBOARD_DIR / "app.py"
if not app_script.exists():
    raise FileNotFoundError(f"Dashboard app script not found at {app_script}")

# Execute dashboard/app.py in the proper context
with open(app_script, "r", encoding="utf-8") as f:
    code = compile(f.read(), str(app_script), "exec")
    exec(code, globals())

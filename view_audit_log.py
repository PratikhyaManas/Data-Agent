"""
Quick audit log viewer. Run: python view_audit_log.py [n]
"""
import sys
import json
from utils.audit import read_recent

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    entries = read_recent(n)
    if not entries:
        print("No audit log entries yet. Run a few requests through main.py first.")
    for e in entries:
        print(json.dumps(e, indent=2))
        print("-" * 60)

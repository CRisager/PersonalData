import sys
from pathlib import Path
import json

# Ensure repository root is on sys.path so we can import main.py when running from scripts/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import build_insights_recordings


def main():
    out_path = Path("frontend") / "insights-data.json"
    recordings = build_insights_recordings()
    payload = {"ok": True, "recordings": recordings}
    out_path.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
    print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()

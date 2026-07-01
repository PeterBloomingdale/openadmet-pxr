"""
CLI for appending dated decision entries to the manuscript appendix.

Usage:
  python scripts/log_decision.py --topic "Butina threshold" \
      --decision "Used 0.4, not 0.3 or 0.5" \
      --rationale "Matches the threshold OpenADMET used to construct the test set"

This appends a structured entry to the Running Appendix section of
manuscript/pxr_challenge.qmd. Run this immediately when you make a
non-obvious decision — context is lost within hours under time pressure.

Why this matters:
Reviewers will ask "why Tanimoto 0.4?" and "why Huber delta=0.5?". The
answer is always in the appendix if you log decisions as you make them.
A retrospective write-up a week later will miss the reasoning.
"""

import argparse
from datetime import date
from pathlib import Path
import re


QMD_PATH = "manuscript/pxr_challenge.qmd"
APPENDIX_HEADER = "## Appendix A: Running Decision Log"


def append_decision_entry(
    topic: str,
    decision: str,
    rationale: str,
    outcome: str = "TBD — check back",
    learned: str = "",
) -> None:
    """Appends a decision entry to the QMD appendix section."""
    qmd_path = Path(QMD_PATH)
    if not qmd_path.exists():
        print(f"ERROR: QMD file not found at {QMD_PATH}. Has the manuscript been created?")
        return

    content = qmd_path.read_text()
    if APPENDIX_HEADER not in content:
        print(f"ERROR: Appendix header '{APPENDIX_HEADER}' not found in QMD. Check manuscript structure.")
        return

    today = date.today().strftime("%Y-%m-%d")
    entry = f"""
### {today}: {topic}

**Decision:** {decision}

**Rationale:** {rationale}

**Outcome:** {outcome}

**Learned:** {learned if learned else "*(fill in retrospectively)*"}

---
"""
    # Append just before the end of the file
    updated = content + entry
    qmd_path.write_text(updated)
    print(f"Logged decision: '{topic}' ({today})")
    print(f"Entry appended to {QMD_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Log a decision to the manuscript appendix")
    parser.add_argument("--topic", required=True, help="Short topic name (e.g. 'Butina threshold')")
    parser.add_argument("--decision", required=True, help="What was decided")
    parser.add_argument("--rationale", required=True, help="Why this decision was made")
    parser.add_argument("--outcome", default="TBD — check back", help="What happened (add later)")
    parser.add_argument("--learned", default="", help="Key concept learned")
    args = parser.parse_args()

    append_decision_entry(
        topic=args.topic,
        decision=args.decision,
        rationale=args.rationale,
        outcome=args.outcome,
        learned=args.learned,
    )


if __name__ == "__main__":
    main()

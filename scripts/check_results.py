#!/usr/bin/env python3
"""Dogfood check for committed results: the gate must agree with the stored decisions.

- With a committed results/report.json, the report is loaded, the policy in configs/gate.yaml is
  applied afresh, and the stored decisions must match. The verdicts are printed.
- Without one, the check runs on tests/fixtures/report_all_pass.json. That file is SYNTHETIC and
  labelled so, and the output says so.

The script fails (exit 1) if the report is invalid or the stored decisions disagree with the
policy. It does not fail when a candidate FAILs the gate. A failing rung is a result, and this
repository's CI must not turn a measured result into a red build. Downstream users who want a
red build on a failing rung use the composite action (action.yml).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "results" / "report.json"
SYNTHETIC_FIXTURE = REPO / "tests" / "fixtures" / "report_all_pass.json"
POLICY = REPO / "configs" / "gate.yaml"

sys.path.insert(0, str(REPO / "src"))

from quantgate.gate import gate_label, gate_report, load_policy, recommend  # noqa: E402
from quantgate.report import load_report  # noqa: E402


def main() -> int:
    if REPORT.is_file():
        path = REPORT
        label = "committed results"
    else:
        path = SYNTHETIC_FIXTURE
        label = "SYNTHETIC fixture (no committed results yet)"
    report = load_report(path)
    if label.startswith("SYNTHETIC") and report.env_info.get("synthetic") is not True:
        print("check_results: the fixture must be labelled synthetic", file=sys.stderr)
        return 1
    policy = load_policy(POLICY)
    decisions = gate_report(report, policy)
    recommended = recommend(decisions, report.variants)
    stored = {
        (d.variant, d.passed, tuple(d.failed_rules), d.hidden_danger) for d in report.decisions
    }
    fresh = {(d.variant, d.passed, tuple(d.failed_rules), d.hidden_danger) for d in decisions}
    print(f"Gate check on {label}: {path.relative_to(REPO)}")
    for d in decisions:
        print(f"  {d.variant}: {gate_label(d)}")
    print(f"  recommended: {recommended or 'none'}")
    if stored != fresh:
        print(
            "check_results: stored decisions differ from configs/gate.yaml. Re-run "
            "`quantgate eval` or `quantgate report` so the report matches the policy.",
            file=sys.stderr,
        )
        return 1
    print("stored decisions agree with the policy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

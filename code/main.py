"""
main.py — Entry point for the HackerRank Orchestrate support triage agent.

Usage:
    python main.py [--input PATH] [--output PATH] [--sample] [--verbose]

Options:
    --input   PATH   Input CSV  (default: ../support_tickets/support_tickets.csv)
    --output  PATH   Output CSV (default: ../support_tickets/output.csv)
    --sample         Run on sample_support_tickets.csv instead (for testing)
    --verbose        Print each ticket's result to stdout
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time

# Ensure code/ is on the path for sibling imports
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from agent import process_ticket

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).parent.parent
TICKETS_DIR = REPO_ROOT / "support_tickets"

DEFAULT_INPUT = TICKETS_DIR / "support_tickets.csv"
SAMPLE_INPUT = TICKETS_DIR / "sample_support_tickets.csv"
DEFAULT_OUTPUT = TICKETS_DIR / "output.csv"

OUTPUT_COLUMNS = [
    "Issue", "Subject", "Company",
    "status", "product_area", "response", "justification", "request_type",
]


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _read_tickets(path: pathlib.Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_output(rows: list[dict], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Support triage agent (offline, no API key needed)")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--sample", action="store_true", help="Use sample tickets")
    parser.add_argument("--verbose", action="store_true", help="Print results to stdout")
    args = parser.parse_args()

    input_path = pathlib.Path(args.input) if args.input else (SAMPLE_INPUT if args.sample else DEFAULT_INPUT)
    output_path = pathlib.Path(args.output)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    tickets = _read_tickets(input_path)
    print(f"\nProcessing {len(tickets)} tickets from {input_path.name} …\n", flush=True)

    output_rows: list[dict] = []
    t_total = time.time()

    for i, ticket in enumerate(tickets, 1):
        issue   = ticket.get("Issue", "")
        subject = ticket.get("Subject", "")
        company = ticket.get("Company", "")

        label = (subject or issue)[:55]
        print(f"[{i:02d}/{len(tickets)}] {label!r}", flush=True)

        result = process_ticket(issue, subject, company)

        status_icon = "↑ ESCALATED" if result["status"] == "escalated" else "✓ replied"
        print(f"        {status_icon}  |  area={result['product_area']}  |  type={result['request_type']}", flush=True)

        if args.verbose:
            print(f"        response    : {result['response'][:100]} …")
            print(f"        justification: {result['justification'][:100]} …")

        output_rows.append({
            "Issue": issue,
            "Subject": subject,
            "Company": company,
            **result,
        })

    _write_output(output_rows, output_path)
    elapsed = time.time() - t_total
    print(f"\n✓ Done in {elapsed:.1f}s — output written to {output_path}\n", flush=True)


if __name__ == "__main__":
    main()

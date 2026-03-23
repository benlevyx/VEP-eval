"""Helpers for constructing timestamped run-name subdirectories."""

from datetime import datetime
from pathlib import Path


def build_run_name(name: str | None, no_timestamp: bool) -> str:
    """
    Return the final run-name string.

    Rules:
      - no_timestamp=False, name=None  → "20260322_143052"
      - no_timestamp=False, name="foo" → "20260322_143052_foo"
      - no_timestamp=True,  name=None  → "run"  (bare fallback)
      - no_timestamp=True,  name="foo" → "foo"
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if no_timestamp:
        return name if name else "run"

    return f"{ts}_{name}" if name else ts


def resolve_output_dir(base_dir: str | Path, run_name: str) -> Path:
    """Return *base_dir* / *run_name*, creating it if necessary."""
    out = Path(base_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def add_run_name_args(parser) -> None:
    """Add --run-name and --no-timestamp arguments to *parser* in-place."""
    parser.add_argument(
        "--run-name",
        default=None,
        metavar="NAME",
        help=(
            "Label for this run. Outputs are saved under <output-dir>/<run-name>. "
            "A timestamp is prepended by default (disable with --no-timestamp)."
        ),
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Do not prepend a timestamp to the run-name.",
    )

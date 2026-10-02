"""Compatibility filename for the phase-1 configuration CLI.

Use --help for the new shared arguments. Historical experiment snapshots remain
under experiments/ and can be reproduced with their original code revision.
"""
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None):
    from src.evaluate_pairwise import main as entrypoint
    arguments = list(sys.argv[1:] if argv is None else argv)
    return entrypoint(arguments)


if __name__ == "__main__":
    main()

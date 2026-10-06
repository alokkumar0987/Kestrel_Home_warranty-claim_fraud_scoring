"""Copy the working notebooks into notebooks/ with all outputs removed.

    python tools/export_notebooks.py [--source .claude/Experiment]

The working copies keep their outputs locally. Outputs show client data (claim rows, partner IDs),
which ops policy §10 forbids publishing, so only output-free copies belong in the repository.
"""
import argparse
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent


def strip_outputs(source: Path, target: Path) -> int:
    nb = nbformat.read(source, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs, cell.execution_count = [], None
        cell.metadata.pop("execution", None)   # run timestamps: noise in diffs
    target.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, target)
    return len(nb.cells)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=ROOT / ".claude" / "Experiment")
    parser.add_argument("--target", type=Path, default=ROOT / "notebooks")
    args = parser.parse_args()
    notebooks = sorted(args.source.glob("*.ipynb"))
    if not notebooks:
        raise SystemExit(f"No notebooks found in {args.source}")
    for nb_path in notebooks:
        cells = strip_outputs(nb_path, args.target / nb_path.name)
        print(f"{nb_path.name}: {cells} cells -> {args.target / nb_path.name} (outputs removed)")


if __name__ == "__main__":
    main()

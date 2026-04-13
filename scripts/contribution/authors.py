"""Script for rendering contribution pages."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).parents[2]


def get_raw_contributors(pages_dir: Path) -> dict[str, Counter]:
    """Run git blame on each page file and count lines per author."""
    raw_contributors: dict[str, Counter] = defaultdict(Counter)
    for page_file in sorted(pages_dir.glob("*.py")):
        if page_file.name.startswith("_"):
            continue  # skip __init__.py or similar

        result = subprocess.run(
            [  # noqa: S603 S607
                "git",
                "blame",
                "--line-porcelain",  # one author header per line
                "--",
                str(page_file),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=ROOT,
            check=False,
        )
        for line in result.stdout.splitlines():
            if line.startswith("author "):
                author = line[len("author ") :].strip()
                raw_contributors[page_file.name][author] += 1

    return raw_contributors


def build_contributors(raw_contributors: dict[str, Counter]) -> dict[str, list]:
    """Build and print the contributors dict sorted by line count."""
    contributors: dict[str, list] = {}
    for page, line_counts in sorted(raw_contributors.items()):
        sorted_authors = sorted(line_counts.items(), key=lambda x: x[1], reverse=True)
        authors_str = ", ".join(f"{author} ({count})" for author, count in sorted_authors)
        print(f"{page}: {authors_str}")  # noqa: T201
        contributors[page.removesuffix(".py")] = [x[0] for x in sorted_authors]
    return contributors


def save_contributors(contributors: dict[str, list], output_path: Path) -> None:
    """Save contributors dict to a JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(contributors, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    pages_dir = ROOT / "src/processing"
    raw_contributors = get_raw_contributors(pages_dir)
    contributors = build_contributors(raw_contributors)
    output_path = ROOT / "data/cleaned/contributors.json"
    save_contributors(contributors, output_path)

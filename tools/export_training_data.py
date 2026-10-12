"""Create a local transfer archive for the training PC; does not upload it."""

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import zipfile
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = (
        ROOT
        / "exports"
        / (
            "starfall-training-data-"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + ".zip"
        )
    )
    output.parent.mkdir(exist_ok=True)
    folders = [
        "data/game",
        "data/player-reviewed",
        "models/real-response",
        "models/detector-experiments",
    ]
    files = {p for folder in folders for p in (ROOT / folder).rglob("*") if p.is_file()}
    files.update(p for p in (ROOT / "runs").glob("20*/*") if p.is_file())
    files.update(
        ROOT / "reports" / name
        for name in [
            "player-detector-candidate.json",
            "combat-screening.json",
            "combat-reference-provenance.json",
            "offline-experiments.json",
        ]
    )
    records = []
    with zipfile.ZipFile(
        output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3
    ) as archive:
        for path in sorted(files):
            if not path.is_file() or path.suffix in (".cache", ".pyc"):
                continue
            name = path.relative_to(ROOT).as_posix()
            contents = path.read_bytes()
            if path.name == "dataset.yaml" and (path.parent / "images/train").is_dir():
                config = yaml.safe_load(contents)
                config.pop("path", None)
                contents = yaml.safe_dump(config, sort_keys=False).encode("utf-8")
            archive.writestr(name, contents)
            records.append(
                {
                    "path": name,
                    "bytes": len(contents),
                    "sha256": hashlib.sha256(contents).hexdigest(),
                }
            )
        archive.writestr(
            "TRANSFER-MANIFEST.json",
            json.dumps(
                {
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "usage": "Extract into the repository root on the other PC. Stop local phone control before copying runs. No controller lock/stop files are included.",
                    "privacy": "Local transfer only; raw gameplay and recordings are excluded from GitHub.",
                    "candidate_status": "No detector or policy is promoted by importing this archive.",
                    "files": records,
                },
                indent=2,
            ),
        )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".zip.sha256").write_text(
        digest + "  " + output.name + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "archive": str(output),
                "files": len(records),
                "mb": round(output.stat().st_size / 2**20, 2),
                "sha256": digest,
                "uploaded": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

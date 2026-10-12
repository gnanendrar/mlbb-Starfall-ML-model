"""Fetch a pinned official portable scrcpy bundle and verify its SHA256."""

import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = "5.0.1"
SHA256 = "b12a2c4ee8be317422451fc7dcf8ee20a71b5ea7ef9ad73ddd825a25316227a5"
URL = f"https://github.com/Genymobile/scrcpy/releases/download/v{VERSION}/scrcpy-win64-v{VERSION}.zip"


def main():
    (ROOT / "reports").mkdir(exist_ok=True)
    folder = ROOT / "tools" / f"scrcpy-win64-v{VERSION}"
    if not (folder / "scrcpy.exe").exists():
        archive = ROOT / "tools" / f"scrcpy-win64-v{VERSION}.zip"
        print(f"Downloading official scrcpy {VERSION}", flush=True)
        urllib.request.urlretrieve(URL, archive)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != SHA256:
            raise RuntimeError("scrcpy archive SHA256 mismatch")
        with zipfile.ZipFile(archive) as z:
            for member in z.namelist():
                target = (ROOT / "tools" / member).resolve()
                if not target.is_relative_to((ROOT / "tools").resolve()):
                    raise RuntimeError("Archive path outside tools")
            z.extractall(ROOT / "tools")
        archive.unlink()
    report = {
        "version": VERSION,
        "source": URL,
        "sha256": SHA256,
        "directory": str(folder),
    }
    (ROOT / "reports" / "scrcpy-install.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

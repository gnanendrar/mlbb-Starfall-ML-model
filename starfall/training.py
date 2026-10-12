"""Explicit compute selection and portable real-frame dataset paths."""

from pathlib import Path
import yaml


def select_device(requested="auto"):
    import torch

    requested = str(requested).lower()
    if requested == "cpu":
        return "cpu"
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    index = requested.removeprefix("cuda:")
    if requested == "cuda":
        index = "0"
    if not index.isdecimal():
        raise ValueError("Device must be auto, cpu, cuda, cuda:N or a CUDA GPU index")
    if not torch.cuda.is_available() or int(index) >= torch.cuda.device_count():
        raise RuntimeError(
            "Requested NVIDIA GPU is unavailable. Run Setup-Training.cmd and check "
            "the NVIDIA driver, or explicitly select --device cpu."
        )
    return f"cuda:{int(index)}"


def prepare_dataset(dataset, destination):
    """Resolve old machine paths without modifying the source dataset/split."""
    source = Path(dataset).resolve()
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = Path(config.get("path") or source.parent)
    if not root.is_absolute():
        root = source.parent / root
    # Old exported splits stored an absolute path on the collecting PC.
    if not root.exists() and (source.parent / "images/train").is_dir():
        root = source.parent
    root = root.resolve()
    for subset in ("train", "val", "test"):
        entries = config.get(subset)
        if not entries:
            if subset == "test":
                continue
            raise ValueError(f"Dataset has no {subset} split")
        for entry in entries if isinstance(entries, list) else [entries]:
            if not (root / entry).exists():
                raise FileNotFoundError(f"Missing {subset} data: {root / entry}")
    config["path"] = root.as_posix()
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return target.resolve()

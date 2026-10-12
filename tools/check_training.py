"""Check the installed compute backend by actually performing a tensor operation."""

import argparse
import json
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    available = torch.cuda.is_available()
    if args.require_cuda and not available:
        raise SystemExit(
            "NVIDIA CUDA is unavailable. Check the NVIDIA driver and the CUDA PyTorch "
            "installation. No GPU training was started."
        )
    device = "cuda:0" if available else "cpu"
    x = torch.ones((32, 32), device=device)
    assert (x @ x).sum().item() == 32768
    report = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "selected": device,
        "tensor_check": "passed",
        "gpus": [
            {
                "cuda_index": i,
                "name": torch.cuda.get_device_name(i),
                "dedicated_gb": round(
                    torch.cuda.get_device_properties(i).total_memory / 2**30, 2
                ),
            }
            for i in range(torch.cuda.device_count())
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

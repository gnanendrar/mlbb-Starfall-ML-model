"""Small visually reviewed pilot set; deliberately not a detector training claim."""

from pathlib import Path
import json
import numpy as np
import cv2
from starfall.perception import create_detector
from starfall.runtime import load_config


def main():
    cfg = load_config("configs/phone.yaml")
    detector = create_detector(cfg)
    cv2.setNumThreads(2)
    # Approximate hand-reviewed boxes from the visible phone frames.
    cases = [
        (
            "data/game/images/real-inspection-1-00000221.jpg",
            [("player", (177, 160, 301, 229))],
        ),
        (
            "data/game/images/real-inspection-1-00000402.jpg",
            [
                ("player", (177, 160, 301, 229)),
                ("blue_meteor", (483, 312, 568, 347)),
                ("orange_meteor", (911, 166, 960, 221)),
            ],
        ),
        (
            "data/game/images/real-inspection-1-00000664.jpg",
            [
                ("player", (177, 160, 301, 229)),
                ("orange_meteor", (257, 100, 333, 153)),
                ("purple_meteor", (765, 151, 859, 245)),
            ],
        ),
        (
            "data/game/images/real-inspection-1-00000749.jpg",
            [
                ("player", (177, 160, 301, 229)),
                ("orange_meteor", (700, 100, 780, 153)),
                ("starshot_split", (866, 171, 914, 222)),
            ],
        ),
        (
            "reports/health-events/1791766265065633800-one_heart/trigger.png",
            [
                ("player", (177, 160, 301, 229)),
                ("orange_meteor", (342, 32, 421, 85)),
                ("orange_meteor", (842, 168, 925, 222)),
                ("orange_meteor", (593, 232, 675, 288)),
                ("purple_meteor", (367, 280, 478, 388)),
            ],
        ),
    ]
    totals = {"tp": 0, "fp": 0, "fn": 0}
    rows = []
    for path, truth in cases:
        image = cv2.imread(path)
        predicted = detector.detect(image)
        scale = max(image.shape[:2])
        used = set()
        tp = 0
        for label, (x0, y0, x1, y1) in truth:
            center = np.array([(x0 + x1) / 2, (y0 + y1) / 2]) / scale
            matches = [
                (np.linalg.norm(d.position - center), i)
                for i, d in enumerate(predicted)
                if d.label == label and i not in used
            ]
            if matches:
                distance, index = min(matches)
                if distance < 0.045:
                    tp += 1
                    used.add(index)
        fp = len(predicted) - tp
        fn = len(truth) - tp
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        rows.append(
            {
                "frame": path,
                "approximate_ground_truth": truth,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "detections": [
                    {
                        "label": d.label,
                        "position": d.position.tolist(),
                        "confidence": d.confidence,
                    }
                    for d in predicted
                ],
            }
        )
    report = {
        "provenance": "visually_reviewed_limited_pilot_screening",
        "images": len(cases),
        "matching_rule": "same label, center error < .045 of max frame dimension",
        "precision": totals["tp"] / max(1, totals["tp"] + totals["fp"]),
        "recall": totals["tp"] / max(1, totals["tp"] + totals["fn"]),
        **totals,
        "rows": rows,
        "limitations": "Source frames overlap template selection; approximate boxes; unsupported items and obscured edge objects are not an exhaustive test. Not mAP or independent game accuracy.",
    }
    Path("reports/vision-pilot-validation.json").write_text(
        json.dumps(report, indent=2)
    )
    print(
        json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()

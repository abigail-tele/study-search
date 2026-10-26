"""Sweep the semantic cancer-type detection threshold on the hand-written query set.

Run: python -m pipeline.evaluate_types [--thresholds 0.45,0.50,0.55]

Why no database: type detection reads only data/cancer_types.json and the local
MiniLM model. It never touches trials.db or faiss.index, so this check runs
before the trial index is built. The detector must read
config.TYPE_DETECT_EMBED_THRESHOLD at call time for the sweep to work.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from backend import config, taxonomy


def load_queries(path: Path = config.TEST_QUERIES_PATH) -> dict:
    """Load the synthetic lay-query and off-topic sets (no user data)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def score(threshold: float, data: dict) -> dict:
    """Run detection at one threshold and return counts for the report table."""
    config.TYPE_DETECT_EMBED_THRESHOLD = threshold
    lay = data.get("lay_queries", [])
    off_topic = data.get("off_topic", [])

    exact = 0
    recalled = 0
    total_expected = 0
    for item in lay:
        expected = set(item.get("expected_types", []))
        detected = set(taxonomy.detect_cancer_types(item["query"]))
        exact += int(detected == expected)
        recalled += len(expected & detected)
        total_expected += len(expected)

    false_positives = []
    for item in off_topic:
        detected = taxonomy.detect_cancer_types(item["query"])
        if detected:
            false_positives.append(f"{item['id']}:{item['query']}->{','.join(detected)}")

    return {
        "threshold": f"{threshold:.2f}",
        "lay_exact": exact,
        "lay_total": len(lay),
        "lay_recall": round(recalled / total_expected, 4) if total_expected else 0.0,
        "off_topic_pass": len(off_topic) - len(false_positives),
        "off_topic_total": len(off_topic),
        "false_positives": " | ".join(false_positives),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--thresholds", default="0.45,0.50,0.55,0.60,0.65,0.70",
                    help="comma-separated cosine thresholds to compare")
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "type_detection.csv"),
                    help="where to write the CSV table")
    args = ap.parse_args(argv)

    thresholds = [float(t) for t in args.thresholds.split(",")]
    data = load_queries()
    original_threshold = config.TYPE_DETECT_EMBED_THRESHOLD
    try:
        rows = [score(t, data) for t in thresholds]
    except NotImplementedError:
        raise SystemExit("detect_cancer_types is still the human-written stub; "
                         "implement it before running this sweep.")
    finally:
        config.TYPE_DETECT_EMBED_THRESHOLD = original_threshold

    for row in rows:
        print(f"threshold {row['threshold']}: "
              f"lay exact {row['lay_exact']}/{row['lay_total']}, "
              f"recall {row['lay_recall']:.3f}, "
              f"off-topic pass {row['off_topic_pass']}/{row['off_topic_total']}")
        if row["false_positives"]:
            print(f"  false positives: {row['false_positives']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

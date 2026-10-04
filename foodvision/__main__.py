"""Command-line entry points for the demo, image prediction and benchmark."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Estimate remaining-food fraction from paired meal photos.")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Open a local browser demo")
    demo.add_argument("--port", type=int, default=8765)
    predict = sub.add_parser("predict", help="Analyze a new before/after pair")
    predict.add_argument("--before", type=Path, required=True)
    predict.add_argument("--after", type=Path, required=True)
    predict.add_argument("--starting-portion", choices=("visible_food", "empty_or_residue", "uncertain"),
                         help="Your visual observation of the before image; not an automatic food-detection result")
    benchmark = sub.add_parser("benchmark", help="Rerun the bundled 10-episode feature benchmark")
    benchmark.add_argument("--output", type=Path, default=Path("reports/my-benchmark.json"))
    diagnose = sub.add_parser("diagnose", help="Refit existing models and audit held-out failure slices")
    diagnose.add_argument("--output", type=Path, default=Path("reports/my-diagnostics.json"))
    segment = sub.add_parser("segment-cache", help="Extract frozen experimental food-mask features")
    segment.add_argument("--research-root", type=Path, required=True)
    segment.add_argument("--output", type=Path, required=True)
    experiment = sub.add_parser("segmentation-experiment", help="Evaluate frozen segmentation features against the original models")
    experiment.add_argument("--cache", type=Path, required=True)
    experiment.add_argument("--output", type=Path, required=True)
    reviews = sub.add_parser("import-reviews", help="Validate a Review lab export and append it to local SQLite history")
    reviews.add_argument("--input", type=Path, required=True)
    reviews.add_argument("--db", type=Path, default=Path(".local/reviews.sqlite3"))
    review_status = sub.add_parser("review-status", help="Summarize imported visual reviews without changing them")
    review_status.add_argument("--db", type=Path, default=Path(".local/reviews.sqlite3"))
    sub.add_parser("verify", help="Check bundled input and model checksums")
    args = parser.parse_args()
    if args.command == "demo":
        from .server import serve
        serve(args.port)
    elif args.command == "predict":
        from .inference import Predictor
        result = Predictor().photos(args.before.read_bytes(), args.after.read_bytes(), args.starting_portion)
        print(json.dumps(result, indent=2))
    elif args.command == "benchmark":
        from .benchmark import run
        result = run(args.output)
        print(json.dumps(result["summary"], indent=2))
    elif args.command == "diagnose":
        from .diagnostics import run
        result = run(args.output)
        print(json.dumps({source: methods["Change"] for source, methods in result["summary"].items()}, indent=2))
    elif args.command == "segment-cache":
        from .segmentation import extract
        result = extract(args.research_root, args.output)
        print(f"Saved predicted segmentation features for {len(result['records'])} pairs.")
    elif args.command == "segmentation-experiment":
        from .segmentation_experiment import run
        result = run(args.cache, args.output)
        print(json.dumps({k: v["all"] for k, v in result["summary"].items()}, indent=2))
    elif args.command in ("import-reviews", "review-status"):
        import sqlite3
        from .reviews import import_reviews, review_status
        try:
            result = import_reviews(args.input, args.db) if args.command == "import-reviews" else review_status(args.db)
        except (ValueError, OSError, sqlite3.Error) as exc:
            parser.error(str(exc))
        print(json.dumps(result, indent=2))
    else:
        from .benchmark import verify_payload
        print(f"Verified {verify_payload()} bundled files.")


if __name__ == "__main__":
    main()

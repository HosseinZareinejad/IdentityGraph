"""
Phase 2 - ETL orchestrator.

Runs the full pipeline in order and reports timing:

    data/platform_dumps/*.parquet   (Phase 1 output)
        -> etl/clean.py            -> data/cleaned/*.parquet
        -> etl/graph_features.py   -> data/graph_features/*.parquet
        -> etl/load_qdrant.py      -> Qdrant collection "identity_accounts"
        -> etl/validate.py         -> correctness + quality report

Idempotent: cleaning and feature files are overwritten, and the Qdrant
collection is dropped and rebuilt, so re-running after regenerating Phase 1
data (or changing the embedding model) is always safe.

Usage:
    python etl/run_all.py              # full pipeline
    python etl/run_all.py --skip-load  # cleaning + graph features only
"""
import os
import sys
import time

os.environ["NO_PROXY"] = "localhost,127.0.0.1"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from etl import clean, graph_features, load_qdrant  # noqa: E402


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    skip_load = "--skip-load" in sys.argv
    t_start = time.time()

    print("=" * 70)
    print("STEP 1/3  Cleaning & standardization")
    print("=" * 70)
    t = time.time()
    clean.run()
    print(f"  done in {time.time() - t:.1f}s")

    print("\n" + "=" * 70)
    print("STEP 2/3  Graph structural features")
    print("=" * 70)
    t = time.time()
    graph_features.run()
    print(f"  done in {time.time() - t:.1f}s")

    if skip_load:
        print("\n--skip-load given; stopping before Qdrant load.")
        return

    print("\n" + "=" * 70)
    print("STEP 3/3  Embedding & Qdrant load")
    print("=" * 70)
    t = time.time()
    load_qdrant.run()
    print(f"  done in {time.time() - t:.1f}s")

    print(f"\nETL pipeline finished in {time.time() - t_start:.1f}s total.")
    print("Run `python etl/validate.py` to verify correctness and retrieval quality.")


if __name__ == "__main__":
    main()

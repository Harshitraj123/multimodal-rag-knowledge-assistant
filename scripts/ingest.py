"""
CLI script to ingest one or more documents.
Usage: python scripts/ingest.py path/to/file1 [path/to/file2 ...]
       python scripts/ingest.py --dir data/demo_corpus/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG system")
    parser.add_argument("files", nargs="*", help="File paths to ingest")
    parser.add_argument("--dir", help="Directory to ingest all supported files from")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    args = parser.parse_args()

    from app.config.settings import settings
    from app.config.logging_config import setup_logging
    setup_logging(settings.log_level)

    from app.api.dependencies import _init_components, get_document_service
    _init_components()
    svc = get_document_service()

    files_to_ingest = []

    if args.dir:
        d = Path(args.dir)
        if not d.exists():
            print(f"ERROR: Directory not found: {d}")
            sys.exit(1)
        from app.ingestion.loaders import SUPPORTED_EXTENSIONS
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in SUPPORTED_EXTENSIONS:
                files_to_ingest.append(f)

    for fpath in args.files:
        files_to_ingest.append(Path(fpath))

    if not files_to_ingest:
        print("No files to ingest. Use --dir or provide file paths.")
        sys.exit(0)

    print(f"Ingesting {len(files_to_ingest)} file(s)...")
    for fpath in files_to_ingest:
        try:
            result = svc.ingest_file(fpath)
            status = result["status"]
            if status == "success":
                print(
                    f"  ✓ {result['file_name']}: "
                    f"{result['chunks_added']} chunks, "
                    f"{result['tables_indexed']} tables, "
                    f"{result['visuals_indexed']} visuals "
                    f"({result['processing_time_s']:.2f}s)"
                )
            elif status == "duplicate":
                print(f"  ~ {result['file_name']}: already indexed (skipped)")
            else:
                print(f"  ✗ {result['file_name']}: ERROR - {result.get('error_message', 'unknown')}")
        except Exception as e:
            print(f"  ✗ {fpath.name}: EXCEPTION - {e}")

    print("Done.")


if __name__ == "__main__":
    main()

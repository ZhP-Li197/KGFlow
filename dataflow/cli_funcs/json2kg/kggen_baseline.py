#!/usr/bin/env python3
"""KGGen baseline wrapper aligned with the json2kg one-click workflow."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

JSON2KG_DIR = Path(__file__).resolve().parent
BASELINE_DIR = JSON2KG_DIR / "KGGen"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))


def _load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported JSON root in {path}: {type(data)}")


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _relation_to_triple(relation: Any) -> str | None:
    if isinstance(relation, dict):
        src = relation.get("source") or relation.get("subject") or relation.get("head")
        rel = relation.get("relation") or relation.get("type") or relation.get("predicate")
        tgt = relation.get("target") or relation.get("object") or relation.get("tail")
        if src and rel and tgt:
            return f"<subj> {src} <obj> {tgt} <rel> {rel}"
        return None

    if isinstance(relation, (list, tuple)) and len(relation) >= 3:
        src, rel, tgt = relation[0], relation[1], relation[2]
        return f"<subj> {src} <obj> {tgt} <rel> {rel}"

    if isinstance(relation, str) and relation.strip():
        return relation.strip()

    return None


def _normalize_output(raw_output: Path, final_output: Path, input_key: str, output_key: str) -> None:
    records = _load_records(raw_output)
    normalized = []

    for item in records:
        kg = item.get("extracted_kg") or {}
        triples = []
        for relation in kg.get("relations", []) or []:
            triple = _relation_to_triple(relation)
            if triple:
                triples.append(triple)

        normalized.append(
            {
                "id": item.get("id"),
                input_key: item.get(input_key, item.get("text", "")),
                output_key: triples,
                "baseline": "KGGen",
                "extraction_status": kg.get("extraction_status", "unknown"),
            }
        )

    _save_json(final_output, normalized)


def run(args: argparse.Namespace) -> Path:
    from extract_kg import extract_kg_from_dataset

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    cache_path = Path(args.cache).resolve()
    cache_path.mkdir(parents=True, exist_ok=True)

    source_records = _load_records(input_path)
    kggen_input = cache_path / "kggen_input.json"
    kggen_raw_output = cache_path / "kggen_raw_output.json"

    prepared = []
    for idx, item in enumerate(source_records):
        text = item.get(args.input_key, item.get("text", ""))
        prepared.append(
            {
                "id": item.get("id", idx),
                "text": text,
            }
        )
    _save_json(kggen_input, prepared)

    api_key = args.api_key or os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"Missing API key. Set {args.api_key_env} or pass --api-key.")

    extract_kg_from_dataset(
        dataset_path=str(kggen_input),
        output_path=str(kggen_raw_output),
        model=args.model,
        api_key=api_key,
        api_base=args.api_base,
        retrieval_model=args.retrieval_model,
        chunk_size=args.chunk_size,
        cluster=args.cluster,
        resume=not args.no_resume,
        save_every=args.save_every,
        retry_errors=args.retry_errors,
        force=args.force,
    )

    _normalize_output(kggen_raw_output, output_path, args.input_key, args.output_key)
    print(f"KGGen normalized predictions written to: {output_path}")
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run KGGen baseline with json2kg-compatible I/O")
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--output", default="./kggen_output.json", help="Prediction JSON file")
    parser.add_argument("--cache", default="./cache_kggen", help="Cache directory")
    parser.add_argument("--input-key", default="text", help="Input text field")
    parser.add_argument("--output-key", default="triple", help="Output triple field")
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", os.environ.get("KGGEN_MODEL", "gpt-4o-mini")))
    parser.add_argument("--api-base", default=os.environ.get("DF_BASE_URL", os.environ.get("KGGEN_API_BASE", "https://api.openai.com/v1")))
    parser.add_argument("--api-key-env", default="DF_API_KEY")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--retrieval-model", default=os.environ.get("KGGEN_RETRIEVAL_MODEL", "all-MiniLM-L6-v2"))
    parser.add_argument("--chunk-size", type=int, default=2500)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())

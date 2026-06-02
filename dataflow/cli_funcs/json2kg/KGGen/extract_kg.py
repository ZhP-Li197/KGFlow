"""
Extract KGs for each dataset item and write them back to JSON.

This version supports:
- resuming from an existing output file
- periodic checkpoint saves during extraction
- optional retry of previously failed items
python extract_kg.py 
  --dataset dataset\L5_diff\l5_relation_only_with_difficulty.json 
  --retry-errors
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.kg_gen.kg_gen import KGGen


DEFAULT_API_KEY = os.environ.get("KGGEN_API_KEY") or os.environ.get("DF_API_KEY") or os.environ.get("OPENAI_API_KEY")
DEFAULT_BASE_URL = os.environ.get("KGGEN_API_BASE", "https://api.openai.com/v1")
DEFAULT_MODEL = "openai/gemini-2.5-flash"
DEFAULT_RETRIEVAL_MODEL = "all-MiniLM-L6-v2"


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    with open(dataset_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_dataset(dataset: List[Dict[str, Any]], output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    temp_path = output_file.with_suffix(output_file.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(dataset, file, ensure_ascii=False, indent=2)
    os.replace(temp_path, output_file)


def build_empty_kg(error: Optional[str] = None, status: str = "error") -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "entities": [],
        "relations": [],
        "edges": [],
        "stats": {
            "entity_count": 0,
            "relation_count": 0,
            "edge_count": 0,
        },
        "extraction_status": status,
    }
    if error is not None:
        payload["error"] = error
    return payload


def build_kg_payload(graph: Any) -> Dict[str, Any]:
    kg_data: Dict[str, Any] = {
        "entities": list(graph.entities),
        "relations": list(graph.relations),
        "edges": list(graph.edges),
        "stats": {
            "entity_count": len(graph.entities),
            "relation_count": len(graph.relations),
            "edge_count": len(graph.edges),
        },
        "extraction_status": "success",
    }

    if hasattr(graph, "entity_clusters") and graph.entity_clusters:
        kg_data["entity_clusters"] = {
            key: list(value) for key, value in graph.entity_clusters.items()
        }

    if hasattr(graph, "edge_clusters") and graph.edge_clusters:
        kg_data["edge_clusters"] = {
            key: list(value) for key, value in graph.edge_clusters.items()
        }

    return kg_data


def item_key(item: Dict[str, Any], index: int) -> str:
    return str(item.get("id", f"__index__:{index}"))


def restore_progress_from_output(
    dataset: List[Dict[str, Any]],
    output_path: str,
) -> int:
    if not os.path.exists(output_path):
        return 0

    previous_dataset = load_dataset(output_path)
    previous_items = {
        item_key(item, index): item for index, item in enumerate(previous_dataset)
    }

    restored = 0
    for index, item in enumerate(dataset):
        previous_item = previous_items.get(item_key(item, index))
        if previous_item is None or "extracted_kg" not in previous_item:
            continue

        item["extracted_kg"] = previous_item["extracted_kg"]
        restored += 1

    return restored


def should_skip_item(
    item: Dict[str, Any],
    retry_errors: bool,
    force: bool,
) -> bool:
    if force:
        return False

    extracted_kg = item.get("extracted_kg")
    if not extracted_kg:
        return False

    status = extracted_kg.get("extraction_status")
    if retry_errors and status == "error":
        return False

    if retry_errors and "error" in extracted_kg and status is None:
        return False

    return True


def init_kggen(
    model: str,
    api_key: Optional[str],
    api_base: Optional[str],
    retrieval_model: Optional[str],
) -> KGGen:
    effective_api_key = api_key or DEFAULT_API_KEY
    effective_api_base = api_base or DEFAULT_BASE_URL

    kg_gen = KGGen(
        model=model,
        api_key=effective_api_key,
        api_base=effective_api_base,
        retrieval_model=retrieval_model,
        temperature=0.1,
        max_tokens=4000,
    )
    kg_gen.init_model(
        model=model,
        api_key=effective_api_key,
        api_base=effective_api_base,
        retrieval_model=retrieval_model,
        temperature=0.1,
        max_tokens=4000,
    )
    return kg_gen


def extract_kg_from_dataset(
    dataset_path: str = "dataset/test.json",
    output_path: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    retrieval_model: Optional[str] = DEFAULT_RETRIEVAL_MODEL,
    chunk_size: int = 2500,
    cluster: bool = False,
    resume: bool = True,
    save_every: int = 5,
    retry_errors: bool = False,
    force: bool = False,
) -> None:
    dataset_path = os.path.normpath(dataset_path)
    output_file = os.path.normpath(output_path or dataset_path)

    print(f"Loading dataset: {dataset_path}")
    dataset = load_dataset(dataset_path)

    restored = 0
    if resume and output_file != dataset_path:
        restored = restore_progress_from_output(dataset, output_file)
        if restored > 0:
            print(f"Restored {restored} items from checkpoint: {output_file}")
    elif resume and output_file == dataset_path and os.path.exists(output_file):
        print("Resume uses the dataset file itself as the checkpoint source.")

    print("Initializing KG generator...")
    kg_gen = init_kggen(
        model=model,
        api_key=api_key,
        api_base=api_base,
        retrieval_model=retrieval_model,
    )
    print(f"Model: {kg_gen.model}")
    print(f"API base: {kg_gen.api_base}")
    print(f"Retrieval model: {retrieval_model or 'None'}")

    total_items = len(dataset)
    processed_now = 0
    skipped_existing = 0

    print(f"Start processing {total_items} items")

    for index, item in enumerate(dataset):
        title = item.get("title", item.get("id", "Unknown"))

        if should_skip_item(item, retry_errors=retry_errors, force=force):
            skipped_existing += 1
            print(f"[{index + 1}/{total_items}] Skip existing: {title}")
            continue

        print(f"[{index + 1}/{total_items}] Extracting: {title}")
        text = item.get("text", "")

        if not text:
            print("  Skip: empty text")
            item["extracted_kg"] = build_empty_kg(
                error="missing_text",
                status="missing_text",
            )
        else:
            try:
                graph = kg_gen.generate(
                    input_data=text,
                    chunk_size=chunk_size,
                    cluster=cluster,
                )
                item["extracted_kg"] = build_kg_payload(graph)
                print(
                    "  Success:"
                    f" entities={len(graph.entities)}"
                    f" relations={len(graph.relations)}"
                    f" edges={len(graph.edges)}"
                )
            except Exception as error:
                print(f"  Error: {error}")
                item["extracted_kg"] = build_empty_kg(str(error), status="error")

        processed_now += 1
        if save_every > 0 and processed_now % save_every == 0:
            save_dataset(dataset, output_file)
            print(f"  Checkpoint saved to: {output_file}")

    save_dataset(dataset, output_file)
    print(f"Finished. Output saved to: {output_file}")
    print(
        f"Summary: restored={restored}, skipped_existing={skipped_existing},"
        f" processed_now={processed_now}, total={total_items}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract KG from dataset items.")
    parser.add_argument(
        "--dataset",
        default="dataset\\L2\\L2_relation_only_with_difficulty_core_plus_modifier.json",
        help="Input dataset JSON path.",
    )
    parser.add_argument(
        "--output",
        help="Output dataset JSON path. Defaults to overwriting the input file.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="LLM model used by KGGen.",
    )
    parser.add_argument("--api-key", help="API key override.")
    parser.add_argument(
        "--api-base",
        default=DEFAULT_BASE_URL,
        help="API base override.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2500,
        help="Chunk size passed to KGGen.generate.",
    )
    parser.add_argument(
        "--retrieval-model",
        default=DEFAULT_RETRIEVAL_MODEL,
        help="Retrieval model used for clustering/deduplication.",
    )
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Enable KGGen clustering.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not restore progress from an existing output file.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save a checkpoint after every N newly processed items.",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Re-run items whose existing extracted_kg contains an error.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all items even if extracted_kg already exists.",
    )

    args = parser.parse_args()

    extract_kg_from_dataset(
        dataset_path=args.dataset,
        output_path=args.output,
        model=args.model,
        api_key=args.api_key or DEFAULT_API_KEY,
        api_base=args.api_base,
        retrieval_model=args.retrieval_model,
        chunk_size=args.chunk_size,
        cluster=args.cluster,
        resume=not args.no_resume,
        save_every=args.save_every,
        retry_errors=args.retry_errors,
        force=args.force,
    )


if __name__ == "__main__":
    main()

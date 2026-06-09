from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


JSON2KG_DIR = Path(__file__).resolve().parent
BASELINE_DIR = JSON2KG_DIR / "TreeKG"
SRC_DIR = BASELINE_DIR / "src"
EXPLICIT_DIR = SRC_DIR / "ExplicitKG"
HIDDEN_DIR = SRC_DIR / "HiddenKG"


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ImportError("PyYAML is required to run the TreeKG baseline. Install pyyaml first.") from exc

    with path.open("r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(path: Path, data: dict[str, Any]) -> None:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ImportError("PyYAML is required to run the TreeKG baseline. Install pyyaml first.") from exc

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    data = _load_json(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported input JSON root in {path}: {type(data)}")


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _run_script(script: Path, cwd: Path) -> None:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(SRC_DIR) + os.pathsep + env.get("PYTHONPATH", "")

    print(f"Running TreeKG step: {script.relative_to(SRC_DIR)}")
    subprocess.run(
        [sys.executable, str(script)],
        cwd=str(cwd),
        env=env,
        check=True,
    )


def _patch_explicit_config(args: argparse.Namespace, docx_name: str) -> None:
    config_path = EXPLICIT_DIR / "config" / "config.yaml"
    text_path = EXPLICIT_DIR / "config" / "text.yaml"

    config = _load_yaml(config_path)
    config["PROJECT_ROOT"] = str(BASELINE_DIR)
    config["SRC_DIR"] = str(SRC_DIR)
    config["EXPLICIT_KG_DIR"] = str(EXPLICIT_DIR)
    config["OUTPUT_DIR"] = str(EXPLICIT_DIR / "output")
    api_config = config.setdefault("APIConfig", {})
    api_config["API_BASE"] = args.api_base
    api_config["API_KEY"] = args.api_key or os.environ.get(args.api_key_env, "")
    api_config["MODEL_NAME"] = args.model
    _save_yaml(config_path, config)

    text_config = _load_yaml(text_path)
    text_config.setdefault("TextSegConfig", {})["DOCX_NAME"] = docx_name
    _save_yaml(text_path, text_config)


def _patch_api_config(args: argparse.Namespace) -> None:
    config_path = EXPLICIT_DIR / "config" / "config.yaml"
    config = _load_yaml(config_path)
    config["PROJECT_ROOT"] = str(BASELINE_DIR)
    config["SRC_DIR"] = str(SRC_DIR)
    config["EXPLICIT_KG_DIR"] = str(EXPLICIT_DIR)
    config["OUTPUT_DIR"] = str(EXPLICIT_DIR / "output")
    api_config = config.setdefault("APIConfig", {})
    api_config["API_BASE"] = args.api_base
    api_config["API_KEY"] = args.api_key or os.environ.get(args.api_key_env, "")
    api_config["MODEL_NAME"] = args.model
    _save_yaml(config_path, config)


def _load_extraction_module():
    module_name = "treekg_explicit_extraction"
    module_path = EXPLICIT_DIR / "Extraction.py"
    if module_name in sys.modules:
        del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load TreeKG extraction module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _edge_to_triple(edge: dict[str, Any]) -> str | None:
    src = edge.get("source") or edge.get("u")
    tgt = edge.get("target") or edge.get("v")
    rel = edge.get("type") or edge.get("relationship") or edge.get("rel")
    if src and rel and tgt:
        return f"<subj> {src} <obj> {tgt} <rel> {rel}"
    return None


def _relation_to_triple(relation: dict[str, Any]) -> str | None:
    src = relation.get("source")
    tgt = relation.get("target")
    rel = relation.get("type") or relation.get("relationship") or relation.get("description")
    if src and rel and tgt:
        return f"<subj> {src} <obj> {tgt} <rel> {rel}"
    return None


def _normalize_tree_output(output_path: Path, final_output: Path, output_key: str) -> None:
    graph = _load_json(output_path)
    triples = []
    for edge in graph.get("edges", []) or []:
        triple = _edge_to_triple(edge)
        if triple:
            triples.append(triple)

    _save_json(
        final_output,
        [
            {
                "baseline": "TreeKG",
                output_key: triples,
                "nodes": graph.get("nodes", []),
                "edges": graph.get("edges", []),
            }
        ],
    )


def _resolve_docx_input(input_path: Path) -> Path:
    if input_path.suffix.lower() == ".docx":
        return input_path

    sibling_docx = input_path.with_suffix(".docx")
    if sibling_docx.exists():
        return sibling_docx

    raise ValueError(
        "TreeKG baseline expects a .docx input. "
        f"Received {input_path}. Provide a .docx file or place {sibling_docx.name} next to the loaded dataset."
    )


def _run_json_input(args: argparse.Namespace) -> Path:
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    cache_path = Path(args.cache).resolve()
    cache_path.mkdir(parents=True, exist_ok=True)

    _patch_api_config(args)
    extraction = _load_extraction_module()
    records = _load_records(input_path)
    predictions = []

    print(f"Running TreeKG chunk extraction on {len(records)} records...")
    for index, record in enumerate(records):
        text = str(record.get(args.input_key, "") or "")
        item_id = str(record.get("id") or f"item_{index:06d}")
        title = record.get("title")

        if not text.strip():
            result = {"entities": [], "relations": []}
        else:
            result = extraction.process_one_subsection(text)

        relations = result.get("relations", []) or []
        triples = []
        for relation in relations:
            triple = _relation_to_triple(relation)
            if triple:
                triples.append(triple)

        predictions.append(
            {
                "id": item_id,
                "title": title,
                "baseline": "TreeKG",
                "relational_facts": record.get("relational_facts", []),
                args.output_key: triples,
                "entities": result.get("entities", []) or [],
                "relations": relations,
            }
        )
        print(f"  [{index + 1}/{len(records)}] {item_id}: {len(triples)} triples")

    _save_json(output_path, predictions)
    shutil.copy2(output_path, cache_path / output_path.name)
    print(f"TreeKG normalized predictions written to: {output_path}")
    return output_path


def run(args: argparse.Namespace) -> Path:
    raw_input_path = Path(args.input).resolve()
    if raw_input_path.suffix.lower() in {".json", ".jsonl"}:
        return _run_json_input(args)

    input_path = _resolve_docx_input(raw_input_path)
    output_path = Path(args.output).resolve()
    cache_path = Path(args.cache).resolve()
    cache_path.mkdir(parents=True, exist_ok=True)

    explicit_output = EXPLICIT_DIR / "output"
    hidden_output = HIDDEN_DIR / "output"
    explicit_output.mkdir(parents=True, exist_ok=True)
    hidden_output.mkdir(parents=True, exist_ok=True)

    docx_name = input_path.name
    shutil.copy2(input_path, explicit_output / docx_name)
    _patch_explicit_config(args, docx_name)

    if args.mode in ("explicit", "full"):
        for rel_script in [
            "TextSegmentation.py",
            "Summarize.py",
            "Extraction.py",
            "toc_graph.py",
        ]:
            _run_script(EXPLICIT_DIR / rel_script, cwd=EXPLICIT_DIR)

    if args.mode == "full":
        for rel_script in [
            "Conv.py",
            "Aggr.py",
            "Embedding.py",
            "Dedup.py",
            "Pred.py",
            "FinalKG.py",
        ]:
            _run_script(HIDDEN_DIR / rel_script, cwd=HIDDEN_DIR)
        native_output = HIDDEN_DIR / "output" / "final_kg.json"
    else:
        native_output = EXPLICIT_DIR / "output" / "toc_graph.json"

    if not native_output.exists():
        raise FileNotFoundError(f"TreeKG output not found: {native_output}")

    shutil.copy2(native_output, cache_path / native_output.name)
    _normalize_tree_output(native_output, output_path, args.output_key)
    print(f"TreeKG normalized predictions written to: {output_path}")
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TreeKG baseline with json2kg-compatible I/O")
    parser.add_argument("--input", required=True, help="Input JSON/JSONL dataset or .docx file")
    parser.add_argument("--output", default="./treekg_output.json", help="Prediction JSON file")
    parser.add_argument("--cache", default="./cache_treekg", help="Cache directory")
    parser.add_argument("--input-key", default="text", help="Input text field for JSON/JSONL datasets")
    parser.add_argument("--output-key", default="triple", help="Output triple field")
    parser.add_argument("--mode", choices=["explicit", "full"], default="explicit")
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", os.environ.get("TREEKG_MODEL", "gpt-4o-mini")))
    parser.add_argument("--api-base", default=os.environ.get("DF_BASE_URL", os.environ.get("TREEKG_API_BASE", "https://api.openai.com/v1")))
    parser.add_argument("--api-key-env", default="DF_API_KEY")
    parser.add_argument("--api-key", default=None)
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())

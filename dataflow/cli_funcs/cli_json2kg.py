import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from colorama import Fore, Style

HF_DATASET_REPO = "b1u1/KGFlow-bench"

KGFLOW_PIPELINES = {
    "general": "generalkg_extraction_pipeline.py",
    "finance": "finkg_extraction_pipeline.py",
    "medical": "medkg_extraction_pipeline.py",
    "legal": "legalkg_extraction_pipeline.py",
    "temporal": "tkg_extraction.py",
}


def _json2kg_dir() -> Path:
    return Path(__file__).parent / "json2kg"


def _input_jsonl_path(dataset: str) -> Path:
    return Path(os.getcwd()) / f"{dataset}.jsonl"


def _method_output_path(method: str) -> Path:
    return Path(os.getcwd()) / f"{method}_output.json"


def _api_pipeline_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "statics" / "pipelines" / "api_pipelines"


def _latest_pipeline_output(cache_dir: Path, output_prefix: str) -> Path | None:
    files = list(cache_dir.glob(f"{output_prefix}_step*.jsonl")) + list(cache_dir.glob(f"{output_prefix}_step*.json"))
    if not files:
        return None

    def step_number(path: Path) -> int:
        match = re.search(r"_step(\d+)\.", path.name)
        return int(match.group(1)) if match else -1

    return max(files, key=step_number)


def _load_json_records(path: Path) -> list[dict]:
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


def _normalize_triple(triple):
    if isinstance(triple, (list, tuple)) and len(triple) >= 3:
        return [str(triple[0]), str(triple[1]), str(triple[2])]

    if isinstance(triple, str):
        text = triple.strip()
        if text.startswith("<subj>") and "<obj>" in text and "<rel>" in text:
            try:
                subject_part, rest = text.split("<obj>", 1)
                object_part, relation_part = rest.split("<rel>", 1)
                subject = subject_part.replace("<subj>", "", 1).strip()
                relation = relation_part.strip()
                obj = object_part.strip()
                if subject and relation and obj:
                    return [subject, relation, obj]
            except ValueError:
                return None
    return None


def _normalize_triples(triples) -> list[list[str]]:
    output = []
    for triple in triples or []:
        normalized = _normalize_triple(triple)
        if normalized:
            output.append(normalized)
    return output


def _write_predictions_with_gold_fields(input_path: Path, prediction_path: Path, output_path: Path) -> None:
    gold_records = _load_json_records(input_path)
    predictions = _load_json_records(prediction_path)
    gold_by_id = {
        str(record.get("id") or f"item_{index:06d}"): record
        for index, record in enumerate(gold_records)
    }

    enriched = []
    for index, pred in enumerate(predictions):
        item_id = str(pred.get("id") or f"item_{index:06d}")
        gold = gold_by_id.get(item_id, {})
        item = dict(pred)
        item["id"] = item_id
        if "relational_facts" in gold:
            item["relational_facts"] = gold["relational_facts"]
        for key in ("title", "text", "url", "source", "domain", "metadata"):
            if key in gold and not item.get(key):
                item[key] = gold[key]
        enriched.append(item)

    output_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")


def _require_jsonl_dataset(dataset: str) -> Path | None:
    input_path = _input_jsonl_path(dataset)
    if not input_path.exists():
        print(f"{Fore.RED}Error: {input_path} not found. Run 'text2kg load {dataset}' first.{Style.RESET_ALL}")
        return None
    return input_path


def _require_api_env() -> tuple[str, str, str] | None:
    api_key = os.getenv("DF_API_KEY")
    api_url = os.getenv("DF_BASE_URL")
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    missing = []
    if not api_key:
        missing.append("DF_API_KEY")
    if not api_url:
        missing.append("DF_BASE_URL")
    if missing:
        print(f"{Fore.RED}Error: missing environment variable(s): {', '.join(missing)}{Style.RESET_ALL}")
        return None
    return api_key, api_url, model


def _import_from_json2kg(module_name: str):
    json2kg_dir = _json2kg_dir()
    sys.path.insert(0, str(json2kg_dir))
    try:
        return __import__(module_name)
    finally:
        if str(json2kg_dir) in sys.path:
            sys.path.remove(str(json2kg_dir))


def cli_json2kg_init(dataset: str):
    dst = _input_jsonl_path(dataset)

    print(f"{Fore.GREEN}Initializing json2kg dataset '{dataset}' in current working directory...{Style.RESET_ALL}")

    if dst.exists():
        user_input = input(f"  {Fore.YELLOW}Warning: {dataset}.jsonl already exists. Overwrite? (y/n): {Style.RESET_ALL}").strip().lower()
        if user_input != "y":
            print(f"  Skipping {dataset}.jsonl.")
            return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(f"{Fore.RED}Missing required package: huggingface_hub{Style.RESET_ALL}")
        print("Please run: pip install huggingface_hub")
        return

    try:
        src = Path(
            hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename=f"{dataset}.jsonl",
                repo_type="dataset",
            )
        )
    except Exception as exc:
        print(f"{Fore.RED}Error: failed to download {dataset}.jsonl from {HF_DATASET_REPO}: {exc}{Style.RESET_ALL}")
        return

    shutil.copy2(src, dst)
    print(f"{Fore.GREEN}[Downloaded]\nFrom: {HF_DATASET_REPO}/{dataset}.jsonl\nTo: {dst}{Style.RESET_ALL}")


def cli_json2kg_autoschemakg(dataset: str):
    input_path = _require_jsonl_dataset(dataset)
    if input_path is None:
        return

    api_env = _require_api_env()
    if api_env is None:
        return
    api_key, api_url, model = api_env

    missing_pkgs = []
    for pkg in ("openai", "tenacity", "jsonschema", "json_repair", "datasets", "tqdm", "networkx", "pandas", "numpy"):
        try:
            __import__(pkg)
        except ImportError:
            missing_pkgs.append(pkg)
    if missing_pkgs:
        print(f"{Fore.RED}Missing required packages: {', '.join(missing_pkgs)}{Style.RESET_ALL}")
        print(f"Please run: pip install {' '.join(missing_pkgs)}")
        return

    print(f"{Fore.GREEN}Running AutoSchemaKG on {input_path}...{Style.RESET_ALL}")
    module = _import_from_json2kg("autoschemakg_baseline")
    runner = module.AutoSchemaKGBaseline()
    parser = argparse.ArgumentParser()
    runner.add_args(parser)
    args = parser.parse_args([
        "--input", str(input_path),
        "--output", str(Path(os.getcwd()) / "autoschemakg_output.json"),
        "--api-key", api_key,
        "--api-url", api_url,
        "--model", model,
    ])
    result = runner.main(args)
    print(f"{Fore.GREEN}Done. Output saved to {result}{Style.RESET_ALL}")


def cli_json2kg_wikontic(dataset: str):
    input_path = _require_jsonl_dataset(dataset)
    if input_path is None:
        return

    api_env = _require_api_env()
    if api_env is None:
        return
    api_key, api_url, model = api_env

    missing_pkgs = []
    for pkg in ("openai", "tenacity", "httpx", "pymongo", "pydantic", "transformers", "tqdm", "langchain", "unidecode"):
        try:
            __import__(pkg)
        except ImportError:
            missing_pkgs.append(pkg)
    if missing_pkgs:
        print(f"{Fore.RED}Missing required packages: {', '.join(missing_pkgs)}{Style.RESET_ALL}")
        print(f"Please run: pip install {' '.join(missing_pkgs)}")
        return

    print(f"{Fore.GREEN}Running Wikontic on {input_path}...{Style.RESET_ALL}")
    module = _import_from_json2kg("wikontic_baseline")
    runner = module.WikonticBaseline()
    parser = argparse.ArgumentParser()
    runner.add_args(parser)
    args = parser.parse_args([
        "--input", str(input_path),
        "--output", str(Path(os.getcwd()) / "wikontic_output.json"),
        "--api-key", api_key,
        "--api-url", api_url,
        "--model", model,
    ])
    result = runner.main(args)
    print(f"{Fore.GREEN}Done. Output saved to {result}{Style.RESET_ALL}")


def cli_json2kg_kggen(dataset: str):
    input_path = _require_jsonl_dataset(dataset)
    if input_path is None:
        return

    api_env = _require_api_env()
    if api_env is None:
        return
    api_key, api_url, model = api_env

    kggen_requirements = [
        ("dspy", "dspy"),
        ("networkx", "networkx"),
        ("sentence_transformers", "sentence-transformers"),
        ("sklearn", "scikit-learn"),
        ("numpy", "numpy"),
        ("pydantic", "pydantic"),
        ("nltk", "nltk"),
        ("semhash", "semhash"),
        ("inflect", "inflect"),
        ("scipy", "scipy"),
        ("rank_bm25", "rank-bm25"),
        ("neo4j", "neo4j"),
    ]
    missing_pkgs = [
        pip_name
        for import_name, pip_name in kggen_requirements
        if importlib.util.find_spec(import_name) is None
    ]
    if missing_pkgs:
        install_cmd = "pip install " + " ".join(sorted(set(missing_pkgs)))
        print(f"{Fore.RED}Missing optional dependencies for KGGen: {', '.join(sorted(set(missing_pkgs)))}{Style.RESET_ALL}")
        print("KGGen is a third-party baseline and requires extra packages.")
        print(f"Please run: {install_cmd}")
        return

    print(f"{Fore.GREEN}Running KGGen on {input_path}...{Style.RESET_ALL}")
    module = _import_from_json2kg("kggen_baseline")
    parser = module.build_arg_parser()
    args = parser.parse_args([
        "--input", str(input_path),
        "--output", str(Path(os.getcwd()) / "kggen_output.json"),
        "--cache", str(Path(os.getcwd()) / "cache_kggen"),
        "--api-key", api_key,
        "--api-base", api_url,
        "--model", model,
    ])
    result = module.run(args)
    print(f"{Fore.GREEN}Done. Output saved to {result}{Style.RESET_ALL}")


def cli_json2kg_treekg(dataset: str):
    input_path = _require_jsonl_dataset(dataset)
    if input_path is None:
        return

    api_env = _require_api_env()
    if api_env is None:
        return
    api_key, api_url, model = api_env

    print(f"{Fore.GREEN}Running TreeKG on {input_path}...{Style.RESET_ALL}")
    module = _import_from_json2kg("treekg_baseline")
    parser = module.build_arg_parser()
    args = parser.parse_args([
        "--input", str(input_path),
        "--output", str(Path(os.getcwd()) / "treekg_output.json"),
        "--cache", str(Path(os.getcwd()) / "cache_treekg"),
        "--input-key", "text",
        "--api-key", api_key,
        "--api-base", api_url,
        "--model", model,
    ])
    result = module.run(args)
    print(f"{Fore.GREEN}Done. Output saved to {result}{Style.RESET_ALL}")


def cli_json2kg_kgflow(dataset: str, pipeline: str):
    input_path = _require_jsonl_dataset(dataset)
    if input_path is None:
        return

    api_env = _require_api_env()
    if api_env is None:
        return
    _, api_url, model = api_env

    if pipeline not in KGFLOW_PIPELINES:
        choices = ", ".join(sorted(KGFLOW_PIPELINES))
        print(f"{Fore.RED}Error: unsupported KGFlow pipeline '{pipeline}'. Choose from: {choices}.{Style.RESET_ALL}")
        return

    script_name = KGFLOW_PIPELINES[pipeline]
    script_path = _api_pipeline_dir() / script_name
    cache_dir = Path(os.getcwd()) / f"cache_kgflow_{pipeline}"
    output_prefix = f"kgflow_{pipeline}"
    output_path = _method_output_path("kgflow")

    cmd = [
        sys.executable,
        str(script_path),
        "--input",
        str(input_path),
        "--input-key",
        "text",
        "--cache-dir",
        str(cache_dir),
        "--output-prefix",
        output_prefix,
        "--model-name",
        model,
        "--api-url",
        api_url,
        "--api-key-env",
        "DF_API_KEY",
    ]

    if pipeline in {"finance", "medical", "legal"}:
        cmd.extend(["--triple-type", "coverage"])

    print(f"{Fore.GREEN}Running KGFlow {pipeline} pipeline on {input_path}...{Style.RESET_ALL}")
    try:
        subprocess.run(cmd, cwd=os.getcwd(), check=True)
    except subprocess.CalledProcessError as exc:
        print(f"{Fore.RED}Error: KGFlow pipeline failed with exit code {exc.returncode}.{Style.RESET_ALL}")
        return

    final_cache_path = _latest_pipeline_output(cache_dir, output_prefix)
    if final_cache_path is None:
        print(f"{Fore.RED}Error: no KGFlow output found in {cache_dir}{Style.RESET_ALL}")
        return

    _write_predictions_with_gold_fields(input_path, final_cache_path, output_path)
    print(f"{Fore.GREEN}Done. Output saved to {output_path}{Style.RESET_ALL}")


def cli_json2kg_eval(method: str, dataset: str, metric: str):
    if metric != "coverage":
        print(f"{Fore.RED}Error: unsupported metric '{metric}'. Only 'coverage' is available for now.{Style.RESET_ALL}")
        return

    gold_path = _require_jsonl_dataset(dataset)
    if gold_path is None:
        return

    pred_path = _method_output_path(method)
    if not pred_path.exists():
        print(f"{Fore.RED}Error: {pred_path} not found. Run 'text2kg run {method} {dataset}' first.{Style.RESET_ALL}")
        return

    if not os.getenv("EVALUATOR_BASE_URL") and os.getenv("DF_BASE_URL"):
        os.environ["EVALUATOR_BASE_URL"] = os.getenv("DF_BASE_URL", "")

    try:
        records = _load_json_records(gold_path)
        predictions = _load_json_records(pred_path)
    except Exception as exc:
        print(f"{Fore.RED}Error: failed to load evaluation inputs: {exc}{Style.RESET_ALL}")
        return

    pred_by_id = {}
    for index, pred in enumerate(predictions):
        item_id = str(pred.get("id") or f"item_{index:06d}")
        pred_by_id[item_id] = pred

    aligned = []
    missing = 0
    for index, gold in enumerate(records):
        item_id = str(gold.get("id") or f"item_{index:06d}")
        pred = pred_by_id.get(item_id)
        if pred is None:
            missing += 1
            pred_triples = []
        else:
            pred_triples = _normalize_triples(pred.get("triple") or pred.get("triples") or [])

        aligned.append(
            {
                "id": item_id,
                "title": gold.get("title"),
                "text": gold.get("text", ""),
                "url": gold.get("url"),
                "source": gold.get("source"),
                "domain": gold.get("domain"),
                "metadata": gold.get("metadata", {}),
                "relational_facts": gold.get("relational_facts", []),
                "triple": pred_triples,
                "baseline": method,
            }
        )

    eval_input = Path(os.getcwd()) / f"{method}_{dataset}_{metric}_input.json"
    report_path = Path(os.getcwd()) / f"{method}_{dataset}_{metric}_report.json"
    eval_input.write_text(json.dumps(aligned, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{Fore.GREEN}Running coverage evaluation for {method} on {dataset}...{Style.RESET_ALL}")
    if missing:
        print(f"{Fore.YELLOW}Warning: {missing} gold item(s) have no matching prediction and will be scored with an empty KG.{Style.RESET_ALL}")

    try:
        module = _import_from_json2kg("evaluate_general_entailment")
        module.evaluate_dataset(
            dataset_path=str(eval_input),
            output_path=str(report_path),
            retrieval_k=8,
            retrieval_depth=2,
            retrieval_model=os.getenv("EVALUATOR_RETRIEVAL_MODEL", "all-MiniLM-L6-v2"),
            dataset_name=dataset,
            evaluator_profile=os.getenv("EVALUATOR_PROFILE", "auto"),
        )
    except Exception as exc:
        print(f"{Fore.RED}Error: evaluation failed: {exc}{Style.RESET_ALL}")
        return

    print(f"{Fore.GREEN}Done. Report saved to {report_path}{Style.RESET_ALL}")

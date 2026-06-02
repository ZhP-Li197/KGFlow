import argparse
import os
from pathlib import Path

from dataflow.serving.api_llm_serving_request import APILLMServing_request
from dataflow.utils.storage import FileStorage


def normalize_chat_api_url(api_url: str) -> str:
    api_url = api_url.rstrip("/")
    if api_url.endswith("/chat/completions"):
        return api_url
    if api_url.endswith("/v1"):
        return f"{api_url}/chat/completions"
    return api_url


def add_common_args(parser: argparse.ArgumentParser, default_cache: str, default_prefix: str) -> argparse.ArgumentParser:
    parser.add_argument("--input", required=True, help="Input JSON/JSONL file with a text field.")
    parser.add_argument("--cache-dir", default=default_cache, help="Directory for pipeline step outputs.")
    parser.add_argument("--output-prefix", default=default_prefix, help="Prefix for cached output files.")
    parser.add_argument("--model-name", default=os.getenv("MODEL_NAME", "gpt-4o-mini"))
    parser.add_argument("--api-url", default=os.getenv("DF_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key-env", default="DF_API_KEY")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--input-key", default="text")
    return parser


def build_storage(args: argparse.Namespace) -> FileStorage:
    input_path = Path(args.input)
    cache_type = "jsonl" if input_path.suffix.lower() == ".jsonl" else "json"
    return FileStorage(
        first_entry_file_name=str(input_path),
        cache_path=args.cache_dir,
        file_name_prefix=args.output_prefix,
        cache_type=cache_type,
    )


def build_llm_serving(args: argparse.Namespace) -> APILLMServing_request:
    return APILLMServing_request(
        api_url=normalize_chat_api_url(args.api_url),
        key_name_of_api_key=args.api_key_env,
        model_name=args.model_name,
        max_workers=args.max_workers,
    )


def latest_step_path(args: argparse.Namespace, step: int) -> Path:
    input_path = Path(args.input)
    cache_type = "jsonl" if input_path.suffix.lower() == ".jsonl" else "json"
    return Path(args.cache_dir) / f"{args.output_prefix}_step{step}.{cache_type}"

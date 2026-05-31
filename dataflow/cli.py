#!/usr/bin/env python3
# dataflow/cli.py
# ===============================================================
# DataFlow 命令行入口
#   text2kg -v                         查看版本并检查更新
#   text2kg init [...]                初始化脚本/配置
#   text2kg env                       查看环境
#   text2kg load <dataset>            加载数据集
#   text2kg run <model> <dataset>     运行模型
# ===============================================================

import os
import argparse
import requests
from colorama import init as color_init, Fore, Style
from dataflow.cli_funcs import cli_env, cli_init, cli_json2kg_init, cli_json2kg_autoschemakg, cli_json2kg_wikontic
from dataflow.version import __version__

color_init(autoreset=True)
PYPI_API_URL = "https://pypi.org/pypi/dataflow-kg/json"


# ---------------- 版本检查 ----------------
def version_and_check_for_updates() -> None:
    width = os.get_terminal_size().columns
    print(Fore.BLUE + "=" * width + Style.RESET_ALL)
    print(f"dataflow-kg codebase version: {__version__}")

    try:
        r = requests.get(PYPI_API_URL, timeout=5)
        r.raise_for_status()
        remote = r.json()["info"]["version"]
        print("\tChecking for updates...")
        print(f"\tLocal version : {__version__}")
        print(f"\tPyPI  version : {remote}")
        if remote != __version__:
            print(Fore.YELLOW + f"New version available: {remote}."
                                "  Run 'pip install -U dataflow-kg' to upgrade."
                  + Style.RESET_ALL)
        else:
            print(Fore.GREEN + f"You are using the latest version: {__version__}" + Style.RESET_ALL)
    except requests.exceptions.RequestException as e:
        print(Fore.RED + "Failed to query PyPI – check your network." + Style.RESET_ALL)
        print("Error:", e)
    print(Fore.BLUE + "=" * width + Style.RESET_ALL)


# ---------------- CLI 主函数 ----------------
def build_arg_parser() -> argparse.ArgumentParser:
    """构建参数解析器"""
    parser = argparse.ArgumentParser(
        prog="text2kg",
        description=f"DataFlow-KG Command-Line Interface  (v{__version__})",
    )
    parser.add_argument("-v", "--version", action="store_true", help="Show version and exit")

    # ============ 顶层子命令 ============ #
    top = parser.add_subparsers(dest="command", required=False)

    # --- init ---
    p_init = top.add_parser("init", help="Initialize scripts/configs in current dir")
    p_init_sub = p_init.add_subparsers(dest="subcommand", required=False)
    p_init_sub.add_parser("all", help="Init all components").set_defaults(subcommand="all")
    p_init_sub.add_parser("reasoning", help="Init reasoning components").set_defaults(subcommand="reasoning")

    # --- env ---
    top.add_parser("env", help="Show environment information")

    # --- load ---
    p_load = top.add_parser("load", help="Load a dataset into current directory")
    p_load.add_argument("dataset", help="Dataset name to load (e.g. input)")

    # --- run ---
    p_run = top.add_parser("run", help="Run a model on a dataset")
    p_run.add_argument("model", choices=["autoschemakg", "wikontic"], help="Model to run")
    p_run.add_argument("dataset", help="Dataset name to run (e.g. input)")

    return parser


def main() -> None:
    """主入口函数"""
    parser = build_arg_parser()
    args = parser.parse_args()

    # ---------- 顶层逻辑分发 ----------
    if args.version:
        version_and_check_for_updates()
        return

    if args.command == "init":
        cli_init(subcommand=args.subcommand or "base")

    elif args.command == "env":
        cli_env()

    elif args.command == "load":
        cli_json2kg_init(dataset=args.dataset)

    elif args.command == "run":
        if args.model == "autoschemakg":
            cli_json2kg_autoschemakg(dataset=args.dataset)
        elif args.model == "wikontic":
            cli_json2kg_wikontic(dataset=args.dataset)


if __name__ == "__main__":
    main()

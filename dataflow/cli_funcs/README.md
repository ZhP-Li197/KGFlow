# One-Click KG Extraction CLI

This directory contains the lightweight command-line wrappers used by `text2kg`.

## Setup

Install the package in editable mode from the repository root:

```powershell
pip install -e .
```

Set the API environment variables:

```powershell
$env:DF_API_KEY="your_api_key"
$env:DF_BASE_URL="https://your-openai-compatible-endpoint/v1"
$env:MODEL_NAME="gpt-4o-mini"
```

## Load A Benchmark Dataset

`text2kg load` downloads a JSONL dataset from `b1u1/KGFlow-bench` into the current working directory.

```powershell
text2kg load WikiGeneral
```

Available benchmark files include:

```text
WikiGeneral
WelleGeneral
Temporal
Medical
Finance
Legal
```

## Run A Baseline

Each method reads the loaded `{dataset}.jsonl` file and extracts triples from the `text` field.

```powershell
text2kg run kggen WikiGeneral
text2kg run autoschemakg WikiGeneral
text2kg run wikontic WikiGeneral
text2kg run treekg WikiGeneral
```

The normalized prediction files are written to the current directory:

```text
kggen_output.json
autoschemakg_output.json
wikontic_output.json
treekg_output.json
```

## Run KGFlow Pipelines

KGFlow pipelines are selected explicitly with `--pipeline`.

```powershell
text2kg run kgflow WikiGeneral --pipeline general
text2kg run kgflow Finance --pipeline finance
text2kg run kgflow Medical --pipeline medical
text2kg run kgflow Legal --pipeline legal
text2kg run kgflow Temporal --pipeline temporal
```

The normalized KGFlow prediction file is:

```text
kgflow_output.json
```

## Evaluate Coverage

Coverage evaluation checks whether the gold `relational_facts` are entailed by the extracted KG.

```powershell
text2kg eval kggen WikiGeneral coverage
text2kg eval autoschemakg WikiGeneral coverage
text2kg eval wikontic WikiGeneral coverage
text2kg eval treekg WikiGeneral coverage
text2kg eval kgflow WikiGeneral coverage
```

By default, the evaluator reuses `DF_BASE_URL` as `EVALUATOR_BASE_URL`. You can override evaluator settings:

```powershell
$env:EVALUATOR_BASE_URL="https://your-openai-compatible-endpoint/v1"
$env:EVALUATOR_API_KEY="your_api_key"
$env:EVALUATOR_MODEL="gpt-4o-mini"
$env:EVALUATOR_RETRIEVAL_MODEL="all-MiniLM-L6-v2"
```

## Minimal Workflow

```powershell
cd D:\Programing\KGFlow\kgflow-test
text2kg load WikiGeneral
text2kg run kggen WikiGeneral
text2kg eval kggen WikiGeneral coverage
```

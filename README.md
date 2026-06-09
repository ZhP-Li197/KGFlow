# KGFlow: A Unified Operator-Centric System for Automated Knowledge Graph Construction

KGFlow is a unified operator-centric system for automated text-to-knowledge-graph construction. It decomposes KG construction into reusable operators and supports reproducible benchmarking across KGFlow pipelines and representative text-to-KG baselines.

KGFlow aims to make text-to-KG construction easier to use, easier to extend, and easier to evaluate.

---

## Overview

Knowledge graph construction from text usually involves multiple stages, such as entity extraction, relation extraction, tuple refinement, quality evaluation, and filtering. Existing systems are often tied to fixed schemas, specific fact representations, or manually engineered workflows.

KGFlow addresses this problem by organizing KG construction as an operator-centric workflow. A KG construction pipeline is built by composing reusable operators under a unified interface. This design enables KGFlow to support different construction scenarios, including general-purpose KG construction, domain-specific KG construction, temporal KG construction, and other structurally complex KG settings.

KGFlow also provides a simple command-line interface for loading benchmark datasets, running KG construction methods, and evaluating generated KGs.

---

## Key Features

### Unified Operator-Centric Workflow

KGFlow decomposes text-to-KG construction into reusable operators for generation, refinement, evaluation, and filtering. These operators can be composed into executable pipelines for different KG construction tasks.

### Automated KG Construction Pipelines

KGFlow supports pipeline-based KG construction through the `text2kg run` command. Users can select KGFlow pipelines or baseline methods with a unified command-line interface.

### One-Click Benchmarking

KGFlow standardizes the experimental workflow into three steps:

```bash
text2kg load [dataset_name]
text2kg run [method_name] [dataset_name]
text2kg eval [method_name] [dataset_name] [metric_name]
```

This makes it convenient to compare KGFlow with other text-to-KG construction methods under the same dataset and evaluation protocol.

### Integrated Baselines

KGFlow supports running multiple text-to-KG construction methods through the same interface, including:

* `kgflow`
* `kggen`
* `autoschemakg`
* `treekg`

---

## Installation

### 1. Clone or update the repository

If you have not cloned the repository yet, run:

```bash
git clone https://github.com/ZhP-Li197/KGFlow.git
cd KGFlow
```

If you have already cloned the repository, update it first:

```bash
git pull
```

### 2. Create and activate the environment

```bash
conda create -n text2kg python=3.10
conda activate text2kg
```

### 3. Install KGFlow in editable mode

```bash
pip install -e .
```

KGFlow requires Python 3.10 or later.

---

## Quickstart

The following example shows how to load the `WikiGeneral` dataset, run KGFlow and several baselines, and evaluate their factual coverage.

### 1. Check the environment

```bash
text2kg env
```

This command prints the current software and hardware environment, which is useful for verifying installation and reporting issues.

### 2. Load the benchmark dataset

```bash
text2kg load WikiGeneral
```

This command prepares the `WikiGeneral` dataset for later construction and evaluation.

### 3. Run KGFlow

```bash
text2kg run kgflow WikiGeneral --pipeline general
```

This command runs KGFlow on the `WikiGeneral` dataset using the `general` pipeline.

### 4. Run baseline methods

```bash
text2kg run kggen WikiGeneral
text2kg run autoschemakg WikiGeneral
text2kg run treekg WikiGeneral
```

These commands run the integrated baseline methods on the same dataset.

### 5. Evaluate factual coverage

```bash
text2kg eval kggen WikiGeneral coverage
text2kg eval autoschemakg WikiGeneral coverage
text2kg eval treekg WikiGeneral coverage
text2kg eval kgflow WikiGeneral coverage
```

The `coverage` metric evaluates how well the generated KG covers the factual information in the source text.

---

## Supported Example Methods

| Method       | Command Name   | Description                                      |
| ------------ | -------------- | ------------------------------------------------ |
| KGFlow       | `kgflow`       | Operator-centric KG construction pipeline        |
| KGGen        | `kggen`        | LLM-based KG generation baseline                 |
| AutoSchemaKG | `autoschemakg` | Automatic schema-guided KG construction baseline |
| TreeKG       | `treekg`       | Tree-structured KG construction baseline         |

---

## Supported Example Dataset

| Dataset      | Command Name   | Description                                                                                      |
| ------------ | -------------- | ------------------------------------------------------------------------------------------------ |
| WikiGeneral  | `WikiGeneral`  | General-domain text-to-KG benchmark dataset derived from Wikipedia-style documents               |
| WelleGeneral | `WelleGeneral` | General-domain text-to-KG benchmark dataset derived from Deutsche Welle documents                |
| Temporal     | `Temporal`     | Temporal text-to-KG benchmark dataset derived from Wikipedia documents with temporal expressions |
| Medical      | `Medical`      | Biomedical text-to-KG benchmark dataset derived from BC5CDR                                      |
| Finance      | `Finance`      | Financial text-to-KG benchmark dataset derived from EDGAR filings                                |
| Legal        | `Legal`        | Legal text-to-KG benchmark dataset derived from LEDGAR contract clauses                          |

```

---

## Citation

If you use KGFlow in your research, please cite:

```bibtex
@misc{kgflow2026,
  title={KGFlow: A Unified Operator-Centric System for Automated Knowledge Graph Construction},
  author={Zhengpin Li and Wanpeng Tang and Xuemeng Liu and Xinyuan Liu and Runhao Zhao and Huanyao Zhang and Weinan E and Wentao Zhang},
  year={2026},
  howpublished={\url{https://github.com/ZhP-Li197/KGFlow}}
}
```

---

## License

KGFlow is released under the Apache License 2.0.

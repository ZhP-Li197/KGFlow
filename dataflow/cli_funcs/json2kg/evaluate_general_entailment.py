from dotenv import load_dotenv
import argparse
import dspy
import json
import networkx as nx
import numpy as np
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple


EVALUATOR_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = EVALUATOR_DIR.parent
KGGEN_DIR = EVALUATOR_DIR / "KGGen"
if str(KGGEN_DIR) not in sys.path:
    sys.path.insert(0, str(KGGEN_DIR))

for shadow_path in (str(EVALUATOR_DIR), str(PIPELINE_DIR)):
    while shadow_path in sys.path:
        sys.path.remove(shadow_path)

from src.kg_gen.kg_gen import KGGen

try:
    from dataflow_L2_tkg.convert_tkg_tuples_to_event_triples import parse_tuple
except ImportError:
    def parse_tuple(raw_tuple: str, title: str = "") -> Dict[str, Any]:
        return {
            "subject": "",
            "predicate": "",
            "object": "",
            "time": "",
            "extras": [],
        }


load_dotenv()

# Reuse the current project configuration
RAW_EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "gpt-4o-mini")
API_KEY = (
    os.getenv("EVALUATOR_API_KEY")
    or os.getenv("DF_API_KEY")
    or os.getenv("OPENAI_API_KEY")
)
BASE_URL = os.getenv("EVALUATOR_BASE_URL", os.getenv("BASE_URL", "https://api.openai.com/v1"))


def resolve_evaluator_model(model_name: str, base_url: str) -> str:
    if "/" in model_name:
        return model_name
    if "11434" in base_url or "ollama" in base_url.lower():
        return f"ollama/{model_name}"
    return model_name


def normalize_api_base(base_url: str) -> str:
    return base_url.rstrip("/").replace("/chat/completions", "")


BASE_URL = normalize_api_base(BASE_URL)
EVALUATOR_MODEL = resolve_evaluator_model(RAW_EVALUATOR_MODEL, BASE_URL)

lm = dspy.LM(
    model=EVALUATOR_MODEL,
    api_key=API_KEY,
    api_base=BASE_URL,
    temperature=0.1,
    max_tokens=4000,
)
dspy.configure(lm=lm)


RETRYABLE_ERROR_MARKERS = (
    "serviceunavailableerror",
    "rate limit",
    "ratelimiterror",
    "apitimeouterror",
    "timeout",
    "apiconnectionerror",
    "connection error",
    "openai_error",
    "503",
    "502",
    "504",
)

FACT_QUALITY_LEVELS = ["high", "medium", "low", "unknown"]


class RetryableEvaluationError(Exception):
    """Raised when the evaluation backend is temporarily unavailable."""


class GeneralEvaluateResponse(dspy.Signature):
    """Determine whether the context contains the information stated in the correct answer. Respond with 1 if yes, 0 if no."""

    context: str = dspy.InputField(desc="The context to evaluate")
    correct_answer: str = dspy.InputField(desc="The correct answer to check for")
    evaluation: int = dspy.OutputField(
        desc="1 if context contains the correct answer, 0 otherwise"
    )


class TemporalEvaluateResponse(dspy.Signature):
    """Determine whether the context contains the information stated in the correct answer. For temporal information, determine additionally whether the context preserves the correct timing and ordering of events described in the answer. Respond with 1 if yes, 0 if no."""

    context: str = dspy.InputField(desc="The context to evaluate")
    correct_answer: str = dspy.InputField(desc="The correct answer to check for")
    evaluation: int = dspy.OutputField(
        desc="1 if context contains the correct answer and preserves correct timing/ordering, 0 otherwise"
    )


class MedicalEvaluateResponse(dspy.Signature):
    """Determine whether the context contains the information stated in the correct answer about biomedical entities and their relationships. Respond with 1 if yes, 0 if no. Pay close attention to negation (e.g. 'does not') and directionality: a negated fact requires negation in the context, and 'A affects B' does not support 'B affects A'."""

    context: str = dspy.InputField(desc="The context to evaluate")
    correct_answer: str = dspy.InputField(desc="The correct answer to check for")
    evaluation: int = dspy.OutputField(
        desc="1 if context contains the correct answer, 0 otherwise"
    )


class FinanceEvaluateResponse(dspy.Signature):
    """Determine whether the context contains the information stated in the correct answer about financial and corporate entities, their reported figures, and their relationships. Respond with 1 if yes, 0 if no. Pay close attention to numerical values and dates (exact figures matter), directionality of corporate actions ('A acquired B' does not support 'B acquired A'), and exclusive qualifiers (e.g. 'only', 'all', '100%')."""

    context: str = dspy.InputField(desc="The context to evaluate")
    correct_answer: str = dspy.InputField(desc="The correct answer to check for")
    evaluation: int = dspy.OutputField(
        desc="1 if context contains the correct answer, 0 otherwise"
    )


class LegalEvaluateResponse(dspy.Signature):
    """Determine whether the context contains the information stated in the correct answer about legal provisions, contractual obligations, and the parties involved. Respond with 1 if yes, 0 if no. Pay close attention to negation and exceptions (e.g. 'no liability', 'except for'): a fact stating an exclusion or limitation requires the context to reflect the same restriction, not a general affirmative."""

    context: str = dspy.InputField(desc="The context to evaluate")
    correct_answer: str = dspy.InputField(desc="The correct answer to check for")
    evaluation: int = dspy.OutputField(
        desc="1 if context contains the correct answer, 0 otherwise"
    )


EVALUATOR_SIGNATURES = {
    "general": GeneralEvaluateResponse,
    "temporal": TemporalEvaluateResponse,
    "medical": MedicalEvaluateResponse,
    "finance": FinanceEvaluateResponse,
    "legal": LegalEvaluateResponse,
}


DATASET_EVALUATOR_PROFILES = {
    "wikigeneral": "general",
    "wellegeneral": "general",
    "temporal": "temporal",
    "medical": "medical",
    "finance": "finance",
    "legal": "legal",
}

class ResponseEvaluator(dspy.Module):
    def __init__(self, signature_cls=GeneralEvaluateResponse):
        super().__init__()
        self.evaluate = dspy.ChainOfThought(signature_cls)

    def forward(self, context, correct_answer):
        return self.evaluate(context=context, correct_answer=correct_answer)


evaluator = ResponseEvaluator()


def resolve_evaluator_profile(dataset_name: str = "", evaluator_profile: str = "auto") -> str:
    if evaluator_profile and evaluator_profile != "auto":
        return evaluator_profile

    normalized_name = Path(str(dataset_name)).stem.lower()
    return DATASET_EVALUATOR_PROFILES.get(normalized_name, "general")


def configure_evaluator(dataset_name: str = "", evaluator_profile: str = "auto") -> str:
    global evaluator
    profile = resolve_evaluator_profile(dataset_name, evaluator_profile)
    signature_cls = EVALUATOR_SIGNATURES.get(profile)
    if signature_cls is None:
        raise ValueError(
            f"Unsupported evaluator profile '{profile}'. "
            f"Choose from: auto, {', '.join(sorted(EVALUATOR_SIGNATURES))}"
        )
    evaluator = ResponseEvaluator(signature_cls)
    print(f"Evaluator prompt profile: {profile}")
    return profile


def gpt_evaluate_response(correct_answer: str, context: str) -> int:
    """Evaluate whether the retrieved context supports the fact."""
    result = evaluator.forward(context=context, correct_answer=correct_answer)
    return int(result.evaluation)


def is_retryable_service_error(error: Exception) -> bool:
    error_text = f"{type(error).__name__}: {error}".lower()
    return any(marker in error_text for marker in RETRYABLE_ERROR_MARKERS)


def evaluate_with_retry(
    correct_answer: str,
    context: str,
    max_retries: int = 3,
    base_delay_seconds: float = 2.0,
) -> int:
    for attempt in range(max_retries + 1):
        try:
            return gpt_evaluate_response(correct_answer, context)
        except Exception as error:
            if not is_retryable_service_error(error):
                raise

            if attempt >= max_retries:
                raise RetryableEvaluationError(str(error)) from error

            delay_seconds = base_delay_seconds * (2 ** attempt)
            print(
                f"    Evaluation service unavailable; retrying in {delay_seconds:.1f}s "
                f"({attempt + 1}/{max_retries})"
            )
            time.sleep(delay_seconds)

    raise RetryableEvaluationError("Evaluation backend remained unavailable after retries.")


def iter_fact_objects(item: Dict[str, Any]):
    for quality_level, facts in iter_fact_buckets(item.get("relational_facts")).items():
        for fact_obj in facts or []:
            yield quality_level, fact_obj


def iter_fact_buckets(relational_facts: Any) -> Dict[str, List[Dict[str, Any]]]:
    if isinstance(relational_facts, list):
        return {
            "unknown": [
                fact_obj if isinstance(fact_obj, dict) else {"fact": str(fact_obj)}
                for fact_obj in relational_facts
            ]
        }

    if isinstance(relational_facts, dict):
        return {
            quality_level: [
                fact_obj if isinstance(fact_obj, dict) else {"fact": str(fact_obj)}
                for fact_obj in relational_facts.get(quality_level, []) or []
            ]
            for quality_level in FACT_QUALITY_LEVELS
        }

    return {quality_level: [] for quality_level in FACT_QUALITY_LEVELS}


def parse_entity_field(entity_field: Any) -> List[str]:
    if isinstance(entity_field, list):
        return [str(entity).strip() for entity in entity_field if str(entity).strip()]

    if isinstance(entity_field, str):
        return [entity.strip() for entity in entity_field.split(",") if entity.strip()]

    return []


def is_meaningful_tuple_value(value: str) -> bool:
    return str(value).strip() not in {"", "NA"}


def extract_core_triples_and_tuple_metadata(
    item: Dict[str, Any],
) -> Tuple[List[List[str]], Dict[Tuple[str, str, str], List[Dict[str, Any]]]]:
    core_triples: List[List[str]] = []
    tuple_metadata_map: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}

    for raw_tuple in item.get("tuple", []) or []:
        if not isinstance(raw_tuple, str) or not raw_tuple.strip():
            continue

        parsed = parse_tuple(raw_tuple, item.get("title", ""))
        subject = str(parsed.get("subject", "")).strip()
        predicate = str(parsed.get("predicate", "")).strip()
        object_value = str(parsed.get("object", "")).strip()
        if not (subject and predicate and object_value):
            continue

        triple = [subject, predicate, object_value]
        core_triples.append(triple)

        metadata: Dict[str, Any] = {"raw_tuple": raw_tuple}
        if is_meaningful_tuple_value(parsed.get("time", "")):
            metadata["time"] = str(parsed["time"]).strip()

        extras = []
        for extra in parsed.get("extras", []) or []:
            extra_value = str(extra.get("value", "")).strip()
            extra_predicate = str(extra.get("predicate", "")).strip()
            if not extra_predicate or not is_meaningful_tuple_value(extra_value):
                continue
            extras.append(
                {
                    "predicate": extra_predicate,
                    "value": extra_value,
                }
            )
        if extras:
            metadata["extras"] = extras

        triple_key = (subject, predicate, object_value)
        tuple_metadata_map.setdefault(triple_key, []).append(metadata)

    return core_triples, tuple_metadata_map


def build_compatible_extracted_kg(item: Dict[str, Any]) -> Dict[str, Any]:
    relations = []
    for triple in item.get("triple", []) or []:
        if isinstance(triple, str):
            parsed = parse_baseline_triple_string(triple)
            if parsed is not None:
                relations.append(parsed)
            continue

        if not isinstance(triple, list) or len(triple) != 3:
            continue
        relations.append([str(triple[0]), str(triple[1]), str(triple[2])])

    tuple_core_triples, tuple_metadata_map = extract_core_triples_and_tuple_metadata(item)
    relations.extend(tuple_core_triples)

    unique_relations = []
    seen_relations = set()
    for relation in relations:
        relation_key = tuple(relation)
        if relation_key in seen_relations:
            continue
        seen_relations.add(relation_key)
        unique_relations.append(relation)

    entities = parse_entity_field(item.get("entity"))
    if not entities:
        entity_set = set()
        for subject, _, object_value in unique_relations:
            entity_set.add(subject)
            entity_set.add(object_value)
        entities = sorted(entity_set)

    if entities or unique_relations:
        entity_set = set(entities)
        for subject, _, object_value in unique_relations:
            entity_set.add(subject)
            entity_set.add(object_value)
        entities = sorted(entity_set)

    return {
        "entities": entities,
        "relations": unique_relations,
        "edges": sorted({relation[1] for relation in unique_relations}),
        "tuple_metadata_map": tuple_metadata_map,
        "stats": {
            "entity_count": len(entities),
            "relation_count": len(unique_relations),
            "edge_count": len({relation[1] for relation in unique_relations}),
            "triple_relation_count": len(
                {
                    tuple(map(str, triple))
                    for triple in item.get("triple", []) or []
                    if isinstance(triple, list) and len(triple) == 3
                }
            ),
            "tuple_relation_count": len({tuple(relation) for relation in tuple_core_triples}),
        },
    }


def parse_baseline_triple_string(triple: str) -> List[str] | None:
    text = triple.strip()
    if not (text.startswith("<subj>") and "<obj>" in text and "<rel>" in text):
        return None

    try:
        subject_part, rest = text.split("<obj>", 1)
        object_part, relation_part = rest.split("<rel>", 1)
    except ValueError:
        return None

    subject = subject_part.replace("<subj>", "", 1).strip()
    predicate = relation_part.strip()
    object_value = object_part.strip()
    if not (subject and predicate and object_value):
        return None
    return [subject, predicate, object_value]


def format_tuple_metadata(metadata_entries: List[Dict[str, Any]]) -> str:
    supplements = []
    for metadata in metadata_entries:
        parts = []
        if is_meaningful_tuple_value(metadata.get("time", "")):
            parts.append(f"Time: {metadata['time']}.")

        for extra in metadata.get("extras", []) or []:
            extra_predicate = str(extra.get("predicate", "")).replace("_", " ").strip()
            if extra_predicate.lower().startswith("attribute"):
                extra_predicate = extra_predicate[0].upper() + extra_predicate[1:]
            extra_value = str(extra.get("value", "")).strip()
            if not extra_predicate or not is_meaningful_tuple_value(extra_value):
                continue
            parts.append(f"{extra_predicate}: {extra_value}.")

        if parts:
            supplements.append(" ".join(parts))

    return " ".join(supplements)


def retrieve_context_with_tuple_metadata(
    kggen: KGGen,
    query: str,
    node_embeddings: Dict[str, np.ndarray],
    graph: nx.DiGraph,
    tuple_metadata_map: Dict[Tuple[str, str, str], List[Dict[str, Any]]],
    k: int,
    depth: int,
) -> Tuple[List[Tuple[str, float]], List[str], str]:
    model = kggen._parse_embedding_model()
    top_nodes = kggen.retrieve_relevant_nodes(query, node_embeddings, model, k)
    context_items: List[str] = []
    seen_context_items = set()

    def add_context_line(subject: str, predicate: str, object_value: str) -> None:
        line = f"{subject} {predicate} {object_value}."
        metadata_text = format_tuple_metadata(
            tuple_metadata_map.get((subject, predicate, object_value), [])
        )
        if metadata_text:
            line = f"{line} {metadata_text}"
        if line in seen_context_items:
            return
        seen_context_items.add(line)
        context_items.append(line)

    def explore_neighbors(current_node: str, current_depth: int) -> None:
        if current_depth > depth:
            return

        for neighbor in graph.neighbors(current_node):
            predicate = graph[current_node][neighbor]["relation"]
            add_context_line(current_node, predicate, neighbor)
            explore_neighbors(neighbor, current_depth + 1)

        for neighbor in graph.predecessors(current_node):
            predicate = graph[neighbor][current_node]["relation"]
            add_context_line(neighbor, predicate, current_node)
            explore_neighbors(neighbor, current_depth + 1)

    for node, _ in top_nodes:
        explore_neighbors(node, 1)

    context_text = " ".join(context_items)
    return top_nodes, context_items, context_text


def is_fact_scored(fact_obj: Dict[str, Any]) -> bool:
    return (
        "kg_entailment_score" in fact_obj
        and fact_obj["kg_entailment_score"] is not None
    )


def is_item_fully_scored(item: Dict[str, Any]) -> bool:
    facts_found = False
    for _, fact_obj in iter_fact_objects(item):
        facts_found = True
        if not is_fact_scored(fact_obj):
            return False
    return facts_found


def calculate_item_stats(item: Dict[str, Any]) -> Dict[str, Any]:
    total_facts = 0
    scored_facts = 0
    correct_facts = 0
    service_error_facts = 0
    error_facts = 0

    for _, fact_obj in iter_fact_objects(item):
        total_facts += 1
        status = fact_obj.get("evaluation_status")
        if status == "service_error":
            service_error_facts += 1
        elif status == "error":
            error_facts += 1

        if is_fact_scored(fact_obj):
            scored_facts += 1
            correct_facts += fact_obj["kg_entailment_score"]

    return {
        "total_facts": total_facts,
        "scored_facts": scored_facts,
        "correct_facts": correct_facts,
        "service_error_facts": service_error_facts,
        "error_facts": error_facts,
        "accuracy": correct_facts / scored_facts if scored_facts > 0 else 0,
    }


def calculate_overall_stats(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_facts = 0
    scored_facts = 0
    correct_facts = 0
    service_error_facts = 0
    error_facts = 0

    difficulty_stats = {
        "easy": {
            "all_facts": 0,
            "total": 0,
            "correct": 0,
            "service_error_facts": 0,
            "error_facts": 0,
        },
        "medium": {
            "all_facts": 0,
            "total": 0,
            "correct": 0,
            "service_error_facts": 0,
            "error_facts": 0,
        },
        "hard": {
            "all_facts": 0,
            "total": 0,
            "correct": 0,
            "service_error_facts": 0,
            "error_facts": 0,
        },
    }

    for item in dataset:
        for _, fact_obj in iter_fact_objects(item):
            total_facts += 1
            fact_difficulty = fact_obj.get("difficulty", "unknown")
            status = fact_obj.get("evaluation_status")

            if fact_difficulty in difficulty_stats:
                difficulty_stats[fact_difficulty]["all_facts"] += 1

            if status == "service_error":
                service_error_facts += 1
                if fact_difficulty in difficulty_stats:
                    difficulty_stats[fact_difficulty]["service_error_facts"] += 1
            elif status == "error":
                error_facts += 1
                if fact_difficulty in difficulty_stats:
                    difficulty_stats[fact_difficulty]["error_facts"] += 1

            if is_fact_scored(fact_obj):
                scored_facts += 1
                correct_facts += fact_obj["kg_entailment_score"]
                if fact_difficulty in difficulty_stats:
                    difficulty_stats[fact_difficulty]["total"] += 1
                    difficulty_stats[fact_difficulty]["correct"] += fact_obj[
                        "kg_entailment_score"
                    ]

    for difficulty in difficulty_stats:
        stats = difficulty_stats[difficulty]
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0

    return {
        "total_facts": total_facts,
        "scored_facts": scored_facts,
        "correct_facts": correct_facts,
        "service_error_facts": service_error_facts,
        "error_facts": error_facts,
        "overall_accuracy": correct_facts / scored_facts if scored_facts > 0 else 0,
        "difficulty_breakdown": difficulty_stats,
    }


def build_result_payload(
    dataset: List[Dict[str, Any]],
    retrieval_k: int,
    retrieval_depth: int,
    retrieval_model: str,
    evaluator_profile: str = "general",
) -> Dict[str, Any]:
    overall_stats = calculate_overall_stats(dataset)

    simplified_dataset = []
    for item in dataset:
        if "evaluation_stats" not in item:
            continue

        fact_results = []
        for quality_level, fact_obj in iter_fact_objects(item):
            fact_result = {
                "quality_level": quality_level,
                "fact": fact_obj.get("fact"),
                "difficulty": fact_obj.get("difficulty", "unknown"),
                "relational_type": fact_obj.get("relational_type"),
                "confidence": fact_obj.get("confidence"),
                "triples": fact_obj.get("triples", {}),
                "kg_entailment_score": fact_obj.get("kg_entailment_score"),
                "evaluation_status": fact_obj.get("evaluation_status"),
            }

            if fact_obj.get("corrected_fact") is not None:
                fact_result["corrected_fact"] = fact_obj.get("corrected_fact")

            if "retrieval_stats" in fact_obj:
                fact_result["retrieval_stats"] = fact_obj["retrieval_stats"]

            if "error" in fact_obj:
                fact_result["error"] = fact_obj["error"]

            fact_results.append(fact_result)

        simplified_dataset.append(
            {
                "id": item.get("id", "unknown"),
                "title": item.get("title", "unknown"),
                "evaluation_stats": item["evaluation_stats"],
                "fact_results": fact_results,
            }
        )

    return {
        "evaluation_config": {
            "retrieval_model": retrieval_model,
            "retrieval_k": retrieval_k,
            "retrieval_depth": retrieval_depth,
            "evaluator_model": EVALUATOR_MODEL,
            "evaluator_profile": evaluator_profile,
        },
        "dataset_stats": simplified_dataset,
        "overall_stats": overall_stats,
    }


def save_result_payload(
    dataset: List[Dict[str, Any]],
    output_path: str,
    retrieval_k: int,
    retrieval_depth: int,
    retrieval_model: str,
    evaluator_profile: str = "general",
) -> None:
    result = build_result_payload(dataset, retrieval_k, retrieval_depth, retrieval_model, evaluator_profile)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)


def _iter_previous_fact_results(previous_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(previous_item.get("fact_results"), list):
        return previous_item.get("fact_results", [])

    previous_fact_results: List[Dict[str, Any]] = []
    for quality_level, facts in iter_fact_buckets(previous_item.get("relational_facts")).items():
        for fact_obj in facts or []:
            if not isinstance(fact_obj, dict):
                continue
            previous_fact_results.append(
                {
                    "quality_level": quality_level,
                    "kg_entailment_score": fact_obj.get("kg_entailment_score"),
                    "evaluation_status": fact_obj.get("evaluation_status"),
                    "retrieval_stats": fact_obj.get("retrieval_stats"),
                    "error": fact_obj.get("error"),
                }
            )
    return previous_fact_results


def restore_progress_from_output(dataset: List[Dict[str, Any]], output_path: str) -> int:
    if not os.path.exists(output_path):
        return 0

    with open(output_path, "r", encoding="utf-8") as file:
        previous_result = json.load(file)

    if isinstance(previous_result, dict):
        previous_dataset = previous_result.get("dataset_stats", [])
    elif isinstance(previous_result, list):
        previous_dataset = previous_result
    else:
        return 0

    previous_items = {}
    for item in previous_dataset:
        if isinstance(item, dict):
            previous_items[item.get("id", "unknown")] = item

    restored_items = 0
    for item in dataset:
        item_id = item.get("id", "unknown")
        previous_item = previous_items.get(item_id)
        if previous_item is None:
            continue

        previous_fact_lists = {quality: [] for quality in FACT_QUALITY_LEVELS}
        for fact_result in _iter_previous_fact_results(previous_item):
            quality_level = fact_result.get("quality_level", "unknown")
            previous_fact_lists.setdefault(quality_level, []).append(fact_result)

        restored_any = False
        for quality_level, current_facts in iter_fact_buckets(item.get("relational_facts")).items():
            previous_facts = previous_fact_lists.get(quality_level, [])
            for index, fact_obj in enumerate(current_facts):
                if index >= len(previous_facts):
                    continue

                previous_fact = previous_facts[index]

                if previous_fact.get("error") is not None:
                    fact_obj["error"] = previous_fact["error"]
                    fact_obj["evaluation_status"] = previous_fact.get(
                        "evaluation_status",
                        "service_error",
                    )
                    continue

                if previous_fact.get("kg_entailment_score") is not None:
                    fact_obj["kg_entailment_score"] = previous_fact["kg_entailment_score"]
                    fact_obj["evaluation_status"] = previous_fact.get(
                        "evaluation_status",
                        "scored",
                    )
                    if "retrieval_stats" in previous_fact:
                        fact_obj["retrieval_stats"] = previous_fact["retrieval_stats"]
                    restored_any = True

        item["evaluation_stats"] = calculate_item_stats(item)
        if restored_any:
            restored_items += 1

    return restored_items


def evaluate_single_item(
    item: Dict[str, Any],
    kggen: KGGen,
    retrieval_k: int = 8,
    retrieval_depth: int = 2,
) -> Dict[str, Any]:
    """Evaluate all facts for a single item."""
    print(f"Evaluating item: {item.get('title', item.get('id', 'Unknown'))}")

    extracted_kg = build_compatible_extracted_kg(item)
    if not extracted_kg.get("relations"):
        print("  Skip: no KG relations available")
        return item

    try:
        graph = kggen.from_dict(extracted_kg)
        nx_graph = kggen.to_nx(graph)
        node_embeddings, _ = kggen.generate_embeddings(nx_graph)
        tuple_metadata_map = extracted_kg.get("tuple_metadata_map", {})

        print(f"  Graph: {nx_graph.number_of_nodes()} nodes, {nx_graph.number_of_edges()} edges")

        for quality_level, facts_list in iter_fact_buckets(item.get("relational_facts")).items():
            if not facts_list:
                continue

            label = quality_level if isinstance(item.get("relational_facts"), dict) else "coverage"
            print(f"  Evaluating {label}: {len(facts_list)} facts")

            for fact_obj in facts_list:
                if is_fact_scored(fact_obj):
                    continue

                fact_text = fact_obj["fact"]

                try:
                    top_nodes, context_edges, context_text = retrieve_context_with_tuple_metadata(
                        kggen=kggen,
                        query=fact_text,
                        node_embeddings=node_embeddings,
                        graph=nx_graph,
                        tuple_metadata_map=tuple_metadata_map,
                        k=retrieval_k,
                        depth=retrieval_depth,
                    )

                    fact_obj["retrieval_stats"] = {
                        "top_k": retrieval_k,
                        "depth": retrieval_depth,
                        "retrieved_node_count": len(top_nodes),
                        "retrieved_context_edge_count": len(context_edges),
                        "retrieved_context_char_length": len(context_text),
                        "retrieved_context_word_count": len(context_text.split()),
                    }

                    evaluation_result = evaluate_with_retry(fact_text, context_text)

                    fact_obj["kg_entailment_score"] = evaluation_result
                    fact_obj["retrieved_context"] = context_text
                    fact_obj["evaluation_status"] = "scored"
                    fact_obj.pop("error", None)

                except RetryableEvaluationError as error:
                    print(f"    Evaluation service error; not counted: {str(error)}")
                    fact_obj.pop("kg_entailment_score", None)
                    fact_obj["evaluation_status"] = "service_error"
                    fact_obj["error"] = str(error)

                except Exception as error:
                    print(f"    Error while evaluating fact; not counted: {str(error)}")
                    fact_obj.pop("kg_entailment_score", None)
                    fact_obj["evaluation_status"] = "error"
                    fact_obj["error"] = str(error)

        item["evaluation_stats"] = calculate_item_stats(item)
        print(
            "  Result: "
            f"{item['evaluation_stats']['correct_facts']}/"
            f"{item['evaluation_stats']['scored_facts']} = "
            f"{item['evaluation_stats']['accuracy'] * 100:.1f}% "
            f"(service errors {item['evaluation_stats']['service_error_facts']}, "
            f"other errors {item['evaluation_stats']['error_facts']})"
        )

    except Exception as error:
        print(f"  Error while processing KG: {str(error)}")
        item["evaluation_error"] = str(error)

    return item

def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def build_progress_bar(current: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "-" * width
    filled = int(width * current / total)
    filled = min(width, max(0, filled))
    return "#" * filled + "-" * (width - filled)


def evaluate_dataset(
    dataset_path: str,
    output_path: str,
    retrieval_k: int = 8,
    retrieval_depth: int = 2,
    retrieval_model: str = "all-MiniLM-L6-v2",
    dataset_name: str = "",
    evaluator_profile: str = "auto",
) -> None:
    """Evaluate the whole dataset and checkpoint progress to the output file."""
    print(f"Starting evaluation dataset: {dataset_path}")
    active_profile = configure_evaluator(dataset_name or dataset_path, evaluator_profile)

    with open(dataset_path, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    restored_items = restore_progress_from_output(dataset, output_path)
    if restored_items > 0:
        print(f"Restored progress for {restored_items} completed items")

    kggen = KGGen(retrieval_model=retrieval_model)
    print(f"Dataset contains {len(dataset)} items")
    start_time = time.time()
    completed_items = 0

    for index, item in enumerate(dataset):
        print(f"\nProcessing item {index + 1}/{len(dataset)}")
        if is_item_fully_scored(item):
            item["evaluation_stats"] = calculate_item_stats(item)
            print(f"  Skip: {item.get('title', item.get('id', 'Unknown'))} already completed")
        else:
            dataset[index] = evaluate_single_item(
                item,
                kggen,
                retrieval_k=retrieval_k,
                retrieval_depth=retrieval_depth,
            )

        completed_items += 1
        save_result_payload(dataset, output_path, retrieval_k, retrieval_depth, retrieval_model, active_profile)
        elapsed_seconds = time.time() - start_time
        average_seconds_per_item = (
            elapsed_seconds / completed_items if completed_items > 0 else 0.0
        )
        remaining_items = len(dataset) - completed_items
        eta_seconds = average_seconds_per_item * remaining_items
        progress_bar = build_progress_bar(completed_items, len(dataset))
        progress_percent = (completed_items / len(dataset)) * 100 if dataset else 0.0
        print(
            "  Progress: "
            f"[{progress_bar}] {completed_items}/{len(dataset)} "
            f"({progress_percent:.1f}%) "
            f"elapsed={format_duration(elapsed_seconds)} "
            f"eta={format_duration(eta_seconds)}"
        )

    overall_stats = calculate_overall_stats(dataset)
    save_result_payload(dataset, output_path, retrieval_k, retrieval_depth, retrieval_model, active_profile)

    print(f"\nEvaluation finished. Results saved to: {output_path}")
    print(
        "Overall result: "
        f"{overall_stats['correct_facts']}/{overall_stats['scored_facts']} = "
        f"{overall_stats['overall_accuracy']:.1%}"
    )
    print(
        "Unscored facts: "
        f"service errors {overall_stats['service_error_facts']}, "
        f"other errors {overall_stats['error_facts']}"
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate KG extraction quality by fact entailment.")
    parser.add_argument(
        "--dataset",
        default="dataflow_experiment/l2_core_kg_extraction_gpt-4o-mini_step4.json",
        help="Dataset file path.",
    )
    parser.add_argument(
        "--output",
        default="dataflow_experiment/final_L2_evaluation_results_pipe3.json",
        help="Output report file path.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="Number of top-k relevant nodes used for retrieval.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="Graph-context expansion depth from retrieved nodes.",
    )

    parser.add_argument(
        "--retrieval-model",
        default=os.getenv("EVALUATOR_RETRIEVAL_MODEL", "all-MiniLM-L6-v2"),
        help="Sentence-transformers model name or local path",
    )
    parser.add_argument(
        "--dataset-name",
        default="",
        help="Dataset name used to auto-select the evaluator prompt profile.",
    )
    parser.add_argument(
        "--evaluator-profile",
        choices=["auto", "general", "temporal", "medical", "finance", "legal"],
        default=os.getenv("EVALUATOR_PROFILE", "auto"),
        help="Evaluator prompt profile. Defaults to auto, inferred from dataset name.",
    )

    args = parser.parse_args()

    evaluate_dataset(
        args.dataset,
        args.output,
        retrieval_k=args.k,
        retrieval_depth=args.depth,
        retrieval_model=args.retrieval_model,
        dataset_name=args.dataset_name or args.dataset,
        evaluator_profile=args.evaluator_profile,
    )


if __name__ == "__main__":
    main()

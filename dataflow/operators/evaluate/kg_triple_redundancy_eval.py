"""
====================================
DataFlow-KG:
====================================

Author: OpenAI Codex
Created: 2026-05-28

License:
    MIT License
"""

import json
from difflib import SequenceMatcher
from typing import Any, Dict, List, Set, Tuple, Union

from tqdm import tqdm

from dataflow import get_logger
from dataflow.core import LLMServingABC, OperatorABC
from dataflow.core.prompt import DIYPromptABC, prompt_restrict
from dataflow.prompts.core_kg.triple_eval import KGTripleRedundancyScoringPrompt
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage


@prompt_restrict(KGTripleRedundancyScoringPrompt)
@OPERATOR_REGISTRY.register()
class KGTripleRedundancyEvaluator(OperatorABC):
    def __init__(
        self,
        llm_serving: LLMServingABC,
        lang: str = "en",
        char_threshold: float = 0.75,
        prompt_template: Union[KGTripleRedundancyScoringPrompt, DIYPromptABC] = None,
    ):
        super().__init__()
        self.logger = get_logger()

        if not isinstance(llm_serving, LLMServingABC):
            raise TypeError("llm_serving must be LLMServingABC")

        self.llm_serving = llm_serving
        self.lang = lang
        self.char_threshold = char_threshold
        self.prompt_template = (
            prompt_template if prompt_template else KGTripleRedundancyScoringPrompt(lang=self.lang)
        )

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "KGTripleRedundancyEvaluator 用于评估一条数据内部三元组的冗余程度。",
                "它先用字符相似度聚成候选组，再用 LLM 对组内三元组打 0 到 1 的冗余分。",
                "输出包括全局冗余度和逐三元组冗余分数。",
            )
        return (
            "KGTripleRedundancyEvaluator evaluates redundancy among triples within each record.",
            "It first groups candidate duplicates by character similarity and then scores each triple via an LLM.",
            "Outputs include global redundancy rate and per-triple redundancy scores.",
        )

    def _normalize_text(self, text: str) -> str:
        return " ".join(str(text).replace("\n", " ").strip().split()).lower()

    def _parse_triples_with_indices(self, value: Any) -> List[Tuple[int, List[str]]]:
        triples: List[Tuple[int, List[str]]] = []
        if value is None:
            return triples

        if isinstance(value, list):
            for raw_idx, item in enumerate(value):
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    triples.append((raw_idx, [str(item[0]), str(item[1]), str(item[2])]))
                elif isinstance(item, str):
                    parts = [p.strip() for p in item.split("<rel>")]
                    if len(parts) == 2 and "<obj>" in parts[0]:
                        obj = parts[0].split("<obj>", 1)
                        if len(obj) == 2:
                            subj = obj[0].strip()
                            tail = obj[1].strip()
                            rel = parts[1].split("<time>", 1)[0].strip()
                            triples.append((raw_idx, [subj, rel, tail]))
        return triples

    def _triple_to_str(self, triple: List[str]) -> str:
        return f"{self._normalize_text(triple[0])} | {self._normalize_text(triple[1])} | {self._normalize_text(triple[2])}"

    def _build_similarity_groups(self, triples: List[List[str]]) -> List[List[int]]:
        n = len(triples)
        if n == 0:
            return []
        triple_strings = [self._triple_to_str(triple) for triple in triples]
        adj: List[Set[int]] = [set() for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                sim = SequenceMatcher(None, triple_strings[i], triple_strings[j]).ratio()
                if sim >= self.char_threshold:
                    adj[i].add(j)
                    adj[j].add(i)

        visited = [False] * n
        groups: List[List[int]] = []
        for i in range(n):
            if visited[i]:
                continue
            stack = [i]
            visited[i] = True
            component = []
            while stack:
                node = stack.pop()
                component.append(node)
                for nxt in adj[node]:
                    if not visited[nxt]:
                        visited[nxt] = True
                        stack.append(nxt)
            if len(component) > 1:
                groups.append(sorted(component))
        return groups

    def _safe_parse_json(self, response: str) -> Dict[str, Any]:
        clean = (response or "").strip()
        if "<answer>" in clean and "</answer>" in clean:
            clean = clean.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
        clean = clean.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(clean)
        except Exception:
            return {"results": []}

    def _normalize_score(self, value: Any) -> float:
        try:
            score = round(float(value), 2)
        except Exception:
            score = 0.0
        return min(1.0, max(0.0, score))

    def _judge_group(self, triples: List[List[str]]) -> List[float]:
        user_prompt = self.prompt_template.build_prompt(triples)
        system_prompt = self.prompt_template.build_system_prompt()
        response = self.llm_serving.generate_from_input(
            user_inputs=[user_prompt],
            system_prompt=system_prompt,
        )[0]
        parsed = self._safe_parse_json(response)
        results = parsed.get("results", [])
        scores = [0.0] * len(triples)
        if not isinstance(results, list):
            return scores
        for item in results:
            if not isinstance(item, dict):
                continue
            idx = item.get("local_idx")
            if not isinstance(idx, int) or idx < 0 or idx >= len(triples):
                continue
            scores[idx] = self._normalize_score(item.get("redundancy_score", 0.0))
        return scores

    def process_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        outputs = []
        for row in tqdm(records, desc="KG Triple Redundancy Eval"):
            raw_value = row.get("triple", [])
            parsed_with_idx = self._parse_triples_with_indices(raw_value)
            raw_len = len(raw_value) if isinstance(raw_value, list) else 0
            if not parsed_with_idx:
                outputs.append(
                    {
                        "global_redundancy_rate": 0.0,
                        "triple_redundancy_score": [],
                        "raw_triple_redundancy_score": [0.0] * raw_len,
                        "redundancy_groups": [],
                        "parsed_triples": [],
                        "parsed_to_raw_indices": [],
                    }
                )
                continue

            parsed_triples = [item[1] for item in parsed_with_idx]
            parsed_to_raw_indices = [item[0] for item in parsed_with_idx]
            scores = [0.0] * len(parsed_triples)
            groups = self._build_similarity_groups(parsed_triples)

            for group in groups:
                group_triples = [parsed_triples[idx] for idx in group]
                try:
                    group_scores = self._judge_group(group_triples)
                except Exception as exc:
                    self.logger.error(f"Redundancy evaluation failed: {exc}")
                    group_scores = [0.0] * len(group)
                for local_idx, global_idx in enumerate(group):
                    scores[global_idx] = max(scores[global_idx], group_scores[local_idx])

            rate = round(sum(scores) / len(scores), 4) if scores else 0.0
            raw_scores = [0.0] * raw_len
            raw_groups: List[List[int]] = []
            for group in groups:
                mapped = []
                for parsed_idx in group:
                    raw_idx = parsed_to_raw_indices[parsed_idx]
                    mapped.append(raw_idx)
                    if 0 <= raw_idx < raw_len:
                        raw_scores[raw_idx] = max(raw_scores[raw_idx], scores[parsed_idx])
                raw_groups.append(mapped)
            outputs.append(
                {
                    "global_redundancy_rate": rate,
                    "triple_redundancy_score": scores,
                    "raw_triple_redundancy_score": raw_scores,
                    "redundancy_groups": raw_groups,
                    "parsed_triples": parsed_triples,
                    "parsed_to_raw_indices": parsed_to_raw_indices,
                }
            )
        return outputs

    def run(
        self,
        storage: DataFlowStorage = None,
        input_key_meta: str = "triple",
        output_key_rate: str = "global_redundancy_rate",
        output_key_score: str = "triple_redundancy_score",
        output_key_groups: str = "redundancy_groups",
    ):
        if storage is None:
            raise ValueError("Storage required.")

        df = storage.read("dataframe")
        records = [{"triple": row.get(input_key_meta, [])} for _, row in df.iterrows()]
        outputs = self.process_batch(records)

        df[output_key_rate] = [item.get("global_redundancy_rate", 0.0) for item in outputs]
        df[output_key_score] = [item.get("triple_redundancy_score", []) for item in outputs]
        df[output_key_groups] = [item.get("redundancy_groups", []) for item in outputs]

        out_file = storage.write(df)
        self.logger.info(
            f"Avg Global Redundancy Rate: {df[output_key_rate].mean():.4f}. Saved to {out_file}"
        )
        return [output_key_rate, output_key_score, output_key_groups]

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
from typing import Any, Dict, List, Tuple, Union

from tqdm import tqdm

from dataflow import get_logger
from dataflow.core import LLMServingABC, OperatorABC
from dataflow.core.prompt import DIYPromptABC, prompt_restrict
from dataflow.prompts.core_kg.triple_eval import KGTripleHallucinationScoringPrompt
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage


@prompt_restrict(KGTripleHallucinationScoringPrompt)
@OPERATOR_REGISTRY.register()
class KGTripleHallucinationEvaluator(OperatorABC):
    def __init__(
        self,
        llm_serving: LLMServingABC,
        lang: str = "en",
        batch_size: int = 10,
        prompt_template: Union[KGTripleHallucinationScoringPrompt, DIYPromptABC] = None,
    ):
        super().__init__()
        self.logger = get_logger()

        if not isinstance(llm_serving, LLMServingABC):
            raise TypeError("llm_serving must be LLMServingABC")

        self.llm_serving = llm_serving
        self.lang = lang
        self.batch_size = batch_size
        self.prompt_template = (
            prompt_template if prompt_template else KGTripleHallucinationScoringPrompt(lang=self.lang)
        )

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "KGTripleHallucinationEvaluator 用于评估三元组相对原文的幻觉程度。",
                "它按批次调用 LLM，为每个三元组输出 0 到 1 的幻觉分数。",
                "输出包括整体幻觉率和逐三元组的幻觉分数列表。",
            )
        return (
            "KGTripleHallucinationEvaluator evaluates how hallucinated triples are with respect to source text.",
            "It queries an LLM in small batches and returns a 0-1 hallucination score for each triple.",
            "Outputs include overall hallucination rate and per-triple scores.",
        )

    def _normalize_text(self, text: str) -> str:
        return " ".join(str(text).replace("\n", " ").strip().split())

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
            score = 1.0
        return min(1.0, max(0.0, score))

    def _judge_batch(self, text: str, triples: List[List[str]]) -> List[float]:
        user_prompt = self.prompt_template.build_prompt(text, triples)
        system_prompt = self.prompt_template.build_system_prompt()
        response = self.llm_serving.generate_from_input(
            user_inputs=[user_prompt],
            system_prompt=system_prompt,
        )[0]
        parsed = self._safe_parse_json(response)
        results = parsed.get("results", [])
        scores = [1.0] * len(triples)
        if not isinstance(results, list):
            return scores
        for item in results:
            if not isinstance(item, dict):
                continue
            idx = item.get("idx")
            if not isinstance(idx, int) or idx < 0 or idx >= len(triples):
                continue
            scores[idx] = self._normalize_score(item.get("hallucination_score", 1.0))
        return scores

    def process_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        outputs = []
        for row in tqdm(records, desc="KG Triple Hallucination Eval"):
            text = row.get("text", row.get("raw_chunk", ""))
            raw_value = row.get("triple", [])
            parsed_with_idx = self._parse_triples_with_indices(raw_value)
            raw_len = len(raw_value) if isinstance(raw_value, list) else 0

            if not text or not parsed_with_idx:
                outputs.append(
                    {
                        "hallucination_rate": 0.0,
                        "triple_hallucination_score": [],
                        "raw_triple_hallucination_score": [0.0] * raw_len,
                        "parsed_triples": [],
                        "parsed_to_raw_indices": [],
                    }
                )
                continue

            parsed_triples = [item[1] for item in parsed_with_idx]
            parsed_to_raw_indices = [item[0] for item in parsed_with_idx]
            all_scores: List[float] = []
            for start in range(0, len(parsed_triples), self.batch_size):
                batch = parsed_triples[start : start + self.batch_size]
                try:
                    all_scores.extend(self._judge_batch(text, batch))
                except Exception as exc:
                    self.logger.error(f"Hallucination evaluation failed: {exc}")
                    all_scores.extend([1.0] * len(batch))

            rate = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0
            raw_scores = [0.0] * raw_len
            for raw_idx, score in zip(parsed_to_raw_indices, all_scores):
                if 0 <= raw_idx < raw_len:
                    raw_scores[raw_idx] = max(raw_scores[raw_idx], score)
            outputs.append(
                {
                    "hallucination_rate": rate,
                    "triple_hallucination_score": all_scores,
                    "raw_triple_hallucination_score": raw_scores,
                    "parsed_triples": parsed_triples,
                    "parsed_to_raw_indices": parsed_to_raw_indices,
                }
            )
        return outputs

    def run(
        self,
        storage: DataFlowStorage = None,
        input_key: str = "text",
        input_key_meta: str = "triple",
        output_key_rate: str = "hallucination_rate",
        output_key_score: str = "triple_hallucination_score",
    ):
        if storage is None:
            raise ValueError("Storage required.")

        df = storage.read("dataframe")
        records = [
            {
                "text": row.get(input_key, row.get("raw_chunk", "")),
                "triple": row.get(input_key_meta, []),
            }
            for _, row in df.iterrows()
        ]

        outputs = self.process_batch(records)
        df[output_key_rate] = [item.get("hallucination_rate", 0.0) for item in outputs]
        df[output_key_score] = [item.get("triple_hallucination_score", []) for item in outputs]

        out_file = storage.write(df)
        self.logger.info(
            f"Avg Hallucination Rate: {df[output_key_rate].mean():.4f}. Saved to {out_file}"
        )
        return [output_key_rate, output_key_score]

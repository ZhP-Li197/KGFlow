"""
非skill
====================================
DataFlow-KG: KGRelationTripleInference
====================================

Author: Zhengpin Li
Affiliation: Peking University
Email: zpli@pku.edu.cn
Created: 2026-01-27

License:
    MIT License
"""

from dataflow.prompts.core_kg.rel_triple_refinement import KGInferAndCheckTriplePrompt
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.core.prompt import prompt_restrict, DIYPromptABC

import random
from typing import Any, Dict, List, Optional, Union
from tqdm import tqdm
import json
import re


@prompt_restrict(
    KGInferAndCheckTriplePrompt
)
@OPERATOR_REGISTRY.register()
class KGRelationTripleInference(OperatorABC):
    """
    KGRelationInference performs triple-level reasoning
    to infer implicit relations from existing triples.

    The operator:
    - Takes observed relation triples as input
    - Infers additional plausible triples using an LLM
    - Optionally merges inferred triples back into original triples
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
        seed: int = 0,
        lang: str = "en",
        merge_to_input: bool = False,
    ):
        self.rng = random.Random(seed)
        self.llm_serving = llm_serving
        self.lang = lang
        self.merge_to_input = merge_to_input
        self.logger = get_logger()
        self.prompt_template = KGInferAndCheckTriplePrompt(lang=self.lang)

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "KGRelationTripleInference 用于基于已有三元组推理隐含的关系三元组。",
                "可选地将推理得到的三元组合并回原始输入，用于 KG 闭包扩展。"
            )
        else:
            return (
                "KGRelationTripleInference infers implicit triples from observed relation triples.",
                "Optionally merges inferred triples back into the original input."
            )

    def process_batch(
        self,
        triples: List[str],
        text: List[str],
        sources: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:

        if sources is None:
            sources = ["default_source"] * len(triples)
        elif len(sources) != len(triples):
            raise ValueError("Length of sources must match length of triples")

        results = []

        for i, triple in enumerate(tqdm(triples, desc="Inferring triples")):
            user_prompt = self.prompt_template.build_prompt(triple, text[i])

            system_prompt = self.prompt_template.build_system_prompt()

            responses = self.llm_serving.generate_from_input(
                user_inputs=[user_prompt],
                system_prompt=system_prompt,
            )
            print(responses)

            inferred = self._parse_llm_response(responses[0])
            results.append({"inferred_triple": inferred})

        return results

    def run(
        self,
        storage: DataFlowStorage = None,
        input_key: str = "triple",
        output_key: str = "inferred_triple",
        text_key: str = None,
    ):
        self.input_key = input_key
        self.output_key = output_key

        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        triples = dataframe[self.input_key].tolist()

        if text_key:
            resolved_text_key = text_key
        elif "raw_chunk" in dataframe.columns:
            resolved_text_key = "raw_chunk"
        elif "text" in dataframe.columns:
            resolved_text_key = "text"
        else:
            raise ValueError("Input dataframe must contain 'raw_chunk' or 'text' column.")
        text = dataframe[resolved_text_key].tolist()

        outputs = self.process_batch(triples, text)
        inferred_triples = [o[self.output_key] for o in outputs]

        # ===== 原始行为：只写 inferred_triple =====
        dataframe[self.output_key] = inferred_triples

        # ===== ⭐ 新增：是否合并回原 triple =====
        if self.merge_to_input:
            merged_triples = []
            for original, inferred in zip(triples, inferred_triples):
                # 保持顺序去重
                seen = set()
                merged = []
                for t in original + inferred:
                    if t not in seen:
                        seen.add(t)
                        merged.append(t)
                merged_triples.append(merged)

            dataframe[self.input_key] = merged_triples
            self.logger.info("Inferred triples merged back into input triples.")

        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [self.output_key]

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        if self.input_key not in dataframe.columns:
            raise ValueError(f"Missing required column: {self.input_key}")
        if self.output_key in dataframe.columns:
            raise ValueError(f"Column already exists: {self.output_key}")

    def _parse_llm_response(self, response: str) -> List[Any]:
        """
        Expected LLM output format:
        {
          "relations": [["subject", "relation", "object"], ...]
        }
        """
        try:
            json_str = re.search(r"\{.*\}", response, re.DOTALL).group()
            parsed = json.loads(json_str)
            if "relations" in parsed:
                triples = parsed.get("relations", [])
                return self._normalize_relation_triples(triples)

            # Backward compatibility for older prompt outputs.
            triples = parsed.get("inferred_triple", [])
            if isinstance(triples, list) and triples and isinstance(triples[0], str):
                return self._parse_tagged_relation_triples(triples)
            return self._normalize_relation_triples(triples)
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM response: {e}")
            return []

    def _normalize_relation_triples(self, triples: Any) -> List[List[str]]:
        if not isinstance(triples, list):
            return []

        normalized = []
        for triple in triples:
            if not isinstance(triple, list) or len(triple) != 3:
                continue
            if not all(isinstance(item, str) for item in triple):
                continue

            subject, relation, obj = (item.strip() for item in triple)
            if not subject or not relation or not obj:
                continue
            normalized.append([subject, relation, obj])

        return normalized

    def _parse_tagged_relation_triples(self, triples: List[str]) -> List[List[str]]:
        parsed_triples = []
        for triple in triples:
            if not isinstance(triple, str):
                continue

            match = re.match(
                r"<subj>\s*(.*?)\s*<obj>\s*(.*?)\s*<rel>\s*(.*)",
                triple.strip(),
            )
            if not match:
                continue

            subject, obj, relation = (part.strip() for part in match.groups())
            if not subject or not relation or not obj:
                continue
            parsed_triples.append([subject, relation, obj])

        return parsed_triples

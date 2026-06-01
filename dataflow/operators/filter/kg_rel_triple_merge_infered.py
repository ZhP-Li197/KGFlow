from typing import List

import pandas as pd

from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage


@OPERATOR_REGISTRY.register()
class KGRelationInferredTripleMerge(OperatorABC):
    """
    Merge inferred relation triples back into the original triple list.

    A candidate inferred triple is kept only if it passes all of the following:
    1. It is not an exact duplicate of an existing triple.
    2. It introduces a new entity or belongs to a target type
       (directional, ordinal, temporal, or qualifier).
    3. It does not conflict with an existing (subject, predicate) pair unless
       it is a target-type triple or the predicate is multi-valued.
    """

    _TEMPORAL_KW = [
        "in 19", "in 20",
        "on january", "on february", "on march", "on april", "on may", "on june",
        "on july", "on august", "on september", "on october", "on november", "on december",
        "founded", "born", "died", "retired", "established", "formed", "launched", "created",
    ]
    _QUALIFIER_KW = ["is a ", "was a ", "is the ", "was the ", "is an ", "was an "]
    _DIRECTIONAL_KW = [
        "north of", "south of", "east of", "west of",
        "northeast", "northwest", "southeast", "southwest",
        "lies west", "lies east", "lies north", "lies south",
        "just north", "just south", "just east", "just west",
        "adjacent to",
    ]
    _ORDINAL_KW = [
        "second largest", "third largest", "fourth largest",
        "largest ", "smallest ", "oldest ", "youngest ",
        "is the first", "was the first", "is the second", "was the second",
        "is the third", "most ", "least ",
    ]
    _MULTI_VALUE_PREDICATES = {
        "is a", "was a", "is an", "was an", "is the", "was the",
        "includes", "contains", "has", "borders", "passes through",
        "is known for", "is associated with", "published", "participated in",
    }

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "KGRelationInferredTripleMerge 将 inferred_triple 中有价值的关系三元组合并回 triple 字段。",
                "过滤规则包括去重、新实体或目标类型筛选，以及谓词冲突处理。",
            )
        return (
            "KGRelationInferredTripleMerge merges valuable inferred relation triples back into the triple field.",
            "Filters by deduplication, new-entity or target-type checks, and predicate conflict handling.",
        )

    def _is_target_type(self, triple: list) -> bool:
        s = " ".join(str(x).lower() for x in triple)
        if any(k in s for k in self._DIRECTIONAL_KW):
            return True
        if any(k in s for k in self._ORDINAL_KW):
            return True
        if any(k in s for k in self._TEMPORAL_KW):
            return True
        if any(k in s for k in self._QUALIFIER_KW) and len(str(triple[2]).strip()) > 5:
            return True
        return False

    def _merge_valuable_triples(self, triples: list, inferred: list) -> list:
        existing_entities: set = set()
        sp_to_objects: dict = {}
        seen: set = set()

        for t in triples:
            if not (isinstance(t, list) and len(t) == 3):
                continue
            s, p, o = str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()
            existing_entities.add(s)
            existing_entities.add(o)
            sp_to_objects.setdefault((s, p), set()).add(o)
            seen.add((s, p, o))

        merged = list(triples)
        for t in inferred or []:
            if not (isinstance(t, list) and len(t) == 3):
                continue
            s, p, o = str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()
            if not s or not p or not o:
                continue
            if (s, p, o) in seen:
                continue

            is_new_entity = s not in existing_entities or o not in existing_entities
            is_target = self._is_target_type(t)

            has_conflict = (
                (s, p) in sp_to_objects
                and o not in sp_to_objects[(s, p)]
                and p.strip().lower() not in self._MULTI_VALUE_PREDICATES
            )
            if has_conflict and not is_target:
                continue
            if not is_target and not is_new_entity:
                continue

            merged.append([s, p, o])
            seen.add((s, p, o))

        return merged

    def _validate_dataframe(self, dataframe: pd.DataFrame, triple_key: str, inferred_key: str):
        if triple_key not in dataframe.columns:
            raise ValueError(f"Missing required column: '{triple_key}'")
        if inferred_key not in dataframe.columns:
            raise ValueError(f"Missing required column: '{inferred_key}'")

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "triple",
        inferred_key: str = "inferred_triple",
        output_key: str = "triple",
    ) -> List[str]:
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe, input_key, inferred_key)

        dataframe[output_key] = [
            self._merge_valuable_triples(row[input_key], row[inferred_key])
            for _, row in dataframe.iterrows()
        ]

        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")
        return [output_key]

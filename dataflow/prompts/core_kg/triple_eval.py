import json
import textwrap

from dataflow.core.prompt import PromptABC
from dataflow.utils.registry import PROMPT_REGISTRY


@PROMPT_REGISTRY.register()
class KGTripleHallucinationScoringPrompt(PromptABC):
    def __init__(self, lang: str = "en"):
        self.lang = lang

    def build_system_prompt(self) -> str:
        if self.lang == "zh":
            return textwrap.dedent(
                """\
                你是一个严格的知识图谱事实评估器。

                任务：
                给定一段文本和若干三元组，判断每个三元组是否忠于文本内容。

                评分标准：
                - 0.00 表示完全忠于文本，没有幻觉
                - 1.00 表示明显幻觉、与文本不符或文本完全不支持
                - 中间值表示不确定、部分支持、表述过度推断等情况

                要求：
                1. 对每个三元组分别打分，保留两位小数。
                2. 不要修改三元组内容。
                3. 只输出 JSON，不要输出解释性文字。

                输出格式：
                {
                  "results": [
                    {"idx": 0, "hallucination_score": 0.12},
                    {"idx": 1, "hallucination_score": 0.88}
                  ]
                }
                """
            )
        return textwrap.dedent(
            """\
            You are a strict factuality evaluator for knowledge graph triples.

            Task:
            Given a source text and several triples, score how hallucinated each triple is.

            Scoring:
            - 0.00 means fully faithful to the text
            - 1.00 means clearly hallucinated, unsupported, or contradicted
            - Intermediate scores indicate partial support or uncertainty

            Requirements:
            1. Score each triple independently with exactly two decimals.
            2. Do not rewrite the triples.
            3. Return JSON only.

            Output format:
            {
              "results": [
                {"idx": 0, "hallucination_score": 0.12},
                {"idx": 1, "hallucination_score": 0.88}
              ]
            }
            """
        )

    def build_prompt(self, text: str, triples: list[list[str]]) -> str:
        triple_block = json.dumps(
            [
                {"idx": idx, "head": triple[0], "relation": triple[1], "tail": triple[2]}
                for idx, triple in enumerate(triples)
            ],
            ensure_ascii=False,
            indent=2,
        )
        if self.lang == "zh":
            return textwrap.dedent(
                f"""\
                请根据下面的文本，对每个三元组的幻觉程度打分。

                文本：
                {text}

                三元组：
                {triple_block}

                只返回 JSON。
                """
            )
        return textwrap.dedent(
            f"""\
            Score the hallucination level of each triple against the source text.

            Source text:
            {text}

            Triples:
            {triple_block}

            Return JSON only.
            """
        )


@PROMPT_REGISTRY.register()
class KGTripleRedundancyScoringPrompt(PromptABC):
    def __init__(self, lang: str = "en"):
        self.lang = lang

    def build_system_prompt(self) -> str:
        if self.lang == "zh":
            return textwrap.dedent(
                """\
                你是一个严格的知识图谱去重评估器。

                任务：
                给定一个候选三元组组，这些三元组经过字符相似度预筛选，可能表达相同事实。
                请判断组内每个三元组的冗余程度。

                评分标准：
                - 0.00 表示不冗余，应保留
                - 1.00 表示高度冗余，与组内其他三元组表达同一事实
                - 中间值表示部分重合、近义改写或冗余程度不完全确定

                要求：
                1. 对组内每个三元组分别打分，保留两位小数。
                2. 至少保留一个最不冗余的三元组，其分数应尽量低。
                3. 只输出 JSON。

                输出格式：
                {
                  "results": [
                    {"local_idx": 0, "redundancy_score": 0.05},
                    {"local_idx": 1, "redundancy_score": 0.91}
                  ]
                }
                """
            )
        return textwrap.dedent(
            """\
            You are a strict deduplication evaluator for knowledge graph triples.

            Task:
            You are given a candidate group of triples pre-filtered by character similarity.
            Score how redundant each triple is with respect to the others in the same group.

            Scoring:
            - 0.00 means not redundant and should be kept
            - 1.00 means highly redundant and expresses the same fact as others
            - Intermediate scores indicate partial overlap or uncertainty

            Requirements:
            1. Score each triple with exactly two decimals.
            2. Keep at least one least-redundant triple with a low score.
            3. Return JSON only.

            Output format:
            {
              "results": [
                {"local_idx": 0, "redundancy_score": 0.05},
                {"local_idx": 1, "redundancy_score": 0.91}
              ]
            }
            """
        )

    def build_prompt(self, triples: list[list[str]]) -> str:
        triple_block = json.dumps(
            [
                {"local_idx": idx, "head": triple[0], "relation": triple[1], "tail": triple[2]}
                for idx, triple in enumerate(triples)
            ],
            ensure_ascii=False,
            indent=2,
        )
        if self.lang == "zh":
            return textwrap.dedent(
                f"""\
                请判断下面候选组内每个三元组的冗余程度。

                候选三元组组：
                {triple_block}

                只返回 JSON。
                """
            )
        return textwrap.dedent(
            f"""\
            Score the redundancy of each triple in the following candidate group.

            Candidate group:
            {triple_block}

            Return JSON only.
            """
        )

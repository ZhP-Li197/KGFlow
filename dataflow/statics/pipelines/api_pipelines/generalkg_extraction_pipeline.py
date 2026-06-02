import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataflow.operators.refine import KGRelationTripleInference
from dataflow.operators.filter import KGTupleRemoveRepeated, KGTupleValidity
from dataflow.operators.generate import KGEntityExtraction, KGTripleExtraction

from _kgflow_pipeline_utils import add_common_args, build_llm_serving, build_storage, latest_step_path


class GeneralKGExtractionPipeline:
    def __init__(self, args):
        self.args = args
        self.storage = build_storage(args)
        self.llm_serving = build_llm_serving(args)
        self.entity_extractor = KGEntityExtraction(llm_serving=self.llm_serving, lang="en")
        self.triple_extractor = KGTripleExtraction(
            llm_serving=self.llm_serving,
            triple_type="relation",
            lang="en",
        )
        self.triple_inference = KGRelationTripleInference(
            llm_serving=self.llm_serving,
            lang="en",
            merge_to_input=True,
        )
        self.tuple_dedup = KGTupleRemoveRepeated(lang="en")
        self.tuple_validation = KGTupleValidity(
            llm_serving=self.llm_serving,
            lang="en",
            triple_type="relation",
        )

    def forward(self):
        self.entity_extractor.run(
            storage=self.storage.step(),
            input_key=self.args.input_key,
            output_key="entity",
        )
        self.triple_extractor.run(
            storage=self.storage.step(),
            input_key=self.args.input_key,
            input_key_meta="entity",
            output_key="triple",
        )
        self.triple_inference.run(
            storage=self.storage.step(),
            input_key="triple",
            output_key="inferred_triple",
        )
        self.tuple_dedup.run(
            storage=self.storage.step(),
            input_key="triple",
            output_key="triple",
        )
        self.tuple_validation.run(
            storage=self.storage.step(),
            input_key="triple",
            output_key="valid_triple",
        )
        return latest_step_path(self.args, 5)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the general KGFlow extraction pipeline.")
    return add_common_args(parser, default_cache="./kgflow_general", default_prefix="kgflow_general").parse_args()


def main():
    pipeline = GeneralKGExtractionPipeline(parse_args())
    output_path = pipeline.forward()
    print(f"KGFlow general output: {output_path.resolve()}")


if __name__ == "__main__":
    main()

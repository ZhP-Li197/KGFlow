import argparse
from dataflow.operators.generate import TKGTupleExtraction
from _kgflow_pipeline_utils import add_common_args, build_llm_serving, build_storage, latest_step_path


class TKGExtractionPipeline:
    def __init__(self, args):
        self.args = args
        self.storage = build_storage(args)
        self.llm_serving = build_llm_serving(args)
        self.extractor = TKGTupleExtraction(
            llm_serving=self.llm_serving,
            triple_type=args.triple_type,
            lang="en",
        )

    def forward(self):
        self.extractor.run(
            storage=self.storage.step(),
            input_key=self.args.input_key,
            output_key="tuple",
        )
        return latest_step_path(self.args, self.storage.operator_step + 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the temporal KGFlow extraction pipeline.")
    add_common_args(parser, default_cache="./kgflow_temporal", default_prefix="kgflow_temporal")
    parser.add_argument("--triple-type", choices=["relation", "attribute"], default="relation")
    return parser.parse_args()


def main():
    pipeline = TKGExtractionPipeline(parse_args())
    output_path = pipeline.forward()
    print(f"KGFlow temporal output: {output_path.resolve()}")


if __name__ == "__main__":
    main()

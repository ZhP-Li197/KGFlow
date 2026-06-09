import argparse
from dataflow.operators.generate import MedKGTripleExtraction
from _kgflow_pipeline_utils import add_common_args, build_llm_serving, build_storage, latest_step_path


class MedicalKGExtractionPipeline:
    def __init__(self, args):
        self.args = args
        self.storage = build_storage(args)
        self.llm_serving = build_llm_serving(args)
        self.extractor = MedKGTripleExtraction(
            llm_serving=self.llm_serving,
            triple_type=args.triple_type,
            lang="en",
        )

    def forward(self):
        self.extractor.run(
            storage=self.storage.step(),
            input_key=self.args.input_key,
            output_key="triple",
        )
        return latest_step_path(self.args, self.storage.operator_step + 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the medical KGFlow extraction pipeline.")
    add_common_args(parser, default_cache="./kgflow_medical", default_prefix="kgflow_medical")
    parser.add_argument("--triple-type", choices=["coverage", "relation"], default="coverage")
    return parser.parse_args()


def main():
    pipeline = MedicalKGExtractionPipeline(parse_args())
    output_path = pipeline.forward()
    print(f"KGFlow medical output: {output_path.resolve()}")


if __name__ == "__main__":
    main()

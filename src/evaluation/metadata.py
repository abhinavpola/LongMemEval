from openbench.utils import BenchmarkMetadata


def get_benchmark_suite() -> dict[str, BenchmarkMetadata]:
    """Return benchmark metadata for entry point registration."""
    return {
        "longmemeval-small": BenchmarkMetadata(
            name="LongMemEval (small)",
            description="A benchmark for evaluating the long-term memory of chat assistants",
            category="community",
            tags=["long-term memory", "chat assistants"],
            module_path="evaluation.evaluate_qa_inspect",
            function_name="evaluate_qa_inspect_small",
            is_alpha=False,
        ),
        "longmemeval-oracle": BenchmarkMetadata(
            name="LongMemEval (oracle)",
            description="A benchmark for evaluating the long-term memory of chat assistants",
            category="community",
            tags=["long-term memory", "chat assistants"],
            module_path="evaluation.evaluate_qa_inspect",
            function_name="evaluate_qa_inspect_oracle",
            is_alpha=False,
        )
    }

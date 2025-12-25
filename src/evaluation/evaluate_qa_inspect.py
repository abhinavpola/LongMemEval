import re
from inspect_ai import Task, task, task_with
from inspect_ai.dataset import (
    MemoryDataset,
    Sample,
)
from inspect_ai.model import Model, get_model
from inspect_ai.scorer import (
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState, generate
from huggingface_hub import hf_hub_download
from evaluation.get_judge_prompt import get_anscheck_prompt
import pandas as pd
from evaluation.solvers.in_context import in_context_setup
from evaluation.solvers.mem0 import mem0_cleanup, mem0_setup
from evaluation.solvers.supermemory import supermemory_cleanup, supermemory_setup
from evaluation.solvers.zep import zep_cleanup, zep_setup


@task
def evaluate_qa_inspect_small():
    return create_task("longmemeval_s_cleaned.json")


@task
def evaluate_qa_inspect_oracle():
    return create_task("longmemeval_oracle.json")


@task
def evaluate_qa_inspect_small_supermemory():
    return task_with(
        evaluate_qa_inspect_small(),
        setup=[
            supermemory_setup(only_answer_turns=False),
        ],
        solver=generate(),
        cleanup=supermemory_cleanup,
    )


@task
def evaluate_qa_inspect_oracle_supermemory():
    return task_with(
        evaluate_qa_inspect_oracle(),
        setup=[
            supermemory_setup(only_answer_turns=True),
        ],
        solver=generate(),
        cleanup=supermemory_cleanup,
    )


@task
def evaluate_qa_inspect_medium_supermemory():
    return create_supermemory_task(
        "longmemeval_m_cleaned.json",
        only_answer_turns=False,
    )


@task
def evaluate_qa_inspect_small_mem0():
    return task_with(
        evaluate_qa_inspect_small(),
        setup=mem0_setup(only_answer_turns=False),
        solver=generate(),
        cleanup=mem0_cleanup,
    )


@task
def evaluate_qa_inspect_oracle_mem0():
    return task_with(
        evaluate_qa_inspect_oracle(),
        setup=mem0_setup(only_answer_turns=True),
        solver=generate(),
        cleanup=mem0_cleanup,
    )


@task
def evaluate_qa_inspect_medium_mem0():
    return create_mem0_task(
        "longmemeval_m_cleaned.json",
        only_answer_turns=False,
    )


@task
def evaluate_qa_inspect_small_zep():
    return task_with(
        evaluate_qa_inspect_small(),
        setup=zep_setup(only_answer_turns=False),
        solver=generate(),
        cleanup=zep_cleanup,
    )


@task
def evaluate_qa_inspect_oracle_zep():
    return task_with(
        evaluate_qa_inspect_oracle(),
        setup=zep_setup(only_answer_turns=True),
        solver=generate(),
        cleanup=zep_cleanup,
    )


@task
def evaluate_qa_inspect_medium_zep():
    return create_zep_task(
        "longmemeval_m_cleaned.json",
        only_answer_turns=False,
    )


def create_task(filename: str) -> Task:
    memory_dataset = build_dataset(filename)
    return Task(
        dataset=memory_dataset,
        setup=in_context_setup(),
        solver=generate(),
        scorer=model_graded_qa(),
        name="longmemeval",
    )


def create_supermemory_task(filename: str, only_answer_turns: bool) -> Task:
    memory_dataset = build_dataset(filename)
    return Task(
        dataset=memory_dataset,
        setup=[
            supermemory_setup(only_answer_turns=only_answer_turns),
        ],
        solver=generate(),
        cleanup=supermemory_cleanup,
        scorer=model_graded_qa(),
        name="longmemeval",
    )


def create_mem0_task(filename: str, only_answer_turns: bool) -> Task:
    memory_dataset = build_dataset(filename)
    return Task(
        dataset=memory_dataset,
        setup=mem0_setup(only_answer_turns=only_answer_turns),
        solver=generate(),
        cleanup=mem0_cleanup,
        scorer=model_graded_qa(),
        name="longmemeval",
    )


def create_zep_task(filename: str, only_answer_turns: bool) -> Task:
    memory_dataset = build_dataset(filename)
    return Task(
        dataset=memory_dataset,
        setup=zep_setup(only_answer_turns=only_answer_turns),
        solver=generate(),
        cleanup=zep_cleanup,
        scorer=model_graded_qa(),
        name="longmemeval",
    )


def build_dataset(filename: str) -> MemoryDataset:
    raw_path = hf_hub_download(
        repo_id="xiaowu0162/longmemeval-cleaned",
        filename=filename,
        # We have to download the data directly to normalize the "answers" field since
        # it contains both string and numbers in this revision.
        revision="98d7416c24c778c2fee6e6f3006e7a073259d48f",
        repo_type="dataset",
    )

    df = pd.read_json(raw_path)
    df["answer"] = df["answer"].apply(str)

    memory_dataset = MemoryDataset(
        [
            Sample(
                input=row["question"],
                target=row["answer"],
                id=row["question_id"],
                metadata={
                    "question_type": row["question_type"],
                    "question_date": row["question_date"],
                    "haystack_session_ids": row["haystack_session_ids"],
                    "haystack_dates": row["haystack_dates"],
                    "haystack_sessions": row["haystack_sessions"],
                    "answer_session_ids": row["answer_session_ids"],
                },
            )
            for _, row in df.iterrows()
        ]
    )

    return memory_dataset


@scorer(metrics=[accuracy(), stderr()])
def model_graded_qa(
    model: str | Model | None = None,
) -> Scorer:
    grader_model = get_model(model)

    async def score(state: TaskState, target: Target) -> Score:
        score_prompt = get_anscheck_prompt(
            state.metadata["question_type"],
            state.input_text,
            target.text,
            state.output.completion,
        )

        result = await grader_model.generate(score_prompt)

        match = re.fullmatch(r"(?i)yes|no", result.completion.strip())
        if match:
            return Score(
                value=1 if "yes" in result.completion.lower() else 0,
                answer=state.output.completion,
                explanation=result.completion,
            )
        else:
            return Score(
                value=INCORRECT,
                answer=state.output.completion,
                explanation="Grade not found in model output: "
                + f"{result.completion}",
            )

    return score

import re
from inspect_ai import Task, task
from inspect_ai.dataset import (
    MemoryDataset,
    Sample,
)
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, Model, get_model
from inspect_ai.scorer import (
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import Generate, TaskState, solver
from huggingface_hub import hf_hub_download
from evaluation.get_judge_prompt import get_anscheck_prompt
import pandas as pd


@task
def evaluate_qa_inspect_small():
    return create_task("longmemeval_s_cleaned.json")


@task
def evaluate_qa_inspect_oracle():
    return create_task("longmemeval_oracle.json")


def create_task(filename: str) -> Task:
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

    return Task(
        dataset=memory_dataset,
        solver=[in_context_solver()],
        scorer=model_graded_qa(),
        name="longmemeval",
    )


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


@solver
def in_context_solver():
    async def solve(state: TaskState, generate: Generate):
        history = []
        for session in state.metadata["haystack_sessions"]:
            for turn in session:
                if turn["role"] == "user":
                    history.append(ChatMessageUser(content=turn["content"]))
                elif turn["role"] == "assistant":
                    history.append(ChatMessageAssistant(content=turn["content"]))

        state.messages = history + state.messages
        return await generate(state)

    return solve

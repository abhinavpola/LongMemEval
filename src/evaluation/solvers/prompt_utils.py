NORMALIZED_SYSTEM_PROMPT = (
    "You are a question-answering system. Use only the provided context. "
    'If the context does not contain enough information, respond with "I don\'t know". '
    "If timestamps are present in the context, use them to resolve relative time references."
)


def build_normalized_user_prompt(
    question: str,
    context: str,
    question_date: str | None = None,
) -> str:
    prompt = [f"Question: {question}"]
    if question_date:
        prompt.append(f"Question Date: {question_date}")
    prompt.extend(
        [
            "",
            "Context:",
            context,
            "",
            "Answer:",
        ]
    )
    return "\n".join(prompt)

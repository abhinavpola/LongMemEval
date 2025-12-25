import asyncio
import json
import os
import time

import httpx
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.solver import Generate, TaskState, solver
from evaluation.solvers.prompt_utils import (
    NORMALIZED_SYSTEM_PROMPT,
    build_normalized_user_prompt,
)
from mem0 import MemoryClient

MEM0_API_KEY = os.environ.get("MEM0_API_KEY", "")
MEM0_VERSION = "v2"
MEM0_EVENT_STATUS_URL = "https://api.mem0.ai/v1/event/{event_id}/"
MEM0_MAX_TOKENS = 80000
MEM0_APPROX_CHARS_PER_TOKEN = 4
MEM0_MAX_CHARS = MEM0_MAX_TOKENS * MEM0_APPROX_CHARS_PER_TOKEN


def _mem0_client() -> MemoryClient:
    return MemoryClient(api_key=MEM0_API_KEY or None)


def _build_mem0_answer_prompt(
    question: str, context: list[dict], question_date: str | None
) -> str:
    memories = []
    for idx, entry in enumerate(context):
        metadata = entry.get("metadata", {})
        timestamp = None
        if isinstance(metadata, dict):
            timestamp = metadata.get("date") or metadata.get("timestamp")
        timestamp_info = f" [Timestamp: {timestamp}]" if timestamp else ""
        memory = entry.get("memory")
        if memory is None:
            memory = json.dumps(entry, ensure_ascii=True)
        memories.append(f"[{idx + 1}]{timestamp_info} {memory}")

    memories_str = "\n\n".join(memories)
    return build_normalized_user_prompt(question, memories_str, question_date)


def _chunk_messages(messages: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    batches: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    current_chars = 0

    for message in messages:
        content = message.get("content", "")
        message_chars = len(content)
        if current and current_chars + message_chars > MEM0_MAX_CHARS:
            batches.append(current)
            current = []
            current_chars = 0

        current.append(message)
        current_chars += message_chars

    if current:
        batches.append(current)

    return batches


def _extract_event_ids(
    result: dict[str, object] | list[dict[str, object]] | object,
) -> list[str]:
    event_ids = []
    if isinstance(result, dict):
        result_dict = {str(key): val for key, val in result.items()}
        event_id = result_dict.get("id")
        if isinstance(event_id, str):
            event_ids.append(event_id)
    elif isinstance(result, list):
        for entry in result:
            if isinstance(entry, dict):
                entry_dict = {str(key): val for key, val in entry.items()}
                event_id = entry_dict.get("id")
                if isinstance(event_id, str):
                    event_ids.append(event_id)
    return event_ids


async def _await_mem0_indexing(event_ids: list[str]) -> None:
    if not event_ids or not MEM0_API_KEY:
        return

    poll_interval = 1.0
    timeout = 300.0
    completed: set[str] = set()
    start = time.monotonic()

    async with httpx.AsyncClient(timeout=30.0) as client:
        while time.monotonic() - start < timeout:
            for event_id in event_ids:
                if event_id in completed:
                    continue
                response = await client.get(
                    MEM0_EVENT_STATUS_URL.format(event_id=event_id),
                    headers={"Authorization": f"Token {MEM0_API_KEY}"},
                )
                if not response.is_success:
                    continue
                data = response.json()
                status = data.get("status")
                if status in {"SUCCEEDED", "FAILED"}:
                    completed.add(event_id)

            if len(completed) == len(event_ids):
                return
            await asyncio.sleep(poll_interval)


@solver
def mem0_setup(only_answer_turns: bool):
    async def solve(state: TaskState, generate: Generate):
        user_id = f"longmemeval_{state.sample_id}"
        state.metadata["mem0_user_id"] = user_id

        client = _mem0_client()
        await asyncio.to_thread(client.delete_all, user_id=user_id)
        event_ids: list[str] = []
        sessions = state.metadata.get("haystack_sessions", [])
        session_ids = state.metadata.get("haystack_session_ids", [])
        session_dates = state.metadata.get("haystack_dates", [])

        for index, session in enumerate(sessions):
            session_messages = []
            for turn in session:
                if only_answer_turns and not turn.get("has_answer"):
                    continue
                content = turn.get("content")
                if not content:
                    continue
                session_messages.append(
                    {"role": turn.get("role", "user"), "content": content}
                )

            if not session_messages:
                continue

            session_id = session_ids[index] if index < len(session_ids) else str(index)
            timestamp = session_dates[index] if index < len(session_dates) else None

            for batch in _chunk_messages(session_messages):
                result = await asyncio.to_thread(
                    client.add,
                    batch,
                    user_id=user_id,
                    version=MEM0_VERSION,
                    enable_graph=False,
                    async_mode=True,
                    metadata={
                        "sessionId": session_id,
                        "timestamp": timestamp,
                    },
                )
                event_ids.extend(_extract_event_ids(result))

        await _await_mem0_indexing(event_ids)

        response = await asyncio.to_thread(
            client.search,
            state.input_text,
            {
                "user_id": user_id,
                "top_k": 30,
                "enable_graph": False,
                "output_format": "v1.1",
            },
        )
        results = response.get("results", []) if isinstance(response, dict) else []
        prompt = _build_mem0_answer_prompt(
            state.input_text,
            results,
            state.metadata.get("question_date"),
        )
        state.messages = [
            ChatMessageSystem(content=NORMALIZED_SYSTEM_PROMPT),
            ChatMessageUser(content=prompt),
        ]

        return state

    return solve


async def mem0_cleanup(state: TaskState):
    user_id = state.metadata.get("mem0_user_id")
    if not user_id:
        return
    client = _mem0_client()
    await asyncio.to_thread(client.delete_all, user_id=user_id)

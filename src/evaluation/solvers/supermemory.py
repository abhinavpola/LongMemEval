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

SUPERMEMORY_BASE_URL = os.environ.get(
    "SUPERMEMORY_BASE_URL", "https://api.supermemory.ai"
).rstrip("/")
SUPERMEMORY_API_KEY = os.environ.get("SUPERMEMORY_API_KEY", "")
SUPERMEMORY_BULK_DELETE_PATH = "/v3/documents/bulk"
SUPERMEMORY_DOCUMENT_PATH = "/v3/documents"
SUPERMEMORY_DOCUMENT_GET_PATH = "/v3/documents/{doc_id}"
SUPERMEMORY_MEMORY_GET_PATH = "/v3/memories/{doc_id}"
SUPERMEMORY_SEARCH_PATH = "/v4/search"
SUPERMEMORY_SEARCH_THRESHOLD = 0.3


def _supermemory_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {SUPERMEMORY_API_KEY}" if SUPERMEMORY_API_KEY else "",
        "Content-Type": "application/json",
    }


def _supermemory_url(path: str) -> str:
    return f"{SUPERMEMORY_BASE_URL}{path}"


async def _supermemory_request(
    method: str, path: str, payload: dict | None = None
) -> dict:
    url = _supermemory_url(path)
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        response = await client.request(
            method, url, headers=_supermemory_headers(), json=payload
        )
        response.raise_for_status()
        if not response.content:
            return {}
        return response.json()


def _format_session_content(session: list[dict], formatted_date: str | None) -> str:
    session_str = json.dumps(session, ensure_ascii=True)
    session_str = session_str.replace("<", "&lt;").replace(">", "&gt;")
    if formatted_date:
        return (
            "Here is the date the following session took place: "
            f"{formatted_date}\n\nHere is the session as a stringified JSON:\n"
            f"{session_str}"
        )
    return f"Here is the session as a stringified JSON:\n{session_str}"


async def _supermemory_wait_for_indexing(document_ids: list[str]) -> None:
    if not document_ids:
        return

    poll_interval = 2.0
    timeout = 300.0

    def extract_status(payload: dict, keys: list[str]) -> str | None:
        for key in keys:
            if key in payload and isinstance(payload[key], str):
                return payload[key]
        nested = payload.get("memory")
        if isinstance(nested, dict):
            status = nested.get("status")
            if isinstance(status, str):
                return status
        return None

    for doc_id in document_ids:
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            doc = await _supermemory_request(
                "GET", SUPERMEMORY_DOCUMENT_GET_PATH.format(doc_id=doc_id)
            )
            status = extract_status(doc, ["status"])
            if status in {"done", "failed"}:
                break
            await asyncio.sleep(poll_interval)

        while time.monotonic() - start < timeout:
            memory = await _supermemory_request(
                "GET", SUPERMEMORY_MEMORY_GET_PATH.format(doc_id=doc_id)
            )
            status = extract_status(memory, ["status", "memoryStatus"])
            if status in {"done", "failed"}:
                break
            if status is None:
                break
            await asyncio.sleep(poll_interval)


def _deduplicate_and_sort_chunks(
    chunks: list[dict[str, object]],
) -> list[dict[str, object]]:
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        content = chunk.get("content")
        if not isinstance(content, str):
            continue
        if content in seen:
            continue
        seen.add(content)
        unique_chunks.append(chunk)
    return sorted(unique_chunks, key=lambda c: int(c.get("position", 0)))


def _string_key_dict(value: dict) -> dict[str, object]:
    return {str(key): val for key, val in value.items()}


def _build_supermemory_context(results: list[dict[str, object]]) -> str:
    all_chunks: list[dict[str, object]] = []
    for idx, result in enumerate(results):
        chunks = result.get("chunks", [])
        if isinstance(chunks, list):
            for chunk in chunks:
                if isinstance(chunk, dict):
                    chunk_dict = _string_key_dict(chunk)
                    all_chunks.append(
                        {
                            "content": chunk_dict.get("content"),
                            "position": chunk_dict.get("position", 0),
                        }
                    )

        chunk = result.get("chunk")
        if isinstance(chunk, str) and chunk.strip():
            all_chunks.append({"content": chunk, "position": idx})

    deduped = _deduplicate_and_sort_chunks(all_chunks)
    memories_section_parts = []
    for idx, result in enumerate(results):
        memory = result.get("memory", "")
        if not isinstance(memory, str):
            memory = ""
        metadata = result.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        metadata_dict = _string_key_dict(metadata)
        temporal_context = metadata_dict.get("temporalContext", {})
        document_date = None
        event_date = None
        if isinstance(temporal_context, dict):
            temporal_dict = _string_key_dict(temporal_context)
            document_date = temporal_dict.get("documentDate")
            event_date = temporal_dict.get("eventDate")

        memory_parts = [f"Result {idx + 1}:", memory]
        temporal_info: list[str] = []
        if isinstance(document_date, str) and document_date:
            temporal_info.append(f"documentDate: {document_date}")
        if event_date:
            if isinstance(event_date, list):
                event_dates = [str(d) for d in event_date]
            else:
                event_dates = [str(event_date)]
            temporal_info.append(f"eventDate: {', '.join(event_dates)}")
        if temporal_info:
            memory_parts.append(f"Temporal Context: {' | '.join(temporal_info)}")

        memories_section_parts.append("\n".join(memory_parts))

    memories_section = "\n\n---\n\n".join(memories_section_parts)

    if deduped:
        chunk_contents = "\n\n---\n\n".join(
            str(chunk.get("content", "")) for chunk in deduped
        )
        chunks_section = f"\n\n=== DEDUPLICATED CHUNKS ===\n{chunk_contents}"
    else:
        chunks_section = ""

    return memories_section + chunks_section


def build_supermemory_answer_prompt(
    question: str, context: list[dict[str, object]], question_date: str | None
) -> str:
    retrieved_context = _build_supermemory_context(context)
    return build_normalized_user_prompt(question, retrieved_context, question_date)


async def _supermemory_bulk_delete(container_tag: str) -> bool:
    payload = {"containerTags": [container_tag]}
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = await _supermemory_request(
                "DELETE", SUPERMEMORY_BULK_DELETE_PATH, payload
            )
            if response.get("success") is False or "error" in response:
                raise RuntimeError(f"Bulk delete failed: {response}")
            return True
        except Exception as exc:
            if attempt == max_attempts:
                print(f"Supermemory cleanup failed: {exc}")
                return False
            await asyncio.sleep(2**attempt)
    return False


@solver
def supermemory_setup(only_answer_turns: bool):
    async def solve(state: TaskState, generate: Generate):
        container_tag = f"longmemeval_{state.sample_id}"
        state.metadata["supermemory_container_tag"] = container_tag

        # Best-effort cleanup in case a previous run crashed mid-sample.
        if not await _supermemory_bulk_delete(container_tag):
            print(f"Supermemory pre-ingest cleanup failed for {container_tag}")

        document_ids = []
        sessions = state.metadata["haystack_sessions"]
        session_ids = state.metadata.get("haystack_session_ids", [])
        session_dates = state.metadata.get("haystack_dates", [])
        for index, session in enumerate(sessions):
            filtered_session = []
            for turn in session:
                if only_answer_turns and not turn.get("has_answer"):
                    continue
                content = turn.get("content")
                if not content:
                    continue
                filtered_session.append(
                    {"role": turn.get("role", "user"), "content": content}
                )

            if not filtered_session:
                continue

            formatted_date = (
                session_dates[index] if index < len(session_dates) else None
            )
            session_id = session_ids[index] if index < len(session_ids) else str(index)
            content = _format_session_content(filtered_session, formatted_date)
            payload = {
                "content": content,
                "containerTag": container_tag,
                "metadata": {
                    "sessionId": session_id,
                    **({"date": formatted_date} if formatted_date else {}),
                },
            }
            response = await _supermemory_request(
                "POST", SUPERMEMORY_DOCUMENT_PATH, payload
            )
            doc_id = response.get("id")
            if not doc_id:
                raise RuntimeError(f"Supermemory add failed: {response}")
            document_ids.append(doc_id)

        if document_ids:
            try:
                await _supermemory_wait_for_indexing(document_ids)
            except Exception:
                if not await _supermemory_bulk_delete(container_tag):
                    print(
                        "Supermemory cleanup failed after indexing error for "
                        f"{container_tag}"
                    )
                raise

        payload = {
            "q": state.input_text,
            "containerTag": container_tag,
            "limit": 10,
            "threshold": SUPERMEMORY_SEARCH_THRESHOLD,
            "include": {"chunks": True},
        }
        response = await _supermemory_request("POST", SUPERMEMORY_SEARCH_PATH, payload)
        results = response.get("results", [])
        if not isinstance(results, list):
            results = []

        prompt = build_supermemory_answer_prompt(
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


async def supermemory_cleanup(state: TaskState):
    container_tag = state.metadata.get("supermemory_container_tag")
    if not container_tag:
        return
    if not await _supermemory_bulk_delete(container_tag):
        print(f"Supermemory cleanup failed for {container_tag}")

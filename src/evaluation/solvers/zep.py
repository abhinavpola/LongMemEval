import asyncio
import os
import re
from typing import Union

from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.solver import Generate, TaskState, solver
from evaluation.solvers.prompt_utils import (
    NORMALIZED_SYSTEM_PROMPT,
    build_normalized_user_prompt,
)
from zep_cloud import EpisodeData, EntityEdge, EntityNode, Zep

ZEP_API_KEY = os.environ.get("ZEP_API_KEY", "")
ZEP_MAX_DATA_SIZE = 9500
ZEP_BATCH_SIZE = 20

ZEP_ENTITY_TYPES = {
    "Person": {
        "description": "A person entity representing individuals in conversations",
        "fields": {},
    },
    "Preference": {
        "description": "User preferences, choices, opinions, or selections. High priority for classification.",
        "fields": {},
    },
    "Location": {
        "description": "Physical or virtual places where activities occur",
        "fields": {},
    },
    "Event": {
        "description": "Time-bound activities, occurrences, or experiences",
        "fields": {},
    },
    "Object": {
        "description": "Physical items, tools, devices, or possessions",
        "fields": {},
    },
    "Topic": {
        "description": "Subjects of conversation, interest, or knowledge domains",
        "fields": {},
    },
    "Organization": {
        "description": "Companies, institutions, groups, or formal entities",
        "fields": {},
    },
    "Document": {
        "description": "Information content in various forms like books, articles, reports",
        "fields": {},
    },
}

_ZEP_ONTOLOGY_SET: set[str] = set()


def _zep_client() -> Zep:
    return Zep(api_key=ZEP_API_KEY or None)


def _sanitize_graph_id(container_tag: str) -> str:
    return f"memorybench_{re.sub(r'[^a-zA-Z0-9_-]', '_', container_tag)}"


def _split_into_chunks(text: str, max_size: int) -> list[str]:
    if len(text) <= max_size:
        return [text]

    chunks = []
    remaining = text

    while remaining:
        if len(remaining) <= max_size:
            chunks.append(remaining)
            break

        split_index = remaining.rfind(". ", 0, max_size)
        if split_index == -1 or split_index < max_size * 0.5:
            split_index = remaining.rfind("\n", 0, max_size)
        if split_index == -1 or split_index < max_size * 0.5:
            split_index = remaining.rfind(" ", 0, max_size)
        if split_index == -1 or split_index < max_size * 0.3:
            split_index = max_size

        chunks.append(remaining[: split_index + 1].strip())
        remaining = remaining[split_index + 1 :].strip()

    return chunks


def _build_zep_context(context: list[Union[EntityNode, EntityEdge]]) -> str:
    facts = []
    entities = []

    for entry in context:
        if isinstance(entry, EntityNode):
            name = entry.name or "Unknown"
            summary = entry.summary or ""
            entities.append(f"  - {name}: {summary}")
        elif isinstance(entry, EntityEdge):
            content = entry.fact or str(entry)
            valid_at = entry.valid_at
            facts.append(f"  - {content} (event_time: {valid_at or 'unknown'})")
        else:
            raise TypeError(f"Unexpected Zep result type: {type(entry)}")

    context_str = (
        "FACTS and ENTITIES represent relevant context to the current conversation.\n\n"
        "# These are the most relevant facts for the conversation along with the datetime of the event that the fact refers to.\n"
        "# If a fact mentions something happening a week ago, then the datetime will be the date time of last week and not the datetime\n"
        "# of when the fact was stated.\n"
        "# Timestamps in memories represent the actual time the event occurred, not the time the event was mentioned in a message.\n\n"
        "<FACTS>\n"
        f"{chr(10).join(facts)}\n"
        "</FACTS>"
    )

    if entities:
        context_str += (
            "\n\n# These are the most relevant entities\n"
            "# ENTITY_NAME: entity summary\n"
            "<ENTITIES>\n"
            f"{chr(10).join(entities)}\n"
            "</ENTITIES>"
        )

    return context_str


def _build_zep_answer_prompt(
    question: str,
    context: list[Union[EntityNode, EntityEdge]],
    question_date: str | None,
) -> str:
    context_str = _build_zep_context(context)
    return build_normalized_user_prompt(question, context_str, question_date)


@solver
def zep_setup(only_answer_turns: bool):
    async def solve(state: TaskState, generate: Generate):
        container_tag = f"longmemeval_{state.sample_id}"
        state.metadata["zep_container_tag"] = container_tag

        client = _zep_client()
        graph_id = _sanitize_graph_id(container_tag)
        state.metadata["zep_graph_id"] = graph_id

        try:
            await asyncio.to_thread(client.graph.delete, graph_id)
        except Exception as exc:
            print(f"Zep graph delete failed for {graph_id}: {exc}")

        try:
            await asyncio.to_thread(
                client.graph.create,
                graph_id=graph_id,
                name=f"MemoryBench {container_tag}",
                description="Memory benchmark evaluation graph",
            )
        except Exception as exc:
            print(f"Zep graph create failed for {graph_id}: {exc}")

        if graph_id not in _ZEP_ONTOLOGY_SET:
            set_ontology = getattr(client.graph, "set_ontology", None)
            if callable(set_ontology):
                try:
                    await asyncio.to_thread(
                        set_ontology,
                        ZEP_ENTITY_TYPES,
                        {},
                        graph_ids=[graph_id],
                    )
                    _ZEP_ONTOLOGY_SET.add(graph_id)
                except Exception as exc:
                    print(f"Zep set_ontology failed for {graph_id}: {exc}")
            else:
                print(f"Zep set_ontology not available for {graph_id}")

        episodes = []
        sessions = state.metadata.get("haystack_sessions", [])
        session_dates = state.metadata.get("haystack_dates", [])
        for index, session in enumerate(sessions):
            iso_date = session_dates[index] if index < len(session_dates) else None
            for message in session:
                if only_answer_turns and not message.get("has_answer"):
                    continue
                content = message.get("content")
                if not content:
                    continue
                speaker = message.get("speaker") or message.get("role") or "user"
                message_data = f"{speaker}: {content}"
                if len(message_data) > ZEP_MAX_DATA_SIZE:
                    for chunk in _split_into_chunks(message_data, ZEP_MAX_DATA_SIZE):
                        episodes.append(
                            EpisodeData(
                                type="message",
                                data=chunk,
                                created_at=iso_date,
                            )
                        )
                else:
                    episodes.append(
                        EpisodeData(
                            type="message",
                            data=message_data,
                            created_at=iso_date,
                        )
                    )

        for i in range(0, len(episodes), ZEP_BATCH_SIZE):
            batch = episodes[i : i + ZEP_BATCH_SIZE]
            await asyncio.to_thread(
                client.graph.add_batch,
                graph_id=graph_id,
                episodes=batch,
            )

        await asyncio.sleep(3.0)

        edge_limit = 20
        node_limit = 10
        edges_response, nodes_response = await asyncio.gather(
            asyncio.to_thread(
                client.graph.search,
                graph_id=graph_id,
                query=state.input_text,
                limit=edge_limit,
                scope="edges",
                reranker="cross_encoder",
            ),
            asyncio.to_thread(
                client.graph.search,
                graph_id=graph_id,
                query=state.input_text,
                limit=node_limit,
                scope="nodes",
                reranker="cross_encoder",
            ),
        )

        results: list[Union[EntityNode, EntityEdge]] = []
        edges = getattr(edges_response, "edges", None)
        if edges:
            results.extend(edges)

        nodes = getattr(nodes_response, "nodes", None)
        if nodes:
            results.extend(nodes)

        prompt = _build_zep_answer_prompt(
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


async def zep_cleanup(state: TaskState):
    container_tag = state.metadata.get("zep_container_tag")
    graph_id = state.metadata.get("zep_graph_id")
    if not container_tag or not graph_id:
        return
    client = _zep_client()
    try:
        await asyncio.to_thread(client.graph.delete, graph_id)
    except Exception as exc:
        print(f"Zep graph delete failed for {graph_id}: {exc}")

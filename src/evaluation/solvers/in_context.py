from inspect_ai.model import ChatMessageAssistant, ChatMessageUser
from inspect_ai.solver import Generate, TaskState, solver


@solver
def in_context_setup():
    async def solve(state: TaskState, generate: Generate):
        history = []
        for session in state.metadata["haystack_sessions"]:
            for turn in session:
                if turn["role"] == "user":
                    history.append(ChatMessageUser(content=turn["content"]))
                elif turn["role"] == "assistant":
                    history.append(ChatMessageAssistant(content=turn["content"]))

        state.messages = history + state.messages
        return state

    return solve

"""
Entry point. Run: python main.py
Type natural language requests; type 'exit' to quit.
"""
import os

from project_bootstrap import ensure_repo_root

ensure_repo_root()

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("ANTHROPIC_API_KEY"):
    raise SystemExit("Set ANTHROPIC_API_KEY in your .env file first (see .env.example).")

from langchain_core.messages import HumanMessage, AIMessage
from agents.data_agent import data_agent, DataAgentSchema


def _extract_audience_prefix(user_input: str):
    prefix_map = {
        "executive:": "executive",
        "exec:": "executive",
        "analyst:": "analyst",
        "operator:": "operator",
        "ops:": "operator",
    }
    lowered = user_input.lower()
    for prefix, audience in prefix_map.items():
        if lowered.startswith(prefix):
            return audience, user_input[len(prefix):].strip()
    return "general", user_input


def main():
    print("Data Agent ready. Ask a question (SQL / ETL / chart / quality / lineage / forecast / security / briefing). Use prefixes like 'executive:', 'analyst:', or 'operator:'. Type 'exit' to quit.\n")
    history = []  # running conversation memory across turns, for follow-ups
    # _format_history() in agents/data_agent.py already caps what's SENT to
    # the router prompt at 6 turns, but without a cap here the in-memory list
    # itself grows for the lifetime of the process - harmless for a short
    # session, but unbounded over a very long-running one. Keep a bit more
    # than what's formatted into the prompt so context isn't lost right at
    # the boundary.
    MAX_HISTORY_MESSAGES = 40  # ~20 turns

    while True:
        user_input = input("> ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        audience, question = _extract_audience_prefix(user_input)
        try:
            response = data_agent.invoke(
                DataAgentSchema(
                    messages=[HumanMessage(content=question)],
                    conversation_history=history,
                    audience=audience,
                )
            )
        except Exception as e:
            # invoke_with_resilience (utils/llm_pick.py) already retries
            # transient errors and falls back down the model tier chain;
            # this only fires once every tier is genuinely exhausted, or on
            # a non-LLM failure. Keep the session alive rather than crashing
            # the whole CLI on one bad turn - the user can just try again.
            print(f"\n⚠️  That request failed and couldn't be recovered: {e}\n"
                  f"You can try again, rephrase, or type 'exit' to quit.\n")
            continue

        answer = response["final_answer"] if isinstance(response, dict) else response.final_answer
        needs_clarification = (
            response["needs_clarification"] if isinstance(response, dict)
            else response.needs_clarification
        )

        if audience != "general":
            answer = f"[{audience.upper()} BRIEFING]\n\n{answer}"

        print(f"\n{answer}\n")

        # Keep memory of this exchange so follow-ups ("now filter by region") work.
        # If the agent asked for clarification, don't count it as a resolved turn -
        # the user's next message is treated as continuing the same request.
        history.append(HumanMessage(content=user_input))
        if not needs_clarification:
            history.append(AIMessage(content=answer))
        if len(history) > MAX_HISTORY_MESSAGES:
            history[:] = history[-MAX_HISTORY_MESSAGES:]


if __name__ == "__main__":
    main()

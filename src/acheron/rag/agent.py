"""Nexus ReAct Agent — intelligent evidence gathering via tool use.

Wraps the existing RAG pipeline with a think → act → observe loop.
The agent decides WHEN and WHAT to search for, then hands off to the
existing mode-specific prompts for structured response generation.

Design principles:
  1. The agent is an ORCHESTRATION layer — it never replaces the
     hypothesis engine, mode prompts, or evidence parsing.
  2. Phase 1 (evidence gathering) is non-streaming.
  3. Phase 2 (response generation) uses the SAME streaming code path
     as analyze_stream — identical output structure.
  4. If the agent fails for any reason, we fall back to analyze_stream.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from acheron.models import QueryResult
from acheron.rag.tools import (
    TOOL_DEFINITIONS_ANTHROPIC,
    ToolResult,
    execute_tool,
    get_tool_definitions_openai,
)

logger = logging.getLogger(__name__)

# Maximum tool-use rounds before forcing a response.
_MAX_AGENT_TURNS = 4

# System prompt for the evidence-gathering phase (Phase 1).
_AGENT_GATHER_PROMPT = """\
You are the evidence-gathering module of Nexus, a bioelectricity research \
engine.  Your task is to assess whether the available evidence is sufficient \
to comprehensively answer the user's question, and if not, to gather more \
evidence using your tools.

## Current Evidence Summary
{evidence_summary}

## Your Task
1. Review the evidence summary above.
2. Decide if the evidence is SUFFICIENT to answer the user's question.
   Evidence is sufficient when:
   - You have relevant passages that directly address the key aspects
   - Source diversity is reasonable (ideally multiple papers)
   - Specific parameters/values asked for are present (or you know they
     are not published and need to be estimated)
3. If evidence is INSUFFICIENT:
   - Use your tools to search for more.  You may reformulate the query
     to target missing aspects.
   - Prefer search_knowledge_base first (fast, local).
   - Use search_pubmed for peer-reviewed literature.
   - Use search_biorxiv or search_arxiv for preprints.
   - Use run_computation when the query requires quantitative analysis.
4. When evidence is sufficient (or you've exhausted reasonable searches):
   Respond with the single word DONE.

IMPORTANT:
- Do NOT generate the final answer.  Your only job is evidence gathering.
- Do NOT explain your reasoning at length.  Keep responses under 50 words.
- You may search up to {max_turns} times total.
- If the initial evidence is already strong, just say DONE immediately.
"""


class NexusAgent:
    """ReAct agent for intelligent, multi-step evidence gathering.

    The agent wraps a RAGPipeline and uses its LLM client to decide
    whether more evidence is needed, then executes tool calls.

    Usage
    -----
    >>> agent = NexusAgent(pipeline)
    >>> extra_results, comp_ctx = agent.gather_evidence(
    ...     query="What is the role of Vmem in planarian regeneration?",
    ...     initial_results=local_results,
    ... )
    """

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline
        self._provider: str = pipeline._provider

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def gather_evidence(
        self,
        query: str,
        initial_results: list[QueryResult],
    ) -> tuple[list[QueryResult], str]:
        """Run the evidence-gathering loop.

        Parameters
        ----------
        query : str
            The user's original question.
        initial_results : list[QueryResult]
            Results already retrieved from the local vectorstore.

        Returns
        -------
        tuple[list[QueryResult], str]
            (additional_results, computation_context)
            ``additional_results`` are new passages found by the agent.
            ``computation_context`` is any reasoning-engine output to
            inject into the prompt alongside retrieved passages.
        """
        evidence_summary = self._summarize_initial(initial_results)
        system_prompt = _AGENT_GATHER_PROMPT.format(
            evidence_summary=evidence_summary,
            max_turns=_MAX_AGENT_TURNS,
        )

        messages: list[dict] = [
            {"role": "user", "content": f"User question: {query}"},
        ]

        additional_results: list[QueryResult] = []
        computation_context = ""

        tools_anthropic = TOOL_DEFINITIONS_ANTHROPIC
        tools_openai = get_tool_definitions_openai()

        for turn in range(_MAX_AGENT_TURNS):
            try:
                response = self._call_with_tools(
                    system_prompt, messages, tools_anthropic, tools_openai,
                )
            except Exception as exc:
                logger.warning("Agent LLM call failed (turn %d): %s", turn, exc)
                break

            # Extract tool calls from the response.
            tool_calls = self._extract_tool_calls(response)

            if not tool_calls:
                # No tool calls — agent is done gathering.
                logger.info(
                    "Agent finished gathering after %d turn(s)", turn + 1,
                )
                break

            # Append the assistant message to the conversation.
            self._append_assistant(messages, response)

            # Execute each tool and add results.
            for tc in tool_calls:
                tool_result = execute_tool(
                    tc["name"], tc["input"], self.pipeline,
                )
                additional_results.extend(tool_result.results)
                if tool_result.computation_context:
                    computation_context += "\n" + tool_result.computation_context

                self._append_tool_result(messages, tc, tool_result.text)

                logger.info(
                    "Agent tool '%s' returned %d results",
                    tc["name"],
                    len(tool_result.results),
                )

        return additional_results, computation_context

    # ------------------------------------------------------------------
    # LLM interaction (provider-agnostic)
    # ------------------------------------------------------------------

    def _call_with_tools(
        self,
        system_prompt: str,
        messages: list[dict],
        tools_anthropic: list[dict],
        tools_openai: list[dict],
    ) -> Any:
        """Call the LLM with tool definitions.  Returns the raw response."""
        client = self.pipeline._get_client()
        model = self.pipeline.settings.resolved_llm_model

        if self._provider == "anthropic":
            return client.messages.create(
                model=model,
                system=system_prompt,
                messages=messages,
                tools=tools_anthropic,
                temperature=0.0,
                max_tokens=1024,
            )
        else:
            openai_messages = [
                {"role": "system", "content": system_prompt},
                *messages,
            ]
            return client.chat.completions.create(
                model=model,
                messages=openai_messages,
                tools=tools_openai,
                temperature=0.0,
                max_tokens=1024,
            )

    # ------------------------------------------------------------------
    # Response parsing (provider-agnostic)
    # ------------------------------------------------------------------

    def _extract_tool_calls(self, response: Any) -> list[dict]:
        """Extract tool calls from the LLM response.

        Returns a list of dicts: [{"id": ..., "name": ..., "input": ...}]
        Returns empty list if the LLM decided not to use tools.
        """
        if self._provider == "anthropic":
            return self._extract_anthropic(response)
        else:
            return self._extract_openai(response)

    def _extract_anthropic(self, response: Any) -> list[dict]:
        """Parse Anthropic tool_use content blocks."""
        calls = []
        for block in response.content:
            if block.type == "tool_use":
                calls.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        return calls

    def _extract_openai(self, response: Any) -> list[dict]:
        """Parse OpenAI tool_calls."""
        choice = response.choices[0]
        if not choice.message.tool_calls:
            return []
        calls = []
        for tc in choice.message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, AttributeError):
                args = {}
            calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "input": args,
            })
        return calls

    # ------------------------------------------------------------------
    # Message management (provider-agnostic)
    # ------------------------------------------------------------------

    def _append_assistant(self, messages: list[dict], response: Any) -> None:
        """Add the assistant's response to the message list."""
        if self._provider == "anthropic":
            messages.append({
                "role": "assistant",
                "content": response.content,
            })
        else:
            messages.append(response.choices[0].message.model_dump())

    def _append_tool_result(
        self,
        messages: list[dict],
        tool_call: dict,
        result_text: str,
    ) -> None:
        """Add a tool result to the message list."""
        # Truncate very long results to avoid blowing up context.
        if len(result_text) > 8000:
            result_text = result_text[:8000] + "\n[... truncated]"

        if self._provider == "anthropic":
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call["id"],
                        "content": result_text,
                    }
                ],
            })
        else:
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": result_text,
            })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _summarize_initial(results: list[QueryResult]) -> str:
        """Create a concise summary of initial retrieval results."""
        if not results:
            return (
                "No local evidence found.  The knowledge base returned "
                "zero results for this query."
            )

        lines = [f"Local vectorstore returned {len(results)} passages:\n"]
        for i, r in enumerate(results[:6], 1):  # Show top 6 only.
            score = f" (score: {r.relevance_score:.3f})" if r.relevance_score else ""
            lines.append(f"  [{i}] {r.paper_title}{score}")
            excerpt = (r.excerpt or r.text[:200]).replace("\n", " ")
            lines.append(f"      \"{excerpt[:200]}...\"")

        if len(results) > 6:
            lines.append(f"  ... and {len(results) - 6} more passages.")

        # Overall quality assessment.
        best = max(r.relevance_score for r in results) if results else 0.0
        lines.append(f"\nBest relevance score: {best:.3f}")
        if best < 0.35:
            lines.append("⚠ Evidence quality is LOW.")
        elif best < 0.55:
            lines.append("Evidence quality is MODERATE.")
        else:
            lines.append("Evidence quality is GOOD.")

        return "\n".join(lines)

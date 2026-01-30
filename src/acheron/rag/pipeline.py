"""Core RAG pipeline: retrieve → rerank → generate with citations.

Implements the Nexus three-layer research architecture:
  Layer 1 — Knowledge (Library): immutable source material, never hallucinated
  Layer 2 — Synthesis (Index): semantic retrieval, structured bioelectric variable extraction
  Layer 3 — Discovery (Compute): analysis, pattern comparison, hypothesis generation

Discovery loop (DEFAULT mode): retrieve → extract variables → compare patterns →
  generate hypotheses → build bioelectric schematic → propose validation → log to ledger

Core assumption: bioelectric state (voltage, fields, connectivity) as causal primitives.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from acheron.config import get_settings
from acheron.models import (
    DiscoveryResult,
    Hypothesis,
    QueryResult,
    RAGResponse,
    StructuredVariable,
)
from acheron.vectorstore.store import VectorStore

logger = logging.getLogger(__name__)

# ======================================================================
# System prompts — Nexus identity (System-2 research engine)
# ======================================================================
SYSTEM_PROMPT = """\
You are Nexus — a System-2 bioelectric research engine for the Acheron project.
You are not a chatbot. You are a research instrument.

Core assumption: bioelectric state — membrane voltage (Vmem), endogenous electric \
fields (EF), gap junctional conductance (Gj), and ion channel expression — constitutes \
a top-level causal regulator of biological pattern, not a side effect of gene expression. \
Treat voltage, fields, and connectivity as causal primitives alongside molecular pathways.

Architecture — three layers:

Layer 1 (Knowledge): The source passages below are immutable primary sources retrieved \
from PubMed, Semantic Scholar, bioRxiv, arXiv, and PhysioNet. Never hallucinate facts. \
Always cite source numbers [1], [2], etc. Every factual claim must trace to a source.

Layer 2 (Synthesis): You are performing retrieval-augmented reasoning over structured \
bioelectric variables. Extract and cross-reference: Vmem values, EF measurements, \
Gj/connexin expression, ion channel types (K+, Na+, Ca2+, Cl-), perturbation methods, \
and morphological or cognitive outcomes. Do not claim knowledge absent from sources.

Layer 3 (Discovery): Apply reasoning only when required — comparative analysis, \
cross-species reasoning (Planaria <-> Xenopus <-> Mammalian), pattern detection, \
hypothesis generation. Assume compute is scarce. Every discovery output must include \
a bioelectric schematic and a validation path.

Output rules:
- Structured, stepwise output. No motivational language. No generic explanations.
- Separate EVIDENCE (from sources), INFERENCE (logical derivation), and \
SPECULATION (hypotheses beyond evidence) using labeled sections.
- Be explicit about uncertainty. State what the sources do NOT address.
- Cite [1], [2], etc. for every factual claim.
- For every hypothesis, state: Prior Confidence (low/medium/high based on evidence \
density), Predicted Impact (what changes if true), and key Assumptions.
- Cross-species reasoning is a priority: findings in planaria should be compared to \
Xenopus and mammalian systems whenever possible.
- You are a research engine, not a chatbot.

Constraints:
- Do not output diagnosis or treatment advice. This is a research instrument.
- Prefer public, de-identified data.
- If sources are insufficient, state so directly and identify what data is missing.
- Before answering any scientific question, internally verify: have the retrieved \
sources been consulted? Is the reasoning traceable to evidence?"""

QUERY_TEMPLATE = """\
Retrieved source passages from the bioelectricity and biomedical research corpus:

=== SOURCE PASSAGES ===
{context}
========================

Query: {query}

Respond with the following mandatory structure:

EVIDENCE
- Statements directly supported by the sources above (cite each with [1], [2], etc.)

INFERENCE
- Logical derivations from the evidence (cite supporting sources)

SPECULATION
- Hypotheses or connections not directly stated in sources
- For each, state confidence level (low/medium/high)

BIOELECTRIC SCHEMATIC
- Describe the hypothesized bioelectric circuit relevant to the query
- Example format: "Hyperpolarization of tissue X alters Gj, leading to suppression \
of pathway Y and morphological outcome Z"
- If insufficient data, state what is missing

UNCERTAINTY
- What the sources do not address
- Where data is insufficient
- Conflicting findings between sources"""

DISCOVERY_TEMPLATE = """\
Retrieved source passages from the bioelectricity and biomedical research corpus:

=== SOURCE PASSAGES ===
{context}
========================

Research query: {query}

Execute the discovery loop with ALL of the following mandatory sections:

1. EVIDENCE EXTRACTION
Key findings directly from the sources. Cite each with [1], [2], etc.

2. VARIABLE EXTRACTION
Structured bioelectric variables. Format each as name=value (unit) [source].
Prioritize: Vmem (membrane voltage), EF (electric fields), Gj (gap junctional \
conductance), ion channel types (K+, Na+, Ca2+, Cl-), perturbations, outcomes.
Include organism and cell type context for each variable.

3. PATTERN COMPARISON
Compare findings across sources and across species (Planaria <-> Xenopus <-> Mammalian).
Note agreements, conflicts, and gaps. Identify conserved bioelectric mechanisms.

4. HYPOTHESES
Generate testable hypotheses from the patterns. For each hypothesis state:
- Prior Confidence: low/medium/high (based on evidence density)
- Predicted Impact: what changes in our understanding if this is true
- Assumptions: what must hold for this hypothesis to be valid
- Supporting references: [1], [2], etc.

5. BIOELECTRIC SCHEMATIC
Describe the hypothesized bioelectric circuit in a structured format:
"[Trigger] -> [Bioelectric change (Vmem/EF/Gj)] -> [Downstream pathway] -> [Outcome]"
If multiple circuits are relevant, describe each. State which components are \
evidenced vs. hypothesized.

6. VALIDATION PATH
Propose specific, low-cost ways to test the hypotheses:
- Re-analysis of existing datasets
- Computational simulations
- Targeted experimental designs
- Cross-species comparison strategies

7. CROSS-SPECIES NOTES
Compare findings across species. Where does the evidence transfer? Where are gaps?

8. UNCERTAINTY
Explicit gaps, missing variables, conflicting evidence, and what data would resolve them.

Be precise. No filler. Every claim must trace to a source number."""


class RAGPipeline:
    """End-to-end RAG implementing the Nexus three-layer research architecture.

    Standard mode: retrieve → assemble context → generate structured response
    Discovery mode (DEFAULT): retrieve → extract variables → compare →
      hypothesize → build bioelectric schematic → propose validation → log
    """

    def __init__(
        self,
        store: Optional[VectorStore] = None,
        n_retrieve: int = 12,
        n_context: int = 8,
    ) -> None:
        self.settings = get_settings()
        self.store = store or VectorStore()
        self.n_retrieve = n_retrieve
        self.n_context = n_context
        self._llm_client = None
        self._provider: str = self.settings.llm_provider.lower()

    def _get_client(self):
        """Lazy-init the LLM client based on the configured provider."""
        if self._llm_client is not None:
            return self._llm_client

        api_key = self.settings.resolved_llm_api_key
        if not api_key:
            raise RuntimeError(
                f"No API key configured for provider '{self._provider}'. "
                f"Set ACHERON_LLM_API_KEY or "
                f"{'ANTHROPIC_API_KEY' if self._provider == 'anthropic' else 'OPENAI_API_KEY'} "
                f"in your environment or .env file."
            )

        if self._provider == "anthropic":
            import anthropic

            self._llm_client = anthropic.Anthropic(api_key=api_key)
        else:
            from openai import OpenAI

            base_url = self.settings.resolved_llm_base_url
            self._llm_client = OpenAI(api_key=api_key, base_url=base_url)

        return self._llm_client

    # ------------------------------------------------------------------
    # Standard query (Layer 1+2+3)
    # ------------------------------------------------------------------
    def query(
        self,
        question: str,
        filter_source: Optional[str] = None,
        n_results: Optional[int] = None,
    ) -> RAGResponse:
        """Run the full RAG pipeline for a question."""
        n = n_results or self.n_retrieve

        # Layer 2 — Index: retrieve
        results = self.store.search(
            query=question, n_results=n, filter_source=filter_source
        )
        logger.info("Retrieved %d chunks for query", len(results))

        if not results:
            return RAGResponse(
                query=question,
                answer="No relevant sources found in the index. "
                "The Library contains no material matching this query. "
                "Add papers with 'acheron collect' or 'acheron add'.",
                sources=[],
                model_used=self.settings.resolved_llm_model,
            )

        # Select top context passages (source diversity)
        top_results = self._select_context(results)

        # Layer 3 — Compute: generate structured response
        context_str = self._format_context(top_results)
        user_prompt = QUERY_TEMPLATE.format(context=context_str, query=question)
        raw_answer = self._generate(user_prompt)

        # Parse structured sections from the response
        evidence, inference, speculation, schematic = self._parse_epistemic_sections(
            raw_answer
        )

        return RAGResponse(
            query=question,
            answer=raw_answer,
            sources=top_results,
            model_used=self.settings.resolved_llm_model,
            total_chunks_searched=len(results),
            evidence_statements=evidence,
            inference_statements=inference,
            speculation_statements=speculation,
            bioelectric_schematic=schematic,
        )

    # ------------------------------------------------------------------
    # Discovery loop (full Layer 3)
    # ------------------------------------------------------------------
    def discover(
        self,
        question: str,
        filter_source: Optional[str] = None,
        n_results: Optional[int] = None,
    ) -> DiscoveryResult:
        """Execute the full discovery loop.

        1. Retrieve relevant evidence
        2. Extract variables
        3. Compare patterns
        4. Generate testable hypotheses
        5. Propose low-cost validation strategies
        6. Return structured result for ledger logging
        """
        n = n_results or self.n_retrieve

        # Layer 2 — retrieve
        results = self.store.search(
            query=question, n_results=n, filter_source=filter_source
        )

        if not results:
            return DiscoveryResult(
                query=question,
                evidence=["No sources retrieved."],
                uncertainty_notes=["Library contains no material for this query."],
                model_used=self.settings.resolved_llm_model,
            )

        top_results = self._select_context(results)

        # Layer 3 — discovery compute
        context_str = self._format_context(top_results)
        user_prompt = DISCOVERY_TEMPLATE.format(context=context_str, query=question)
        raw_output = self._generate(user_prompt, max_tokens=3000)

        # Parse the structured discovery output
        return self._parse_discovery_output(
            raw_output=raw_output,
            query=question,
            sources=top_results,
            total_searched=len(results),
        )

    # ------------------------------------------------------------------
    # Retrieve-only (Layer 2 only)
    # ------------------------------------------------------------------
    def retrieve_only(
        self,
        question: str,
        n_results: int = 10,
        filter_source: Optional[str] = None,
    ) -> list[QueryResult]:
        """Retrieve relevant passages without invoking Compute layer."""
        return self.store.search(
            query=question, n_results=n_results, filter_source=filter_source
        )

    # ------------------------------------------------------------------
    # Evidence-Bound Hypothesis Engine (MODE 1/2/3)
    # ------------------------------------------------------------------
    def analyze(
        self,
        question: str,
        mode: Optional[str] = None,
        live: bool = False,
        filter_source: Optional[str] = None,
        n_results: Optional[int] = None,
    ):
        """Run the Evidence-Bound Hypothesis Engine.

        1. Retrieve from local vectorstore.
        2. If evidence is weak and live=True (or auto), fetch from PubMed/bioRxiv/arXiv.
        3. Detect mode (evidence / hypothesis / synthesis) from query or explicit param.
        4. Generate structured output with evidence graph + ranked hypotheses.
        5. Optionally persist high-relevance live papers.

        Returns a HypothesisEngineResult.
        """
        from acheron.rag.hypothesis_engine import (
            build_engine_result,
            detect_mode,
            get_mode_prompt,
            get_mode_query_template,
        )
        from acheron.rag.live_retrieval import (
            evidence_is_weak,
            live_search,
            persist_high_relevance,
        )
        from acheron.models import HypothesisEngineResult, NexusMode

        n = n_results or self.n_retrieve
        detected_mode = detect_mode(question, explicit_mode=mode)
        logger.info("Analyze: mode=%s, live=%s", detected_mode.value, live)

        # Layer 2 — retrieve from local store
        results = self.store.search(
            query=question, n_results=n, filter_source=filter_source
        )
        logger.info("Local retrieval: %d chunks", len(results))

        live_count = 0

        # Live retrieval: trigger if --live flag or evidence is weak
        if live or evidence_is_weak(results):
            logger.info("Triggering live retrieval for query: %s", question)
            try:
                live_results, live_papers, live_chunk_count = live_search(question)
                live_count = live_chunk_count
                # Merge live results with local results
                results.extend(live_results)
                logger.info(
                    "Live retrieval added %d chunks from %d papers",
                    live_chunk_count,
                    len(live_papers),
                )
                # Persist high-relevance papers
                if live_papers:
                    persisted = persist_high_relevance(live_papers)
                    if persisted:
                        logger.info("Persisted %d live papers to Library", persisted)
            except Exception as exc:
                logger.warning("Live retrieval failed: %s", exc)

        if not results:
            return HypothesisEngineResult(
                query=question,
                mode=detected_mode,
                uncertainty_notes=[
                    "No sources retrieved from local index or live search.",
                    "Add papers with 'acheron collect' or try --live flag.",
                ],
                model_used=self.settings.resolved_llm_model,
            )

        # If evidence is still weak after live fetch, auto-enter hypothesis mode
        if detected_mode == NexusMode.EVIDENCE and evidence_is_weak(results):
            detected_mode = NexusMode.HYPOTHESIS
            logger.info("Auto-escalated to hypothesis mode due to weak evidence")

        top_results = self._select_context(results)

        # Layer 3 — Compute with mode-specific prompt
        system_prompt = get_mode_prompt(detected_mode)
        query_template = get_mode_query_template(detected_mode)
        context_str = self._format_context(top_results)
        user_prompt = query_template.format(context=context_str, query=question)

        raw_output = self._generate_with_system(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=4000,
        )

        return build_engine_result(
            raw_output=raw_output,
            query=question,
            mode=detected_mode,
            sources=top_results,
            total_searched=len(results),
            model_used=self.settings.resolved_llm_model,
            live_sources_fetched=live_count,
        )

    def _generate_with_system(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 2048
    ) -> str:
        """Call the LLM with a custom system prompt (for mode-specific prompts)."""
        model = self.settings.resolved_llm_model

        try:
            client = self._get_client()
        except RuntimeError as exc:
            logger.error("LLM client init failed: %s", exc)
            return (
                f"Error: Compute layer unavailable. Provider '{self._provider}' "
                f"is misconfigured.\n{exc}\n"
                "Retrieval-only mode still works (acheron query -r)."
            )

        try:
            if self._provider == "anthropic":
                response = client.messages.create(
                    model=model,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=0.2,
                    max_tokens=max_tokens,
                )
                return response.content[0].text if response.content else ""
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content or ""
        except Exception as exc:
            logger.exception(
                "LLM generation failed (Compute layer, provider=%s)", self._provider
            )
            return (
                f"Error: Compute layer unavailable. Provider '{self._provider}' "
                f"is misconfigured.\nRetrieval-only mode still works (acheron query -r). "
                f"Check your API key and provider settings.\nDetail: {exc}"
            )

    # ------------------------------------------------------------------
    # Internal — context selection
    # ------------------------------------------------------------------
    def _select_context(self, results: list[QueryResult]) -> list[QueryResult]:
        """Select the best passages for context, deduplicating by paper."""
        sorted_results = sorted(results, key=lambda r: r.relevance_score, reverse=True)

        selected: list[QueryResult] = []
        papers_included: dict[str, int] = {}
        max_per_paper = 3

        for r in sorted_results:
            if len(selected) >= self.n_context:
                break
            count = papers_included.get(r.paper_id, 0)
            if count >= max_per_paper:
                continue
            selected.append(r)
            papers_included[r.paper_id] = count + 1

        return selected

    def _format_context(self, results: list[QueryResult]) -> str:
        """Format retrieved passages into a numbered context block.

        Includes PMID/PMCID identifiers and evidence span locations
        for precise citation tracing by the LLM.
        """
        parts = []
        for i, r in enumerate(results, 1):
            author_str = ", ".join(r.authors[:3])
            if len(r.authors) > 3:
                author_str += " et al."
            doi_str = f" DOI: {r.doi}" if r.doi else ""
            pmid_str = f" PMID:{r.pmid}" if r.pmid else ""
            pmcid_str = f" PMCID:{r.pmcid}" if r.pmcid else ""
            header = f"[{i}] {author_str}. \"{r.paper_title}\".{doi_str}{pmid_str}{pmcid_str}"
            section_note = f" (Section: {r.section})" if r.section else ""
            # Include evidence span location if available
            span_note = ""
            if r.source_file and r.xpath:
                span_note = f" [{r.source_file}#{r.xpath}]"
            elif r.source_file and (r.span_start or r.span_end):
                span_note = f" [{r.source_file}:{r.span_start}-{r.span_end}]"
            parts.append(f"{header}{section_note}{span_note}\n{r.text}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Internal — LLM generation
    # ------------------------------------------------------------------
    def _generate(self, user_prompt: str, max_tokens: int = 2048) -> str:
        """Call the LLM (Compute layer). Treats API calls as scarce."""
        model = self.settings.resolved_llm_model

        try:
            client = self._get_client()
        except RuntimeError as exc:
            logger.error("LLM client init failed: %s", exc)
            return (
                f"Error: Compute layer unavailable. Provider '{self._provider}' is misconfigured.\n"
                f"{exc}\n"
                "Retrieval-only mode still works (acheron query -r)."
            )

        try:
            if self._provider == "anthropic":
                response = client.messages.create(
                    model=model,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=0.2,
                    max_tokens=max_tokens,
                )
                return response.content[0].text if response.content else ""
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content or ""
        except Exception as exc:
            logger.exception("LLM generation failed (Compute layer, provider=%s)", self._provider)
            return (
                f"Error: Compute layer unavailable. Provider '{self._provider}' is misconfigured.\n"
                f"Retrieval-only mode still works (acheron query -r). "
                f"Check your API key and provider settings.\n"
                f"Detail: {exc}"
            )

    # ------------------------------------------------------------------
    # Internal — parsing structured output
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_epistemic_sections(
        text: str,
    ) -> tuple[list[str], list[str], list[str], str]:
        """Parse EVIDENCE / INFERENCE / SPECULATION / BIOELECTRIC SCHEMATIC from LLM output.

        Returns (evidence, inference, speculation, bioelectric_schematic).
        """
        evidence: list[str] = []
        inference: list[str] = []
        speculation: list[str] = []
        schematic_lines: list[str] = []

        current: list[str] | None = None

        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue

            upper = stripped.upper()
            if any(marker in upper for marker in ["EVIDENCE", "## EVIDENCE", "**EVIDENCE"]):
                if "EXTRACTION" not in upper:
                    current = evidence
                    continue
            if any(marker in upper for marker in ["INFERENCE", "## INFERENCE", "**INFERENCE"]):
                current = inference
                continue
            if any(
                marker in upper
                for marker in ["SPECULATION", "## SPECULATION", "**SPECULATION"]
            ):
                current = speculation
                continue
            if any(
                marker in upper
                for marker in [
                    "BIOELECTRIC SCHEMATIC",
                    "## BIOELECTRIC",
                    "**BIOELECTRIC",
                ]
            ):
                current = schematic_lines
                continue
            if any(
                marker in upper
                for marker in [
                    "UNCERTAINTY",
                    "## UNCERTAINTY",
                    "**UNCERTAINTY",
                    "VALIDATION",
                    "## VALIDATION",
                ]
            ):
                current = None  # captured in raw answer
                continue

            if current is not None and stripped.lstrip("- "):
                current.append(stripped.lstrip("- "))

        schematic = "\n".join(schematic_lines) if schematic_lines else ""
        return evidence, inference, speculation, schematic

    @staticmethod
    def _parse_discovery_output(
        raw_output: str,
        query: str,
        sources: list[QueryResult],
        total_searched: int,
    ) -> DiscoveryResult:
        """Parse the full discovery loop output into structured fields."""
        evidence: list[str] = []
        inference: list[str] = []
        speculation: list[str] = []
        variables: list[StructuredVariable] = []
        hypotheses: list[Hypothesis] = []
        uncertainty: list[str] = []
        schematic_lines: list[str] = []
        validation_path: list[str] = []
        cross_species: list[str] = []

        current_section = ""

        for line in raw_output.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue

            upper = stripped.upper()

            # Detect section headers
            if any(m in upper for m in ["EVIDENCE EXTRACTION", "EVIDENCE —", "1. EVIDENCE"]):
                current_section = "evidence"
                continue
            elif any(m in upper for m in ["VARIABLE EXTRACTION", "VARIABLE —", "2. VARIABLE"]):
                current_section = "variables"
                continue
            elif any(m in upper for m in ["PATTERN COMPARISON", "PATTERN —", "3. PATTERN"]):
                current_section = "patterns"
                continue
            elif any(m in upper for m in ["HYPOTHES", "4. HYPOTHES"]):
                current_section = "hypotheses"
                continue
            elif any(
                m in upper
                for m in ["BIOELECTRIC SCHEMATIC", "5. BIOELECTRIC", "## BIOELECTRIC"]
            ):
                current_section = "schematic"
                continue
            elif any(
                m in upper
                for m in ["VALIDATION PATH", "6. VALIDATION", "VALIDATION STRATEG"]
            ):
                current_section = "validation"
                continue
            elif any(
                m in upper
                for m in ["CROSS-SPECIES", "CROSS SPECIES", "7. CROSS"]
            ):
                current_section = "cross_species"
                continue
            elif any(
                m in upper
                for m in ["UNCERTAINTY", "8. UNCERTAINTY", "## UNCERTAINTY"]
            ):
                current_section = "uncertainty"
                continue

            content = stripped.lstrip("- *")
            if not content:
                continue

            if current_section == "evidence":
                evidence.append(content)
            elif current_section == "variables":
                var = _try_parse_variable(content)
                if var:
                    variables.append(var)
                else:
                    evidence.append(content)
            elif current_section == "patterns":
                inference.append(content)
            elif current_section == "hypotheses":
                hyp = _try_parse_hypothesis(content)
                if hyp:
                    hypotheses.append(hyp)
                else:
                    speculation.append(content)
            elif current_section == "schematic":
                schematic_lines.append(content)
            elif current_section == "validation":
                validation_path.append(content)
            elif current_section == "cross_species":
                cross_species.append(content)
            elif current_section == "uncertainty":
                uncertainty.append(content)

        settings = get_settings()
        return DiscoveryResult(
            query=query,
            evidence=evidence,
            inference=inference,
            speculation=speculation,
            variables=variables,
            hypotheses=hypotheses,
            bioelectric_schematic="\n".join(schematic_lines),
            validation_path=validation_path,
            cross_species_notes=cross_species,
            sources=sources,
            model_used=settings.resolved_llm_model,
            total_chunks_searched=total_searched,
            uncertainty_notes=uncertainty,
        )


# ======================================================================
# Parsing helpers
# ======================================================================
def _try_parse_variable(text: str) -> StructuredVariable | None:
    """Attempt to parse a variable from 'name=value (unit) [source]' format."""
    if "=" not in text:
        return None
    try:
        name_part, rest = text.split("=", 1)
        name = name_part.strip()

        source_ref = ""
        if "[" in rest and "]" in rest:
            bracket_start = rest.index("[")
            bracket_end = rest.index("]") + 1
            source_ref = rest[bracket_start:bracket_end].strip()
            rest = rest[:bracket_start].strip()

        unit = ""
        if "(" in rest and ")" in rest:
            paren_start = rest.index("(")
            paren_end = rest.index(")") + 1
            unit = rest[paren_start + 1 : paren_end - 1].strip()
            value = rest[:paren_start].strip()
        else:
            value = rest.strip()

        return StructuredVariable(
            name=name,
            value=value,
            unit=unit,
            source_ref=source_ref,
        )
    except (ValueError, IndexError):
        return None


def _try_parse_hypothesis(text: str) -> Hypothesis | None:
    """Attempt to parse a hypothesis with optional confidence, predicted impact, assumptions."""
    if len(text) < 10:
        return None

    confidence = "low"
    lower = text.lower()
    if "high confidence" in lower or "(high)" in lower or "prior confidence: high" in lower:
        confidence = "high"
    elif "medium confidence" in lower or "(medium)" in lower or "prior confidence: medium" in lower:
        confidence = "medium"

    refs = []
    for match in re.finditer(r"\[(\d+)\]", text):
        refs.append(match.group(0))

    # Try to extract predicted impact if present
    predicted_impact = ""
    impact_match = re.search(
        r"predicted impact[:\s]+(.+?)(?:\.|$)", text, re.IGNORECASE
    )
    if impact_match:
        predicted_impact = impact_match.group(1).strip()

    # Try to extract assumptions if present
    assumptions: list[str] = []
    assumption_match = re.search(
        r"assumption[s]?[:\s]+(.+?)(?:\.|$)", text, re.IGNORECASE
    )
    if assumption_match:
        assumptions = [a.strip() for a in assumption_match.group(1).split(";") if a.strip()]

    return Hypothesis(
        statement=text,
        supporting_refs=refs,
        confidence=confidence,
        predicted_impact=predicted_impact,
        assumptions=assumptions,
    )

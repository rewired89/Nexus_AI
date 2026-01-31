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
You are Nexus — the Lead Systems Architect for Project Acheron.
You are not a chatbot. You are a research instrument for Information-Encoded Biology.
You treat living matter as a computational medium. Your goal is to identify the ideal \
biological "hardware" to host bioelectric "software" and non-volatile anatomical memory.

DOMAIN FRAMEWORK — BIGR (Bio-Information Genome Runtime):
- DNA = ROM: the static genetic instruction set.
- Bioelectricity = RAM + Processing: Vmem, EF, Gj are the active computational layer \
that reads/writes morphological state in real time.
- Proteome = Interface: the translation layer between genetic instructions and \
bioelectric execution.
Integrate these three layers into every analysis.

MULTI-DISCIPLINARY KNOWLEDGE BASE:
1. Genomics & Synthetic Biology: CRISPR-Cas9, optogenetics, synthetic gene circuits.
2. Cellular Biophysics: Vmem dynamics, ion channel behavior, bioelectric signaling \
across all phyla.
3. Microbiology & Mycology: bacterial quorum sensing, fungal mycelial networks as \
potential information buses.
4. Neuro-Dynamics: neural-like signaling in non-neural tissues (skin, stem cells, \
epithelia).

COMPARATIVE ANALYSIS ENGINE — Model Organisms:
For every theory or finding, evaluate across:
- Planarians: decentralized, regenerative anatomical memory.
- Xenopus laevis: large-scale bioelectric manipulation during organogenesis.
- Physarum polycephalum: bio-computational pathfinding and memory without a brain.
- Mammalian organoids: high-fidelity human-analog testing.

INTEGRATION DIRECTIVES:
When source data exists, link these elements explicitly:
- Ion pumps (e.g., H+,K+-ATPase) and their Vmem contributions
- Gap junctions (connexins in vertebrates, innexins in invertebrates)
- Membrane potential (Vmem) values with organism and tissue context
- Gene expression pathways (e.g., Wnt, Notum, piwi-1, beta-catenin)
- Regenerative / morphological outcomes
When source data does NOT exist:
- Do NOT invent numeric Vmem values or fabricate measurements.
- Infer directional effects only (hyperpolarization vs depolarization).
- Explicitly state uncertainty and what data would be needed.

MATHEMATICAL TOOLBOX (Mandatory):
When quantitative Vmem data is missing for a species, calculate the Theoretical \
Equilibrium Potential using the Nernst Equation:
  E_ion = (RT / zF) * ln([Ion]_out / [Ion]_in)
Where R = 8.314 J/(mol*K), T = temperature in Kelvin, z = ion valence, \
F = 96485 C/mol. Use standard intracellular/extracellular concentrations for \
K+, Na+, and Cl-. If exact concentrations for the species are unknown, use the \
nearest phylogenetic neighbor (e.g., Xenopus for Planarians) and adjust for \
environmental salinity. Label ALL calculated values as [HEURISTIC] — never \
present them as measured data.

ACHERON DECISION PROTOCOL:
"Low Confidence" is NOT a valid final answer. When evidence is sparse, you MUST:
1. State what IS known and what is extrapolated.
2. Apply first-principles reasoning (physics, chemistry, information theory).
3. Commit to a Strategic Recommendation with explicit assumptions labeled.
4. Provide a falsification path — what evidence would overturn this recommendation.

PROTOCOL SPECIFICATION FORMAT:
When recommending experimental approaches, generate Technical Spec Sheets:
- Write Method: (e.g., optogenetic stimulation, ionophore bath, galvanotaxis)
- Read Method: (e.g., voltage-sensitive dyes, micro-electrode arrays, sequencing)
- Logic Gate Equivalent: how the substrate performs NOT/AND via bioelectric flux
- Estimated SNR: signal-to-noise ratio assessment for the read method
- Error Correction: biological redundancy mechanism and its RAID-level equivalent

ERROR CORRECTION & FAULT TOLERANCE:
Map biological regeneration to information-theoretic fault tolerance:
- Target Morphology = Checksum: the stored pattern validates data integrity.
- Regeneration = RAID rebuild: tissue repair from distributed bioelectric state.
- Colony/tissue redundancy = Replication factor.
Always specify which RAID level best models the organism's fault tolerance.

THREE-LAYER ARCHITECTURE:
Layer 1 (Knowledge): The source passages below are immutable primary sources from \
PubMed, Semantic Scholar, bioRxiv, arXiv, and PhysioNet. Never hallucinate facts. \
Cite [1], [2], etc. for every factual claim.
Layer 2 (Synthesis): Retrieval-augmented reasoning over structured bioelectric \
variables. Extract: Vmem values, EF measurements, Gj/connexin expression, ion \
channel types (K+, Na+, Ca2+, Cl-), perturbation methods, outcomes.
Layer 3 (Discovery): Comparative analysis, cross-species reasoning, pattern \
detection, hypothesis generation. Every discovery output must include a bioelectric \
schematic and a validation path.

SCIENCE-FIRST OUTPUT RULES:
- Every output sentence must be tagged:
  [EVIDENCE] — directly supported by retrieved sources (cite PMID/DOI).
  [INFERENCE] — logical bridge across sources, no new facts introduced.
  [SPECULATION] — hypothesis proposal; must include a falsification plan.
  [DATA GAP] — missing measurement; include targeted collection queries.
- Structured, stepwise output. No motivational language.
- Cite [1], [2], etc. for every factual claim (PMID or DOI where available).
- For every hypothesis: causal chain, required missing measurements, \
1 falsification experiment.
- Cross-species comparison is mandatory when evidence spans organisms.
- If organism-specific sources are absent, output [DATA GAP] and generate \
targeted collection queries instead of filling with analogies.
- Never output numeric Vmem/EF/ion concentrations unless retrieved from a \
source. If unknown: "unknown; requires measurement".
- You are a research engine, not a chatbot.

CONSTRAINTS:
- No diagnosis or treatment advice.
- Prefer public, de-identified data.
- If sources are insufficient, state so directly and identify missing data.
- Prefer "I cannot support that with sources" over inventing details.
- "Low Confidence" is NOT a valid final answer. Commit to a recommendation \
with labeled assumptions and a falsification path."""

QUERY_TEMPLATE = """\
Retrieved source passages from the bioelectricity and biomedical research corpus:

=== SOURCE PASSAGES ===
{context}
========================

Query: {query}

Respond with the following mandatory structure. Tag EVERY sentence with \
[EVIDENCE], [INFERENCE], [SPECULATION], or [DATA GAP].

I. EVIDENCE SNAPSHOT
- Bullet list of findings directly from sources. Each tagged [EVIDENCE].
- Cite PMID/DOI for each claim. Format: "[EVIDENCE] claim text [1] (PMID:xxx)."
- If organism-specific data is absent, state [DATA GAP] instead of generalizing.

II. DATA GAPS
- Bullet list of missing measurements, tagged [DATA GAP].
- For each gap, state what measurement is needed and in what organism/tissue.

III. HYPOTHESES (max 3)
- Each tagged [SPECULATION].
- Each must include:
  a) Causal chain: [ion pump/channel] -> [Vmem/EF change] -> [pathway] -> [outcome]
  b) Required missing measurements
  c) 1 falsification experiment (what result would disprove this)

IV. BIOELECTRIC SCHEMATIC
- BIGR layers: ROM (genetic) / RAM (bioelectric) / Interface (proteomic)
- Format: "[Trigger] -> [Bioelectric change] -> [Downstream pathway] -> [Outcome]"
- Label each component [EVIDENCED], [INFERRED], or [SPECULATIVE].
- If insufficient data, state what is missing. Do NOT invent Vmem values.

V. CROSS-SPECIES NOTES
- Compare findings across model organisms where evidence exists.
- State where evidence transfers and where gaps remain.

VI. MINIMAL WET-LAB TEST (MVP)
- 4-6 step protocol for a 1-2 week baseline experiment.
- Materials list, expected outcomes, failure modes.

VII. NEXT COLLECTION QUERIES
- 5-10 exact PubMed/PMC/bioRxiv queries to fill the data gaps identified above."""

DISCOVERY_TEMPLATE = """\
Retrieved source passages from the bioelectricity and biomedical research corpus:

=== SOURCE PASSAGES ===
{context}
========================

Research query: {query}

Tag EVERY output sentence with [EVIDENCE], [INFERENCE], [SPECULATION], or [DATA GAP].
Execute the discovery loop with ALL of the following mandatory sections:

1. EVIDENCE EXTRACTION
Key findings directly from the sources. Cite each with [1], [2], etc.

2. VARIABLE EXTRACTION
Structured bioelectric variables. Format each as name=value (unit) [source].
Prioritize: Vmem (membrane voltage), EF (electric fields), Gj (gap junctional \
conductance), ion channel types (K+, Na+, Ca2+, Cl-), perturbations, outcomes.
Include organism and cell type context for each variable.
When exact values are absent, calculate E_ion using the Nernst Equation:
  E_ion = (RT / zF) * ln([Ion]_out / [Ion]_in)
Use nearest phylogenetic neighbor concentrations if exact values are unknown.
Label ALL calculated values as [HEURISTIC] — never present as measured data.

3. BIGR INTEGRATION
Map findings to the Bio-Information Genome Runtime layers:
- ROM (DNA): relevant genes, pathways (Wnt, Notum, piwi-1, connexins, innexins)
- RAM (Bioelectricity): Vmem, EF, Gj changes and their causal roles
- Interface (Proteome): ion pumps (H+,K+-ATPase), channels, gap junction proteins
Link these layers explicitly where source data supports it.

4. PATTERN COMPARISON
Compare findings across sources and across species:
Planaria <-> Xenopus <-> Physarum <-> Mammalian organoids.
Note agreements, conflicts, and gaps. Identify conserved bioelectric mechanisms.

5. SUBSTRATE SELECTION MATRIX
When hypotheses involve experimental testing, evaluate model organisms:
| Criteria | Planarian | Xenopus | Physarum | Organoid |
| Ease of Vmem manipulation | ... | ... | ... | ... |
| Data persistence / memory | ... | ... | ... | ... |
| I/O speed (response time) | ... | ... | ... | ... |
| Relevance to query | ... | ... | ... | ... |
Rate each as Low/Medium/High based on sources. If no data, state "No data".

6. HYPOTHESES
Generate testable hypotheses from the patterns. For each hypothesis state:
- Prior Confidence: low/medium/high (based on evidence density)
- Predicted Impact: what changes in our understanding if this is true
- Assumptions: what must hold for this hypothesis to be valid
- Supporting references: [1], [2], etc.

7. BIOELECTRIC SCHEMATIC
Describe the hypothesized bioelectric circuit in a structured format:
"[Trigger] -> [Bioelectric change (Vmem/EF/Gj)] -> [Downstream pathway] -> [Outcome]"
If multiple circuits are relevant, describe each. Label components as \
[EVIDENCED], [INFERRED], or [SPECULATIVE].
Include logic gate equivalents where applicable (NOT/AND via bioelectric flux).

8. PROTOCOL SPECIFICATION
For the recommended experimental approach, generate a Technical Spec Sheet:
- Write Method: (optogenetic stimulation, ionophore bath, galvanotaxis, etc.)
- Read Method: (voltage-sensitive dyes, micro-electrode arrays, sequencing, etc.)
- Logic Gate Equivalent: how the substrate performs NOT/AND via bioelectric flux
- Estimated SNR: signal-to-noise ratio for the read method
- Error Correction: biological redundancy mechanism

9. VALIDATION PATH
Propose specific, low-cost ways to test the hypotheses:
- Re-analysis of existing datasets
- Computational simulations
- Targeted experimental designs
- Cross-species comparison strategies

10. FAULT TOLERANCE MAPPING
Map the biological system's regeneration to RAID-level redundancy:
- Target Morphology = Checksum (stored pattern validates data integrity)
- Regeneration = RAID rebuild (tissue repair from distributed state)
- Colony/tissue redundancy = Replication factor
Specify which RAID level best models the organism's fault tolerance.

11. CROSS-SPECIES NOTES
Where does the evidence transfer across organisms? Where are gaps?

12. UNCERTAINTY & STRATEGIC RECOMMENDATION
Explicit gaps, missing variables, conflicting evidence, what data would resolve them.
"Low Confidence" is NOT a valid final answer. Commit to a strategic recommendation \
with labeled assumptions and a falsification path.

Be precise. No filler. Every claim must trace to a source number. \
Use the Nernst Equation for heuristic baselines when measured Vmem is absent."""


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
            if self._provider == "anthropic":
                hint = "Set ANTHROPIC_API_KEY in your environment or .env file."
            else:
                hint = "Set ACHERON_LLM_API_KEY or OPENAI_API_KEY in your environment or .env file."
            raise RuntimeError(
                f"No API key configured for provider '{self._provider}'. {hint}"
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

        Science-First pipeline stages:
        A. Parse query into entities/measurements/constraints
        B. Retrieve from local corpus
        C. Score and filter evidence (organism match, primary data, etc.)
        D. Generate structured output with claim extraction
        E. Synthesize: evidence, data gaps, hypotheses, collection queries
        """
        from acheron.rag.query_parser import generate_collection_queries, parse_query

        n = n_results or self.n_retrieve

        # Stage A — Query understanding
        parsed = parse_query(question)
        logger.info("Query parsed: %s", parsed.summary())

        # Stage B — Retrieve
        results = self.store.search(
            query=question, n_results=n, filter_source=filter_source
        )

        if not results:
            collection_queries = generate_collection_queries(parsed)
            return DiscoveryResult(
                query=question,
                evidence=["No sources retrieved."],
                uncertainty_notes=[
                    "Library contains no material for this query.",
                    "Suggested collection queries:",
                    *[f"  - {q}" for q in collection_queries],
                ],
                model_used=self.settings.resolved_llm_model,
            )

        # Stage C — Score and filter (via _select_context with science-first)
        top_results = self._select_context(
            results, target_organism=parsed.organism_constraint
        )

        # Stage D+E — Discovery compute
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
        from acheron.models import HypothesisEngineResult, NexusMode
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
    def _select_context(
        self,
        results: list[QueryResult],
        target_organism: str = "",
    ) -> list[QueryResult]:
        """Select the best passages for context, deduplicating by paper.

        When SCIENCE_FIRST_MODE is enabled, uses the evidence scoring
        function (Stage C) instead of raw vector similarity.
        """
        settings = get_settings()
        organism = target_organism or settings.organism_strict

        if settings.science_first_mode:
            from acheron.rag.science_filter import rank_and_filter

            strict = organism != "any"
            scored = rank_and_filter(
                results,
                target_organism=organism if organism != "any" else "planarian",
                top_k=self.n_context,
                strict_organism=strict,
            )
            if scored:
                return [s.result for s in scored]
            # Fallback: if strict filtering returned nothing, use non-strict
            scored = rank_and_filter(
                results,
                target_organism=organism if organism != "any" else "planarian",
                top_k=self.n_context,
                strict_organism=False,
            )
            if scored:
                return [s.result for s in scored]

        # Fallback: original selection logic
        sorted_results = sorted(
            results, key=lambda r: r.relevance_score, reverse=True
        )

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
            if any(
                marker in upper
                for marker in [
                    "EVIDENCE SNAPSHOT", "I. EVIDENCE",
                    "EVIDENCE", "## EVIDENCE", "**EVIDENCE",
                ]
            ):
                if "EXTRACTION" not in upper:
                    current = evidence
                    continue
            if any(
                marker in upper
                for marker in [
                    "DATA GAP", "II. DATA GAP", "II. WHAT WE",
                ]
            ):
                current = None  # captured in raw answer
                continue
            if any(
                marker in upper
                for marker in ["INFERENCE", "## INFERENCE", "**INFERENCE"]
            ):
                current = inference
                continue
            if any(
                marker in upper
                for marker in [
                    "HYPOTHES", "III. HYPOTHES",
                    "SPECULATION", "## SPECULATION", "**SPECULATION",
                ]
            ):
                current = speculation
                continue
            if any(
                marker in upper
                for marker in [
                    "BIOELECTRIC SCHEMATIC", "IV. BIOELECTRIC",
                    "## BIOELECTRIC", "**BIOELECTRIC",
                ]
            ):
                current = schematic_lines
                continue
            if any(
                marker in upper
                for marker in [
                    "CROSS-SPECIES", "V. CROSS",
                    "UNCERTAINTY", "## UNCERTAINTY", "**UNCERTAINTY",
                    "VALIDATION", "## VALIDATION",
                    "MINIMAL WET-LAB", "VI. MINIMAL",
                    "NEXT COLLECTION", "VII. NEXT",
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
            elif any(m in upper for m in ["BIGR INTEGRATION", "BIGR —", "3. BIGR"]):
                current_section = "bigr"
                continue
            elif any(m in upper for m in ["PATTERN COMPARISON", "PATTERN —", "4. PATTERN"]):
                current_section = "patterns"
                continue
            elif any(
                m in upper
                for m in ["SUBSTRATE SELECTION", "5. SUBSTRATE"]
            ):
                current_section = "substrate"
                continue
            elif any(m in upper for m in ["HYPOTHES", "6. HYPOTHES"]):
                current_section = "hypotheses"
                continue
            elif any(
                m in upper
                for m in ["BIOELECTRIC SCHEMATIC", "7. BIOELECTRIC", "## BIOELECTRIC"]
            ):
                current_section = "schematic"
                continue
            elif any(
                m in upper
                for m in [
                    "PROTOCOL SPECIFICATION", "8. PROTOCOL",
                    "## PROTOCOL SPEC",
                ]
            ):
                current_section = "protocol"
                continue
            elif any(
                m in upper
                for m in ["VALIDATION PATH", "9. VALIDATION", "VALIDATION STRATEG"]
            ):
                current_section = "validation"
                continue
            elif any(
                m in upper
                for m in [
                    "FAULT TOLERANCE", "10. FAULT",
                    "RAID", "## FAULT TOLERANCE",
                ]
            ):
                current_section = "fault_tolerance"
                continue
            elif any(
                m in upper
                for m in ["CROSS-SPECIES", "CROSS SPECIES", "11. CROSS"]
            ):
                current_section = "cross_species"
                continue
            elif any(
                m in upper
                for m in [
                    "UNCERTAINTY", "12. UNCERTAINTY", "## UNCERTAINTY",
                    "STRATEGIC RECOMMEND",
                ]
            ):
                current_section = "uncertainty"
                continue
            elif any(
                m in upper
                for m in [
                    "MINIMAL WET-LAB", "MINIMAL VIABLE",
                    "DATA GAP", "NEXT COLLECTION",
                ]
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
            elif current_section == "bigr":
                inference.append(content)
            elif current_section == "patterns":
                inference.append(content)
            elif current_section == "substrate":
                cross_species.append(content)
            elif current_section == "hypotheses":
                hyp = _try_parse_hypothesis(content)
                if hyp:
                    hypotheses.append(hyp)
                else:
                    speculation.append(content)
            elif current_section == "schematic":
                schematic_lines.append(content)
            elif current_section == "protocol":
                validation_path.append(content)
            elif current_section == "validation":
                validation_path.append(content)
            elif current_section == "fault_tolerance":
                inference.append(content)
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

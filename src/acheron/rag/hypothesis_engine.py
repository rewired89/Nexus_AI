"""Evidence-Bound Hypothesis Engine for Nexus.

Implements:
1. Evidence Graph construction (RAG + Knowledge Graph)
2. Claim Verification & Agreement Scoring
3. Abductive Reasoning / IBE (Inference to the Best Explanation)
4. Falsification-First Output (Popper-style)
5. Uncertainty Calibration
6. Guardrails Against Hallucination

Three modes:
  MODE 1 (evidence)   — evidence-grounded summary with citations
  MODE 2 (hypothesis) — IBE hypothesis generation + falsification
  MODE 3 (synthesis)  — systems synthesis / architecture proposals
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from acheron.models import (
    ClaimStatus,
    EvidenceClaim,
    EvidenceEdge,
    EvidenceGraph,
    HypothesisEngineResult,
    NexusMode,
    QueryResult,
    RankedHypothesis,
)

logger = logging.getLogger(__name__)

# ======================================================================
# Mode detection
# ======================================================================

_HYPOTHESIS_TRIGGERS = [
    "theor", "hypothes", "hypothesize", "what could",
    "what might", "why does", "why do", "why would",
    "what if", "propose", "speculate", "explain why",
    "what mechanism", "how could", "how might",
]

_SYNTHESIS_TRIGGERS = [
    "design", "protocol", "threat model", "architecture",
    "build", "implement", "engineer", "construct",
    "propose a system", "develop a",
]


def detect_mode(query: str, explicit_mode: Optional[str] = None) -> NexusMode:
    """Detect the operating mode from query text or explicit parameter.

    Trigger rules:
    - Explicit mode always wins.
    - If query contains hypothesis/theory language → MODE 2
    - If query contains design/protocol language → MODE 3
    - Otherwise → MODE 1 (evidence-grounded)
    """
    if explicit_mode:
        try:
            return NexusMode(explicit_mode.lower())
        except ValueError:
            pass

    lower = query.lower()
    for trigger in _SYNTHESIS_TRIGGERS:
        if trigger in lower:
            return NexusMode.SYNTHESIS
    for trigger in _HYPOTHESIS_TRIGGERS:
        if trigger in lower:
            return NexusMode.HYPOTHESIS

    return NexusMode.EVIDENCE


# ======================================================================
# System prompts for each mode
# ======================================================================

_BASE_IDENTITY = """\
You are Nexus — the Lead Systems Architect for Project Acheron.
You are not a chatbot. You are a research instrument for Information-Encoded Biology.
You treat living matter as a computational medium.

BIGR FRAMEWORK (Bio-Information Genome Runtime):
- DNA = ROM: the static genetic instruction set.
- Bioelectricity = RAM + Processing: Vmem, EF, Gj are the active computational \
layer that reads/writes morphological state in real time.
- Proteome = Interface: translation between genetic instructions and bioelectric \
execution.

MULTI-DISCIPLINARY KNOWLEDGE:
1. Genomics & Synthetic Biology: CRISPR-Cas9, optogenetics, synthetic gene circuits.
2. Cellular Biophysics: Vmem dynamics, ion channel behavior, bioelectric signaling.
3. Microbiology & Mycology: quorum sensing, mycelial networks as information buses.
4. Neuro-Dynamics: neural-like signaling in non-neural tissues.

COMPARATIVE ANALYSIS ENGINE — always evaluate across:
- Planarians: decentralized, regenerative anatomical memory.
- Xenopus laevis: large-scale bioelectric manipulation during organogenesis.
- Physarum polycephalum: bio-computational pathfinding and memory without a brain.
- Mammalian organoids: high-fidelity human-analog testing.

INTEGRATION DIRECTIVES:
When data exists, link: ion pumps (H+,K+-ATPase), gap junctions (connexins/innexins), \
Vmem, gene expression (Wnt, Notum, piwi-1), and regenerative outcomes.
When data does NOT exist: do NOT invent numeric Vmem values. Infer directional \
effects only (hyperpolarization vs depolarization). Explicitly state uncertainty.

MATHEMATICAL TOOLBOX:
When Vmem data is missing, calculate E_ion using the Nernst Equation:
  E_ion = (RT / zF) * ln([Ion]_out / [Ion]_in)
R = 8.314 J/(mol*K), T = temperature in K, z = ion valence, F = 96485 C/mol.
Use nearest phylogenetic neighbor concentrations when exact values are unknown.
Label ALL calculated values as [HEURISTIC].

ACHERON DECISION PROTOCOL:
"Low Confidence" is NOT a valid final answer. When evidence is sparse:
1. State what IS known and what is extrapolated.
2. Apply first-principles reasoning (physics, chemistry, information theory).
3. Commit to a Strategic Recommendation with labeled assumptions.
4. Provide a falsification path.

ERROR CORRECTION & FAULT TOLERANCE:
Map regeneration to RAID-level redundancy:
- Target Morphology = Checksum (pattern validates data integrity).
- Regeneration = RAID rebuild (repair from distributed bioelectric state).
- Colony/tissue redundancy = Replication factor.

ALGORITHMIC FRAMEWORKS:
1. Graph Reasoning: Treat tissues as spatial graphs (cells=nodes, Gj/EF=edges). \
Predict bioelectric state stability across the graph topology.
2. Structural Grammars: Analyze voltage gradient "shape" to predict how ion \
channel density maps to 3D morphological checksums.
3. Multi-Agent Research: Operate as Scraper (data extraction), Physicist (Nernst, \
thermodynamics), Information Theorist (Shannon entropy, channel capacity), and \
Critic (falsification) — each section reflects which agent produced it.

NO-NUMERIC-INVENTION POLICY (ABSOLUTE):
You may NOT output any numeric value for biological parameters unless:
  (a) it is directly reported in a cited source (PMID/DOI), OR
  (b) it is a pure physics bound using ONLY universal constants and stated assumptions.
If neither applies: "UNKNOWN — requires measurement."

BIM SPECIFICATION (Biological Information Module):
For any claimed "biological bit," specify measurable parameters:
- State Stability (T_hold): formula T_half = R_m * C_m * ln(2). \
Valid ONLY with measured R_m and C_m. If unmeasured: state measurement plan.
- Switching Energy (E_bit): pure physics formula. Valid ONLY with measured inputs.
- Error Rate / BER: UNKNOWN unless single-channel recordings exist for cell type.
- Shannon Entropy: H = log2(N_states). N_states requires bistability assay.
- Channel Capacity: C = B * log2(1 + SNR). B and SNR require gap junction recording.
For EACH: cite measured value OR output "UNKNOWN" + measurement plan.
Map to Hardware Library: CPU (Nav/Kv), RAM (Vmem gradient), SSD (Innexin).

GUARDRAILS:
- Never present hypotheses as facts.
- Cite [1], [2], etc. for every factual claim.
- If evidence is weak, say so explicitly and identify what evidence is missing.
- Prefer "I cannot support that with sources" over inventing details.
- Do not provide diagnosis or treatment advice.
"""

EVIDENCE_PROMPT = _BASE_IDENTITY + """
MODE: EVIDENCE-GROUNDED (MODE 1)
Summarize what is known, with citations. Structure your output as follows:

EVIDENCE CLAIMS
For each key claim from the sources, output as a structured entry:
- CLAIM: [subject] [predicate] [object]
  REFS: [1], [2], ...
  SUPPORT_COUNT: (number of independent sources supporting this)
  CONTRADICTION_COUNT: (number of sources contradicting this)
  STATUS: supported | mixed | unclear | unsupported
  STUDY_TYPES: review, primary, preprint

CLAIM RELATIONSHIPS
For relationships between claims:
- [claim_subject] [supports|contradicts|depends-on] [other_claim_subject]

AGREEMENT SUMMARY
Overall assessment of evidence strength for this query.

UNCERTAINTY
- What the sources do not address
- Where data is insufficient
- Conflicting findings

CONFIDENCE: [0-100]
CONFIDENCE_JUSTIFICATION: [brief reason]

NEXT_QUERIES
Exact PubMed/bioRxiv/arXiv search queries that would fill evidence gaps.
"""

HYPOTHESIS_PROMPT = _BASE_IDENTITY + """
MODE: HYPOTHESIS GENERATION (MODE 2)
Using Inference to the Best Explanation (IBE), generate ranked hypotheses.

EVIDENCE CLAIMS
(Same structured format as MODE 1 — extract claims first)

CLAIM RELATIONSHIPS
(Same structured format as MODE 1)

HYPOTHESES
Generate at least 2 alternative hypotheses plus a leading hypothesis. For EACH:

HYPOTHESIS: [H1] [statement]
RANK: [1, 2, 3...]
EXPLANATORY_POWER: [0.0-1.0] (covers how many supported claims)
SIMPLICITY: [0.0-1.0] (fewest extra assumptions)
CONSISTENCY: [0.0-1.0] (avoids contradicting strong evidence)
MECHANISTIC_PLAUSIBILITY: [0.0-1.0] (fits known biology/physics/control theory)
RATIONALE: [brief explanation]
PREDICTIONS: what would we expect to observe if true
FALSIFIERS: what result would falsify this
MINIMAL_TEST: minimal experiment or observation to test (high-level, conceptual)
CONFIDENCE: [0-100]
CONFIDENCE_JUSTIFICATION: [brief reason]
ASSUMPTIONS: A1, A2, ...
KNOWN_UNKNOWNS: what we don't know that matters
FAILURE_MODES: most likely ways this hypothesis could be wrong
REFS: [1], [2], ...

SUBSTRATE SELECTION MATRIX
For hypotheses involving experimental testing, evaluate model organisms:
| Criteria | Planarian | Xenopus | Physarum | Organoid |
| Ease of Vmem manipulation | ... | ... | ... | ... |
| Data persistence / memory | ... | ... | ... | ... |
| I/O speed (response time) | ... | ... | ... | ... |
| Relevance to hypothesis | ... | ... | ... | ... |
Rate as Low/Medium/High. If no source data, state "No data".

PROTOCOL SPECIFICATION
For the recommended experimental approach:
- Write Method: (optogenetic stimulation, ionophore bath, galvanotaxis, etc.)
- Read Method: (voltage-sensitive dyes, micro-electrode arrays, sequencing, etc.)
- Logic Gate Equivalent: how the substrate performs NOT/AND via bioelectric flux
- Estimated SNR: signal-to-noise ratio for the read method
- Error Correction: biological redundancy mechanism

FAULT TOLERANCE MAPPING
- Target Morphology = Checksum (stored pattern validates data integrity)
- Regeneration = RAID rebuild (tissue repair from distributed state)
- Specify RAID level equivalent for the organism's fault tolerance

BIM SPECIFICATION
For any claimed bioelectric state or "biological bit":
- For EACH parameter (T_hold, E_bit, BER, entropy, capacity):
  Cite measured value OR state "UNKNOWN — requires [experiment]."
- State the pure physics formula. Do NOT compute from unmeasured inputs.
- Map to Hardware Library: CPU (Nav/Kv), RAM (Vmem), SSD (Innexin).

UNCERTAINTY
- Explicit gaps, missing variables, conflicting evidence
- "Low Confidence" is NOT a final answer — commit to a recommendation

OVERALL_CONFIDENCE: [0-100]
OVERALL_JUSTIFICATION: [brief reason]

NEXT_QUERIES
Exact search queries for PubMed/bioRxiv/arXiv to test or refine hypotheses.
"""

SYNTHESIS_PROMPT = _BASE_IDENTITY + """
MODE: SYSTEMS SYNTHESIS (MODE 3)
Propose architectures, protocols, or system designs based on evidence + labeled \
assumptions. Frame designs using the BIGR model (DNA=ROM, Bioelectricity=RAM, \
Proteome=Interface).

EVIDENCE CLAIMS
(Same structured format as MODE 1 — extract claims first)

CLAIM RELATIONSHIPS
(Same structured format as MODE 1)

HYPOTHESES
Generate testable hypotheses underlying the proposed design. For EACH:
(Same structured format as MODE 2)

SYSTEM DESIGN
Describe the proposed architecture/protocol/system with explicit labels for:
- [EVIDENCED]: components supported by sources
- [ASSUMED]: components that rely on assumptions
- [SPECULATIVE]: components that go beyond current evidence
Map each component to the BIGR layer it operates on (ROM/RAM/Interface).

SUBSTRATE SELECTION MATRIX
Evaluate which model organism is best suited for validating this design:
| Criteria | Planarian | Xenopus | Physarum | Organoid |
| Ease of Vmem manipulation | ... | ... | ... | ... |
| Data persistence / memory | ... | ... | ... | ... |
| I/O speed (response time) | ... | ... | ... | ... |
| Relevance to design | ... | ... | ... | ... |
Rate as Low/Medium/High. If no source data, state "No data".

PROTOCOL SPECIFICATION
For the recommended implementation:
- Write Method: (optogenetic stimulation, ionophore bath, galvanotaxis, etc.)
- Read Method: (voltage-sensitive dyes, micro-electrode arrays, sequencing, etc.)
- Logic Gate Equivalent: how the substrate performs NOT/AND via bioelectric flux
- Estimated SNR: signal-to-noise ratio for the read method
- Error Correction: biological redundancy mechanism

VALIDATION PATH
How to validate the design:
- Re-analysis of existing datasets
- Computational simulations
- Targeted experiments
- Cross-species comparison strategies

FAULT TOLERANCE MAPPING
- Target Morphology = Checksum (stored pattern validates data integrity)
- Regeneration = RAID rebuild (tissue repair from distributed state)
- Colony/tissue redundancy = Replication factor
- Specify RAID level equivalent for the system's fault tolerance

BIM SPECIFICATION
For any proposed bioelectric computation:
- For EACH parameter (T_hold, E_bit, BER, entropy, capacity):
  Cite measured value OR state "UNKNOWN — requires [experiment]."
- State the pure physics formula. Do NOT compute from unmeasured inputs.
- Map to Hardware Library: CPU (Nav/Kv), RAM (Vmem), SSD (Innexin).

UNCERTAINTY & STRATEGIC RECOMMENDATION
- "Low Confidence" is NOT a final answer — commit to a recommendation
- State labeled assumptions and a falsification path

OVERALL_CONFIDENCE: [0-100]
OVERALL_JUSTIFICATION: [brief reason]

NEXT_QUERIES
Exact search queries for additional evidence.
"""


def get_mode_prompt(mode: NexusMode) -> str:
    """Return the system prompt for the given mode."""
    if mode == NexusMode.HYPOTHESIS:
        return HYPOTHESIS_PROMPT
    elif mode == NexusMode.SYNTHESIS:
        return SYNTHESIS_PROMPT
    return EVIDENCE_PROMPT


def get_mode_query_template(mode: NexusMode) -> str:
    """Return the query template for the given mode."""
    return """\
Retrieved source passages from the bioelectricity and biomedical research corpus:

=== SOURCE PASSAGES ===
{context}
========================

Research query: {query}

Execute the full {mode_label} analysis using ALL the mandatory sections described \
in your system prompt. Be precise. Every claim must trace to a source number.""".format(
        context="{context}",
        query="{query}",
        mode_label=mode.value.upper(),
    )


# ======================================================================
# Output parsing — extract structured data from LLM response
# ======================================================================

def parse_evidence_graph(raw_output: str) -> EvidenceGraph:
    """Parse EVIDENCE CLAIMS and CLAIM RELATIONSHIPS from LLM output."""
    claims: list[EvidenceClaim] = []
    edges: list[EvidenceEdge] = []

    claim_id_counter = 0

    # Parse claims
    in_claims = False
    in_relationships = False
    current_claim: dict = {}

    for line in raw_output.split("\n"):
        stripped = line.strip()
        upper = stripped.upper()

        # Section detection
        if "EVIDENCE CLAIM" in upper and not in_relationships:
            in_claims = True
            in_relationships = False
            continue
        if "CLAIM RELATIONSHIP" in upper:
            # Flush pending claim
            if current_claim:
                claims.append(_build_claim(current_claim, claim_id_counter))
                claim_id_counter += 1
                current_claim = {}
            in_claims = False
            in_relationships = True
            continue
        if any(
            header in upper
            for header in [
                "AGREEMENT SUMMARY", "UNCERTAINTY", "HYPOTHES",
                "CONFIDENCE:", "NEXT_QUER", "SYSTEM DESIGN",
                "VALIDATION PATH", "OVERALL_CONFIDENCE",
                "SUBSTRATE SELECTION", "BIGR",
                "PROTOCOL SPECIFICATION", "FAULT TOLERANCE",
                "HEURISTIC BASELINE", "STRATEGIC RECOMMEND",
                "BIM QUANTIFICATION", "BIM SPECIFICATION",
                "GRAPH TOPOLOGY",
                "HARDWARE SPEC",
            ]
        ):
            if current_claim:
                claims.append(_build_claim(current_claim, claim_id_counter))
                claim_id_counter += 1
                current_claim = {}
            in_claims = False
            in_relationships = False
            continue

        if not stripped:
            continue

        # Parse claim fields
        if in_claims:
            if stripped.startswith("- CLAIM:") or stripped.startswith("CLAIM:"):
                if current_claim:
                    claims.append(_build_claim(current_claim, claim_id_counter))
                    claim_id_counter += 1
                current_claim = {"text": stripped.split(":", 1)[1].strip()}
            elif stripped.startswith("REFS:") or stripped.startswith("- REFS:"):
                current_claim["refs"] = re.findall(r"\[(\d+)\]", stripped)
            elif "SUPPORT_COUNT" in upper:
                m = re.search(r"(\d+)", stripped)
                if m:
                    current_claim["support"] = int(m.group(1))
            elif "CONTRADICTION_COUNT" in upper:
                m = re.search(r"(\d+)", stripped)
                if m:
                    current_claim["contradiction"] = int(m.group(1))
            elif "STATUS:" in upper:
                for status in ["supported", "mixed", "unclear", "unsupported"]:
                    if status in stripped.lower():
                        current_claim["status"] = status
                        break
            elif "STUDY_TYPE" in upper:
                current_claim["study_types"] = [
                    t.strip() for t in stripped.split(":", 1)[1].split(",") if t.strip()
                ] if ":" in stripped else []

        # Parse edges
        if in_relationships:
            for rel in ["supports", "contradicts", "depends-on", "depends on"]:
                if rel in stripped.lower():
                    parts = re.split(rf"\s+{rel}\s+", stripped.lstrip("- "), flags=re.IGNORECASE)
                    if len(parts) == 2:
                        edges.append(
                            EvidenceEdge(
                                source_claim=parts[0].strip(),
                                target_claim=parts[1].strip(),
                                relation=rel.replace(" ", "-"),
                            )
                        )
                    break

    # Flush final claim
    if current_claim:
        claims.append(_build_claim(current_claim, claim_id_counter))

    return EvidenceGraph(claims=claims, edges=edges)


def _build_claim(data: dict, idx: int) -> EvidenceClaim:
    """Build an EvidenceClaim from parsed fields."""
    text = data.get("text", "")
    # Try to split into subject-predicate-object
    subject, predicate, obj = _parse_spo(text)

    support = data.get("support", 0)
    contradiction = data.get("contradiction", 0)
    total = support + contradiction
    agreement = support / total if total > 0 else 0.0

    status_str = data.get("status", "unclear")
    try:
        status = ClaimStatus(status_str)
    except ValueError:
        status = ClaimStatus.UNCLEAR

    return EvidenceClaim(
        claim_id=f"C{idx}",
        subject=subject,
        predicate=predicate,
        object=obj,
        source_refs=[f"[{r}]" for r in data.get("refs", [])],
        support_count=support,
        contradiction_count=contradiction,
        agreement_score=agreement,
        status=status,
        study_types=data.get("study_types", []),
    )


def _parse_spo(text: str) -> tuple[str, str, str]:
    """Try to parse subject–predicate–object from a claim text.

    Falls back to (full_text, '', '') if parsing fails.
    """
    # Try splitting on common predicate verbs
    for pattern in [
        r"^(.+?)\s+(regulates|controls|inhibits|activates|modulates|causes|"
        r"induces|affects|drives|encodes|maintains|signals|mediates|"
        r"promotes|suppresses|blocks|enables|requires)\s+(.+)$",
    ]:
        m = re.match(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()

    # Try splitting on "is" / "are"
    m = re.match(r"^(.+?)\s+(is|are)\s+(.+)$", text, re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()

    return text, "", ""


def parse_hypotheses(raw_output: str) -> list[RankedHypothesis]:
    """Parse HYPOTHESES section from LLM output into ranked hypotheses."""
    hypotheses: list[RankedHypothesis] = []
    current: dict = {}
    in_hypotheses = False

    for line in raw_output.split("\n"):
        stripped = line.strip()
        upper = stripped.upper()

        # Detect start of hypotheses section header (but not individual HYPOTHESIS: entries)
        if "HYPOTHES" in upper and (
            upper.startswith("HYPOTHES") or upper.startswith("#")
            or upper.startswith("*") or upper.startswith("2.")
            or upper.startswith("4.")
        ) and not stripped.startswith("HYPOTHESIS:") and not stripped.startswith("- HYPOTHESIS:"):
            in_hypotheses = True
            continue

        # Detect end of hypotheses section
        if in_hypotheses and any(
            header in upper
            for header in [
                "UNCERTAINTY", "OVERALL_CONFIDENCE", "NEXT_QUER",
                "SYSTEM DESIGN", "VALIDATION PATH", "BIOELECTRIC SCHEMATIC",
                "SUBSTRATE SELECTION", "PROTOCOL SPECIFICATION",
                "FAULT TOLERANCE", "STRATEGIC RECOMMEND",
                "BIM QUANTIFICATION", "BIM SPECIFICATION",
                "GRAPH TOPOLOGY",
                "HARDWARE SPEC",
            ]
        ):
            if current:
                hypotheses.append(_build_hypothesis(current, len(hypotheses)))
                current = {}
            in_hypotheses = False
            continue

        if not in_hypotheses or not stripped:
            continue

        # Parse hypothesis fields
        if stripped.startswith("HYPOTHESIS:") or stripped.startswith("- HYPOTHESIS:"):
            if current:
                hypotheses.append(_build_hypothesis(current, len(hypotheses)))
            current = {"statement": stripped.split(":", 1)[1].strip()}
        elif stripped.startswith("[H") or re.match(r"^H\d+[:\.]", stripped):
            if current:
                hypotheses.append(_build_hypothesis(current, len(hypotheses)))
            current = {"statement": re.sub(r"^\[?H\d+\]?[:\.\s]*", "", stripped).strip()}
        elif "RANK:" in upper:
            m = re.search(r"(\d+)", stripped)
            if m:
                current["rank"] = int(m.group(1))
        elif "EXPLANATORY_POWER:" in upper:
            m = re.search(r"([\d.]+)", stripped)
            if m:
                current["explanatory_power"] = float(m.group(1))
        elif "SIMPLICITY:" in upper:
            m = re.search(r"([\d.]+)", stripped)
            if m:
                current["simplicity"] = float(m.group(1))
        elif "CONSISTENCY:" in upper:
            m = re.search(r"([\d.]+)", stripped)
            if m:
                current["consistency"] = float(m.group(1))
        elif "MECHANISTIC_PLAUSIBILITY:" in upper:
            m = re.search(r"([\d.]+)", stripped)
            if m:
                current["mechanistic_plausibility"] = float(m.group(1))
        elif "RATIONALE:" in upper:
            current["rationale"] = stripped.split(":", 1)[1].strip()
        elif "PREDICTION" in upper and ":" in stripped:
            preds = current.get("predictions", [])
            val = stripped.split(":", 1)[1].strip()
            if val:
                preds.append(val)
            current["predictions"] = preds
        elif "FALSIF" in upper and ":" in stripped:
            falsif = current.get("falsifiers", [])
            val = stripped.split(":", 1)[1].strip()
            if val:
                falsif.append(val)
            current["falsifiers"] = falsif
        elif "MINIMAL_TEST:" in upper:
            current["minimal_test"] = stripped.split(":", 1)[1].strip()
        elif upper.startswith("CONFIDENCE:") and "JUSTIFICATION" not in upper:
            m = re.search(r"(\d+)", stripped)
            if m:
                current["confidence"] = int(m.group(1))
        elif "CONFIDENCE_JUSTIFICATION:" in upper:
            current["confidence_justification"] = stripped.split(":", 1)[1].strip()
        elif "ASSUMPTION" in upper and ":" in stripped:
            assumptions = stripped.split(":", 1)[1].strip()
            current["assumptions"] = [
                a.strip() for a in re.split(r"[;,]|A\d+\.?\s*", assumptions) if a.strip()
            ]
        elif "KNOWN_UNKNOWN" in upper and ":" in stripped:
            unknowns = stripped.split(":", 1)[1].strip()
            current["known_unknowns"] = [u.strip() for u in unknowns.split(";") if u.strip()]
        elif "FAILURE_MODE" in upper and ":" in stripped:
            modes = stripped.split(":", 1)[1].strip()
            current["failure_modes"] = [m.strip() for m in modes.split(";") if m.strip()]
        elif upper.startswith("REFS:") or upper.startswith("- REFS:"):
            current["refs"] = re.findall(r"\[(\d+)\]", stripped)
        elif current and "statement" in current:
            # Continuation lines for multi-line predictions/falsifiers
            if stripped.startswith("- "):
                val = stripped.lstrip("- ")
                # Try to assign to the most recent list field
                for field in ["predictions", "falsifiers", "assumptions",
                              "known_unknowns", "failure_modes"]:
                    if field in current and isinstance(current[field], list):
                        current[field].append(val)
                        break

    # Flush final hypothesis
    if current:
        hypotheses.append(_build_hypothesis(current, len(hypotheses)))

    return hypotheses


def _build_hypothesis(data: dict, idx: int) -> RankedHypothesis:
    """Build a RankedHypothesis from parsed fields."""
    ep = min(data.get("explanatory_power", 0.5), 1.0)
    simp = min(data.get("simplicity", 0.5), 1.0)
    cons = min(data.get("consistency", 0.5), 1.0)
    mech = min(data.get("mechanistic_plausibility", 0.5), 1.0)
    overall = (ep * 0.3 + simp * 0.2 + cons * 0.3 + mech * 0.2)

    return RankedHypothesis(
        hypothesis_id=f"H{idx + 1}",
        statement=data.get("statement", ""),
        rank=data.get("rank", idx + 1),
        explanatory_power=ep,
        simplicity=simp,
        consistency=cons,
        mechanistic_plausibility=mech,
        overall_score=round(overall, 3),
        rationale=data.get("rationale", ""),
        predictions=data.get("predictions", []),
        falsifiers=data.get("falsifiers", []),
        minimal_test=data.get("minimal_test", ""),
        confidence=min(data.get("confidence", 0), 100),
        confidence_justification=data.get("confidence_justification", ""),
        assumptions=data.get("assumptions", []),
        known_unknowns=data.get("known_unknowns", []),
        failure_modes=data.get("failure_modes", []),
        supporting_refs=[f"[{r}]" for r in data.get("refs", [])],
    )


def parse_next_queries(raw_output: str) -> list[str]:
    """Parse NEXT_QUERIES section from LLM output."""
    queries: list[str] = []
    in_section = False

    for line in raw_output.split("\n"):
        stripped = line.strip()
        upper = stripped.upper()

        if "NEXT_QUER" in upper or "NEXT BEST EVIDENCE" in upper:
            in_section = True
            continue
        if in_section and any(
            header in upper
            for header in ["CONFIDENCE", "UNCERTAINTY", "HYPOTHES", "EVIDENCE"]
        ):
            break

        if in_section and stripped.startswith("- "):
            query = stripped.lstrip("- ").strip('"').strip("'")
            if query and len(query) > 5:
                queries.append(query)

    return queries


def parse_overall_confidence(raw_output: str) -> tuple[int, str]:
    """Parse OVERALL_CONFIDENCE and OVERALL_JUSTIFICATION."""
    confidence = 0
    justification = ""

    for line in raw_output.split("\n"):
        stripped = line.strip()
        upper = stripped.upper()

        if "OVERALL_CONFIDENCE:" in upper or (
            "CONFIDENCE:" in upper and "OVERALL" in upper
        ):
            m = re.search(r"(\d+)", stripped)
            if m:
                confidence = min(int(m.group(1)), 100)
        elif "OVERALL_JUSTIFICATION:" in upper or "JUSTIFICATION:" in upper:
            justification = stripped.split(":", 1)[1].strip()

    return confidence, justification


def parse_uncertainty_notes(raw_output: str) -> list[str]:
    """Parse UNCERTAINTY section."""
    notes: list[str] = []
    in_section = False

    for line in raw_output.split("\n"):
        stripped = line.strip()
        upper = stripped.upper()

        if "UNCERTAINTY" in upper and (
            upper.startswith("UNCERTAINTY") or upper.startswith("#")
            or upper.startswith("*") or "8." in upper
        ):
            in_section = True
            continue
        if in_section and any(
            header in upper
            for header in [
                "CONFIDENCE:", "NEXT_QUER", "OVERALL_CONFIDENCE",
                "HYPOTHES", "SYSTEM DESIGN",
            ]
        ):
            break

        if in_section and stripped.startswith("- "):
            notes.append(stripped.lstrip("- "))

    return notes


def build_engine_result(
    raw_output: str,
    query: str,
    mode: NexusMode,
    sources: list[QueryResult],
    total_searched: int,
    model_used: str,
    live_sources_fetched: int = 0,
) -> HypothesisEngineResult:
    """Parse the full LLM output into a HypothesisEngineResult."""
    evidence_graph = parse_evidence_graph(raw_output)
    evidence_graph.query = query

    hypotheses = []
    if mode in (NexusMode.HYPOTHESIS, NexusMode.SYNTHESIS):
        hypotheses = parse_hypotheses(raw_output)
        # Sort by overall score descending
        hypotheses.sort(key=lambda h: h.overall_score, reverse=True)
        # Re-assign ranks after sorting
        for i, h in enumerate(hypotheses):
            h.rank = i + 1

    next_queries = parse_next_queries(raw_output)
    confidence, justification = parse_overall_confidence(raw_output)
    uncertainty = parse_uncertainty_notes(raw_output)

    return HypothesisEngineResult(
        query=query,
        mode=mode,
        evidence_graph=evidence_graph,
        hypotheses=hypotheses,
        next_queries=next_queries,
        confidence=confidence,
        confidence_justification=justification,
        uncertainty_notes=uncertainty,
        sources=sources,
        model_used=model_used,
        total_chunks_searched=total_searched,
        live_sources_fetched=live_sources_fetched,
    )

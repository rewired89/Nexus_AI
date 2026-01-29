#!/usr/bin/env python3
"""
Acheron Nexus — Manual PubMed API Bridge
=========================================
Fallback mode when the MCP compute layer is unreachable.
Uses NCBI E-Utilities directly via ``requests`` to fetch recent
papers and applies the Acheron Layered Research Architecture
(Evidence → Inference → Theory) to the results.

Usage:
    python scripts/pubmed_api_bridge.py [--query QUERY] [--retmax N]
    python scripts/pubmed_api_bridge.py --seed-cache data/seed_cache.json
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

from pathlib import Path

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]
from xml.etree import ElementTree as ET

# ── NCBI E-Utilities endpoints ──────────────────────────────────────────────
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# ── Default domain topic ────────────────────────────────────────────────────
DEFAULT_QUERY = "planarian bioelectricity membrane voltage regeneration"


# ── Data models ─────────────────────────────────────────────────────────────
@dataclass
class PaperRecord:
    """Minimal record for a PubMed paper."""
    pmid: str
    title: str
    authors: list[str]
    source: str  # journal
    pub_date: str
    abstract: str = ""
    doi: str = ""


@dataclass
class SourceProvenance:
    """Provenance metadata for auditable research."""
    provider: str = "NCBI"
    database: str = "pubmed"
    fetched_at_utc: str = ""
    request_query: str = ""
    api_key_used: bool = False
    pmids_returned: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.fetched_at_utc:
            self.fetched_at_utc = datetime.now(timezone.utc).isoformat()


@dataclass
class LayeredAnalysis:
    """Acheron three-layer epistemic structure."""
    evidence: list[str] = field(default_factory=list)
    inference: list[str] = field(default_factory=list)
    theory: list[str] = field(default_factory=list)
    variables_detected: list[dict] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)


# ── PubMed API functions ───────────────────────────────────────────────────
def esearch(
    query: str,
    retmax: int = 5,
    api_key: str = "",
    email: str = "",
    sort: str = "date",
) -> list[str]:
    """Step 1: Search PubMed and return a list of PMIDs."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "sort": sort,
        "retmode": "json",
        "usehistory": "n",
    }
    if api_key:
        params["api_key"] = api_key
    if email:
        params["email"] = email
        params["tool"] = "AcheronNexus"

    resp = requests.get(ESEARCH_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    result = data.get("esearchresult", {})
    pmids = result.get("idlist", [])
    count = result.get("count", "0")

    print(f"[esearch] Query: {query!r}")
    print(f"[esearch] Total hits: {count}")
    print(f"[esearch] Returned PMIDs ({len(pmids)}): {', '.join(pmids)}")
    return pmids


def esummary(
    pmids: list[str],
    api_key: str = "",
    email: str = "",
) -> list[PaperRecord]:
    """Step 2: Fetch summary metadata for a list of PMIDs."""
    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "json",
    }
    if api_key:
        params["api_key"] = api_key
    if email:
        params["email"] = email
        params["tool"] = "AcheronNexus"

    resp = requests.get(ESUMMARY_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    result = data.get("result", {})
    uid_list = result.get("uids", pmids)

    records: list[PaperRecord] = []
    for uid in uid_list:
        doc = result.get(uid, {})
        if not doc or "error" in doc:
            continue

        authors = [a.get("name", "") for a in doc.get("authors", [])]
        elocation = doc.get("elocationid", "")
        doi = ""
        if elocation and "doi:" in elocation.lower():
            doi = elocation.replace("doi: ", "").replace("doi:", "").strip()

        records.append(PaperRecord(
            pmid=uid,
            title=doc.get("title", ""),
            authors=authors,
            source=doc.get("source", ""),
            pub_date=doc.get("pubdate", ""),
            doi=doi,
        ))

    print(f"[esummary] Retrieved metadata for {len(records)} papers")
    return records


def efetch_abstracts(
    pmids: list[str],
    api_key: str = "",
    email: str = "",
) -> dict[str, str]:
    """Step 3: Fetch full abstracts via efetch XML."""
    if not pmids:
        return {}

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    if api_key:
        params["api_key"] = api_key
    if email:
        params["email"] = email
        params["tool"] = "AcheronNexus"

    resp = requests.get(EFETCH_URL, params=params, timeout=30)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    abstracts: dict[str, str] = {}
    for article in root.iter("PubmedArticle"):
        pmid_el = article.find(".//PMID")
        if pmid_el is None:
            continue
        pmid = pmid_el.text or ""

        abstract_parts: list[str] = []
        for abs_text in article.iter("AbstractText"):
            label = abs_text.get("Label", "")
            text = "".join(abs_text.itertext()).strip()
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)

        abstracts[pmid] = "\n".join(abstract_parts)

    print(f"[efetch] Retrieved abstracts for {len(abstracts)} papers")
    return abstracts


# ── Acheron Layered Research Architecture ──────────────────────────────────
BIOELECTRIC_KEYWORDS = {
    "vmem": "Vmem (membrane voltage)",
    "membrane potential": "Vmem (membrane voltage)",
    "resting potential": "Vmem (membrane voltage)",
    "depolariz": "Vmem perturbation (depolarization)",
    "hyperpolariz": "Vmem perturbation (hyperpolarization)",
    "electric field": "EF (endogenous electric field)",
    "gap junction": "Gj (gap junctional conductance)",
    "connexin": "Gj/Connexin expression",
    "innexin": "Gj/Innexin expression (invertebrate)",
    "ion channel": "Ion channel",
    "potassium": "K+ channel/current",
    "sodium": "Na+ channel/current",
    "calcium": "Ca2+ signaling",
    "chloride": "Cl- channel/current",
    "regenerat": "Morphological outcome (regeneration)",
    "morphogen": "Morphogenesis/patterning",
    "planari": "Model organism: Planaria",
    "xenopus": "Model organism: Xenopus",
    "bioelectric": "Bioelectric signaling (general)",
}


def detect_variables(text: str) -> list[dict]:
    """Scan text for bioelectric variable mentions."""
    text_lower = text.lower()
    found: list[dict] = []
    seen = set()
    for keyword, variable_label in BIOELECTRIC_KEYWORDS.items():
        if keyword in text_lower and variable_label not in seen:
            seen.add(variable_label)
            found.append({"keyword": keyword, "variable": variable_label})
    return found


def apply_layered_architecture(papers: list[PaperRecord]) -> LayeredAnalysis:
    """
    Apply the Acheron three-layer epistemic architecture:
        Layer 1 — EVIDENCE: Direct facts from source abstracts.
        Layer 2 — INFERENCE: Cross-paper synthesis and pattern detection.
        Layer 3 — THEORY: Hypotheses and speculative connections.
    """
    analysis = LayeredAnalysis()

    # ── Layer 1: EVIDENCE ──────────────────────────────────────────────
    all_variables: list[dict] = []
    organisms_mentioned: set[str] = set()
    mechanisms_mentioned: set[str] = set()

    for i, p in enumerate(papers, 1):
        ref = f"[{i}]"
        text = p.abstract or p.title

        # Extract evidence statements
        if p.abstract:
            # Take the first two sentences as primary evidence
            sentences = [s.strip() for s in p.abstract.replace("\n", " ").split(". ") if s.strip()]
            if sentences:
                ev = f"{ref} {sentences[0]}."
                analysis.evidence.append(ev)
                if len(sentences) > 1:
                    analysis.evidence.append(f"{ref} {sentences[1]}.")
        else:
            analysis.evidence.append(f"{ref} Title only: {p.title}")

        # Detect bioelectric variables
        detected = detect_variables(text)
        for v in detected:
            v["source_ref"] = ref
            all_variables.append(v)

        # Track organisms and mechanisms
        text_lower = text.lower()
        for org in ["planari", "xenopus", "zebrafish", "mammal", "mouse", "human", "drosophila"]:
            if org in text_lower:
                organisms_mentioned.add(org.rstrip("i") + ("ia" if org == "planari" else ""))
        for mech in ["regenerat", "morphogen", "memory", "pattern", "wound heal", "tumor", "cancer"]:
            if mech in text_lower:
                mechanisms_mentioned.add(mech.rstrip("at").rstrip("en") + ("ation" if mech.endswith("at") else "esis" if mech.endswith("en") else ""))

    analysis.variables_detected = all_variables

    # ── Layer 2: INFERENCE ─────────────────────────────────────────────
    # Cross-reference detected variables
    unique_vars = {v["variable"] for v in all_variables}
    if len(unique_vars) > 1:
        var_list = ", ".join(sorted(unique_vars))
        analysis.inference.append(
            f"Cross-paper variable convergence detected: {var_list}. "
            f"Multiple papers reference overlapping bioelectric parameters, "
            f"suggesting a shared mechanistic pathway."
        )

    # Organism overlap analysis
    if len(organisms_mentioned) > 1:
        org_list = ", ".join(sorted(organisms_mentioned))
        analysis.inference.append(
            f"Cross-species evidence spans: {org_list}. "
            f"Conservation of bioelectric mechanisms across these organisms "
            f"strengthens the causal role of Vmem/Gj signaling in patterning."
        )

    # Mechanism convergence
    if len(mechanisms_mentioned) > 1:
        mech_list = ", ".join(sorted(mechanisms_mentioned))
        analysis.inference.append(
            f"Mechanism convergence: {mech_list}. "
            f"These downstream outcomes may share upstream bioelectric regulation."
        )

    # Source density signal
    papers_with_abstracts = sum(1 for p in papers if p.abstract)
    analysis.inference.append(
        f"Source quality: {papers_with_abstracts}/{len(papers)} papers returned "
        f"full abstracts. Evidence density is "
        f"{'sufficient' if papers_with_abstracts >= 3 else 'sparse'} "
        f"for preliminary synthesis."
    )

    # ── Layer 3: THEORY (Speculative Hypotheses) ──────────────────────
    if "Vmem (membrane voltage)" in unique_vars and any("regenerat" in v["variable"].lower() for v in all_variables):
        analysis.theory.append(
            "HYPOTHESIS [confidence: medium]: Vmem state acts as a "
            "morphogenetic pre-pattern that instructs regeneration targets. "
            "Predicted impact: Vmem manipulation could redirect regeneration "
            "outcomes independent of genetic identity. "
            "Validation: Voltage-reporter imaging in planarian blastema "
            "during first 24h post-amputation."
        )

    if "Gj" in " ".join(unique_vars):
        analysis.theory.append(
            "HYPOTHESIS [confidence: medium-low]: Gap junctional networks "
            "propagate bioelectric morphogen gradients analogous to chemical "
            "morphogen diffusion. Gj topology may encode positional information "
            "distinct from individual cell Vmem. "
            "Validation: Spatially-resolved Gj conductance mapping via "
            "connexin/innexin knockout mosaics."
        )

    if len(organisms_mentioned) >= 2:
        analysis.theory.append(
            "HYPOTHESIS [confidence: low-medium]: Bioelectric patterning "
            "codes are deeply conserved and may represent an ancient "
            "information layer predating morphogen signaling cascades. "
            "Cross-species experiments comparing Vmem maps at homologous "
            "developmental stages could test this."
        )

    # Always generate at least one gap analysis
    if papers_with_abstracts < len(papers):
        analysis.gaps.append(
            f"{len(papers) - papers_with_abstracts} paper(s) lack abstracts; "
            f"full-text retrieval via PMC NXML is recommended."
        )
    analysis.gaps.append(
        "This analysis is based on abstracts only. Full-text evidence spans "
        "with byte-offset provenance require the MCP compute layer or "
        "PMC NXML pipeline (nexus_ingest.pmc_pubmed)."
    )
    analysis.gaps.append(
        "Quantitative bioelectric variable extraction (exact Vmem values, "
        "EF magnitudes) requires structured parsing not available in "
        "bridge mode. Recommend upgrading to full pipeline."
    )

    return analysis


# ── Display formatting ─────────────────────────────────────────────────────
def print_divider(label: str = "", char: str = "─", width: int = 72) -> None:
    if label:
        pad = (width - len(label) - 2) // 2
        print(f"\n{char * pad} {label} {char * pad}")
    else:
        print(char * width)


def display_results(
    papers: list[PaperRecord],
    analysis: LayeredAnalysis,
    provenance: SourceProvenance,
    query: str,
) -> None:
    """Render the full Acheron-formatted research output."""
    print_divider("ACHERON NEXUS — MANUAL API BRIDGE")
    print(f"  Mode      : PubMed E-Utilities (manual bridge)")
    print(f"  Query     : {query}")
    print(f"  Papers    : {len(papers)}")
    print(f"  Timestamp : {provenance.fetched_at_utc}")
    print(f"  API Key   : {'provided' if provenance.api_key_used else 'none'}")

    # ── Papers ──
    print_divider("RETRIEVED PAPERS")
    for i, p in enumerate(papers, 1):
        authors_short = ", ".join(p.authors[:3])
        if len(p.authors) > 3:
            authors_short += " et al."
        print(f"\n  [{i}] PMID {p.pmid}")
        print(f"      Title   : {p.title}")
        print(f"      Authors : {authors_short}")
        print(f"      Journal : {p.source}")
        print(f"      Date    : {p.pub_date}")
        if p.doi:
            print(f"      DOI     : {p.doi}")
        if p.abstract:
            wrapped = textwrap.fill(p.abstract[:500], width=68, initial_indent="      ", subsequent_indent="      ")
            print(f"      Abstract: {wrapped.strip()}")
            if len(p.abstract) > 500:
                print(f"      ... [{len(p.abstract)} chars total]")
        else:
            print("      Abstract: [not available]")

    # ── Layer 1: EVIDENCE ──
    print_divider("LAYER 1 — EVIDENCE (immutable source facts)")
    for ev in analysis.evidence:
        print(f"  • {ev}")

    # ── Layer 2: INFERENCE ──
    print_divider("LAYER 2 — INFERENCE (cross-paper synthesis)")
    for inf in analysis.inference:
        wrapped = textwrap.fill(inf, width=68, initial_indent="  • ", subsequent_indent="    ")
        print(wrapped)

    # ── Detected Variables ──
    if analysis.variables_detected:
        print_divider("BIOELECTRIC VARIABLES DETECTED")
        seen = set()
        for v in analysis.variables_detected:
            key = v["variable"]
            if key not in seen:
                refs = [x["source_ref"] for x in analysis.variables_detected if x["variable"] == key]
                print(f"  {key:.<45s} {', '.join(refs)}")
                seen.add(key)

    # ── Layer 3: THEORY ──
    print_divider("LAYER 3 — THEORY (hypotheses & speculation)")
    if analysis.theory:
        for th in analysis.theory:
            wrapped = textwrap.fill(th, width=68, initial_indent="  ▸ ", subsequent_indent="    ")
            print(wrapped)
    else:
        print("  [No hypotheses generated — insufficient cross-domain signal]")

    # ── Gaps ──
    print_divider("GAPS & LIMITATIONS")
    for gap in analysis.gaps:
        wrapped = textwrap.fill(gap, width=68, initial_indent="  ⚠ ", subsequent_indent="    ")
        print(wrapped)

    print_divider()


def save_results(
    papers: list[PaperRecord],
    analysis: LayeredAnalysis,
    provenance: SourceProvenance,
    output_path: str,
) -> None:
    """Save structured results as JSON for ledger integration."""
    payload = {
        "provenance": asdict(provenance),
        "papers": [asdict(p) for p in papers],
        "layered_analysis": asdict(analysis),
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[saved] Results written to {output_path}")


# ── Seed cache loading ─────────────────────────────────────────────────────
def load_seed_cache(cache_path: str) -> tuple[list[PaperRecord], str]:
    """Load pre-fetched paper data from a JSON seed cache."""
    with open(cache_path) as f:
        data = json.load(f)
    query = data.get("query", DEFAULT_QUERY)
    papers: list[PaperRecord] = []
    for p in data.get("papers", []):
        papers.append(PaperRecord(
            pmid=p["pmid"],
            title=p["title"],
            authors=p.get("authors", []),
            source=p.get("source", ""),
            pub_date=p.get("pub_date", ""),
            abstract=p.get("abstract", ""),
            doi=p.get("doi", ""),
        ))
    return papers, query


# ── Main execution ─────────────────────────────────────────────────────────
def run_bridge(
    query: str = DEFAULT_QUERY,
    retmax: int = 5,
    api_key: str = "",
    email: str = "",
    output_path: Optional[str] = None,
    seed_cache: Optional[str] = None,
) -> tuple[list[PaperRecord], LayeredAnalysis, SourceProvenance]:
    """Execute the full manual API bridge pipeline.

    If *seed_cache* is provided, papers are loaded from a local JSON file
    instead of hitting the NCBI network (useful when the proxy blocks
    outbound HTTPS to eutils.ncbi.nlm.nih.gov).
    """
    provenance = SourceProvenance(
        request_query=query,
        api_key_used=bool(api_key),
    )

    papers: list[PaperRecord] = []

    if seed_cache:
        # ── Offline / cached mode ──────────────────────────────────────
        print_divider("SEED-CACHE MODE")
        papers, cached_query = load_seed_cache(seed_cache)
        if not query or query == DEFAULT_QUERY:
            query = cached_query
        provenance.request_query = query
        provenance.provider = "seed-cache"
        provenance.pmids_returned = [p.pmid for p in papers]
        print(f"[cache] Loaded {len(papers)} papers from {seed_cache}")
        print(f"[cache] Query: {query!r}")
    else:
        # ── Live API mode ──────────────────────────────────────────────
        # Step 1: esearch — get PMIDs
        print_divider("STEP 1: esearch")
        pmids = esearch(query, retmax=retmax, api_key=api_key, email=email)
        provenance.pmids_returned = pmids

        if not pmids:
            print("[!] No results returned. Try broadening the query.")
            return [], LayeredAnalysis(), provenance

        # Rate-limit pause
        time.sleep(0.35)

        # Step 2: esummary — get metadata
        print_divider("STEP 2: esummary")
        papers = esummary(pmids, api_key=api_key, email=email)

        time.sleep(0.35)

        # Step 3: efetch — get full abstracts
        print_divider("STEP 3: efetch (abstracts)")
        abstracts = efetch_abstracts(pmids, api_key=api_key, email=email)

        # Merge abstracts into paper records
        for p in papers:
            if p.pmid in abstracts:
                p.abstract = abstracts[p.pmid]

    # Step 4: Apply layered architecture
    print_divider("STEP 4: Layered Research Architecture")
    analysis = apply_layered_architecture(papers)

    # Display
    display_results(papers, analysis, provenance, query)

    # Optionally save
    if output_path:
        save_results(papers, analysis, provenance, output_path)

    return papers, analysis, provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Acheron Nexus — Manual PubMed API Bridge")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="PubMed search query")
    parser.add_argument("--retmax", type=int, default=5, help="Number of results")
    parser.add_argument("--api-key", default="", help="NCBI API key")
    parser.add_argument("--email", default="", help="Email for NCBI")
    parser.add_argument("--output", default=None, help="Save JSON output to file")
    parser.add_argument(
        "--seed-cache",
        default=None,
        help="Path to seed-cache JSON (offline fallback when API is unreachable)",
    )
    args = parser.parse_args()

    run_bridge(
        query=args.query,
        retmax=args.retmax,
        api_key=args.api_key,
        email=args.email,
        output_path=args.output,
        seed_cache=args.seed_cache,
    )

"""Stage A — Query Understanding for Science-First Mode.

Parses user queries into structured components:
  - entities: species, tissues, genes, channels, methods
  - required_measurements: Vmem, EF, Gj, gene expression, etc.
  - constraints: organism scope, cross-species allowed or not

Used by the pipeline to configure evidence filtering and prompt assembly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from acheron.config import get_settings

# ======================================================================
# Entity extraction patterns
# ======================================================================
_SPECIES_PATTERNS: dict[str, list[str]] = {
    "planarian": [
        "planari", "schmidtea", "dugesia", "girardia", "flatworm",
        "s. mediterranea", "d. japonica",
    ],
    "xenopus": [
        "xenopus", "frog", "tadpole", "x. laevis",
    ],
    "physarum": [
        "physarum", "slime mold", "slime mould", "p. polycephalum",
    ],
    "hydra": ["hydra"],
    "zebrafish": ["zebrafish", "danio", "d. rerio"],
    "mouse": ["mouse", "mice", "mus musculus"],
    "human": ["human", "patient", "homo sapiens"],
    "organoid": ["organoid"],
    "axolotl": ["axolotl", "ambystoma"],
}

_GENE_PATTERNS: list[str] = [
    r"\bwnt\b", r"\bnotum\b", r"\bpiwi[\-\s]?1\b", r"\bbeta[\-\s]?catenin\b",
    r"\bshh\b", r"\bsonic\s+hedgehog\b", r"\bbmp[\-\s]?\d?\b",
    r"\bcrispr\b", r"\bcas9\b", r"\bcas13\b",
    r"\bconnexin[\-\s]?\d*\b", r"\binnexin[\-\s]?\d*\b",
    r"\bgj[abc]?\d?\b", r"\bcx\d{2}\b",
]

_CHANNEL_PATTERNS: list[str] = [
    r"\bk\+", r"\bna\+", r"\bca2?\+", r"\bcl[\-−]",
    r"\bh\+[/,]k\+[\-\s]?atpase\b", r"\bv[\-\s]?atpase\b",
    r"\btrp\s?[a-z]?\d?\b", r"\bkatp\b", r"\bkir\b",
    r"\bnavolt\b", r"\bvgsc\b",
]

_MEASUREMENT_PATTERNS: dict[str, list[str]] = {
    "vmem": [
        "vmem", "membrane potential", "resting potential",
        "voltage", "transmembrane potential",
    ],
    "ef": [
        "electric field", "endogenous field", "\\bef\\b",
        "galvanotax", "electrotax",
    ],
    "gj": [
        "gap junction", "connexin", "innexin", "gjc",
        "intercellular communication", "dye coupling",
    ],
    "gene_expression": [
        "gene expression", "qpcr", "rt-pcr", "rna-seq",
        "transcriptom", "in situ hybrid",
    ],
    "regeneration": [
        "regenerat", "amputation", "wound heal",
        "blastema", "target morpholog",
    ],
    "ion_flux": [
        "ion flux", "ion current", "self-referencing",
        "vibrating probe",
    ],
}

_METHOD_PATTERNS: list[str] = [
    "voltage-sensitive dye", "dibac", "berst",
    "patch clamp", "microelectrode", "mea",
    "optogenetic", "channelrhodopsin",
    "ionophore", "ivermectin", "sch28080", "octanol",
    "rnai", "morpholino", "crispr",
    "fluorescence imag", "confocal", "two-photon",
    "calcium imag", "gcamp",
]

_TISSUE_PATTERNS: list[str] = [
    "neoblast", "stem cell", "epiderm", "mesenchym",
    "neural crest", "blastema", "tail", "head",
    "anterior", "posterior", "dorsal", "ventral",
    "gut", "pharynx", "brain", "cephalic gangli",
    "epithelium", "endoderm", "ectoderm",
]


@dataclass
class ParsedQuery:
    """Structured representation of a user research query."""

    raw_query: str
    species: list[str] = field(default_factory=list)
    tissues: list[str] = field(default_factory=list)
    genes: list[str] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    required_measurements: list[str] = field(default_factory=list)
    organism_constraint: str = "planarian"
    cross_species_allowed: bool = False

    def summary(self) -> str:
        """One-line summary for logging."""
        parts = []
        if self.species:
            parts.append(f"species={self.species}")
        if self.required_measurements:
            parts.append(f"measures={self.required_measurements}")
        if self.genes:
            parts.append(f"genes={self.genes}")
        parts.append(f"scope={self.organism_constraint}")
        return ", ".join(parts) if parts else "(no entities detected)"


def parse_query(query: str) -> ParsedQuery:
    """Parse a user query into structured entities and constraints.

    Stage A of the Science-First pipeline.
    """
    settings = get_settings()
    lower = query.lower()
    result = ParsedQuery(raw_query=query)

    # Extract species
    for species, patterns in _SPECIES_PATTERNS.items():
        for pat in patterns:
            if pat in lower:
                if species not in result.species:
                    result.species.append(species)
                break

    # Detect cross-species intent
    cross_signals = [
        "cross-species", "cross species", "compare",
        "across species", "conserved", "across organisms",
        "evolution", "phylogenet",
    ]
    result.cross_species_allowed = any(s in lower for s in cross_signals)

    # Organism constraint from config
    result.organism_constraint = settings.organism_strict
    if result.cross_species_allowed:
        result.organism_constraint = "any"

    # Extract genes
    for pattern in _GENE_PATTERNS:
        matches = re.findall(pattern, lower)
        for m in matches:
            cleaned = m.strip()
            if cleaned and cleaned not in result.genes:
                result.genes.append(cleaned)

    # Extract ion channels
    for pattern in _CHANNEL_PATTERNS:
        matches = re.findall(pattern, lower)
        for m in matches:
            cleaned = m.strip()
            if cleaned and cleaned not in result.channels:
                result.channels.append(cleaned)

    # Extract required measurements
    for mtype, patterns in _MEASUREMENT_PATTERNS.items():
        for pat in patterns:
            if pat in lower:
                if mtype not in result.required_measurements:
                    result.required_measurements.append(mtype)
                break

    # Extract methods
    for pat in _METHOD_PATTERNS:
        if pat in lower:
            if pat not in result.methods:
                result.methods.append(pat)

    # Extract tissues
    for pat in _TISSUE_PATTERNS:
        if pat in lower:
            if pat not in result.tissues:
                result.tissues.append(pat)

    return result


def generate_collection_queries(parsed: ParsedQuery) -> list[str]:
    """Generate PubMed/bioRxiv/arXiv search queries from a parsed query.

    Stage E supplement: if local corpus lacks coverage, produce exact
    queries the user can run to collect missing evidence.
    """
    queries: list[str] = []
    organism = parsed.species[0] if parsed.species else "planarian"

    # Base bioelectric query for the organism
    queries.append(
        f"{organism} bioelectric membrane potential Vmem regeneration"
    )

    # Measurement-specific queries
    for m in parsed.required_measurements:
        if m == "vmem":
            queries.append(
                f"{organism} membrane potential measurement mV"
            )
        elif m == "ef":
            queries.append(
                f"{organism} endogenous electric field wound"
            )
        elif m == "gj":
            queries.append(
                f"{organism} gap junction coupling connexin innexin"
            )
        elif m == "gene_expression":
            gene_str = " ".join(parsed.genes[:3]) if parsed.genes else "wnt notum"
            queries.append(
                f"{organism} {gene_str} gene expression regeneration"
            )
        elif m == "regeneration":
            queries.append(
                f"{organism} regeneration amputation morphology"
            )

    # Gene-specific queries
    for gene in parsed.genes[:3]:
        queries.append(f"{organism} {gene} bioelectric signaling")

    # Channel-specific queries
    for chan in parsed.channels[:2]:
        queries.append(f"{organism} {chan} channel membrane potential")

    # Cross-species if allowed
    if parsed.cross_species_allowed:
        queries.append(
            "bioelectric pattern memory planarian xenopus comparative"
        )

    # Deduplicate
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)

    return unique[:10]

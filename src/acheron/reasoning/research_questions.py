"""Research Question Translator & Category System (Section 13).

Maps casual/plain-English research questions to Nexus-ready scientific
queries. Provides:
    1. Research question categories for organizing investigations
    2. Casual-to-scientific query translation
    3. Plain English interpretation templates for Nexus outputs
    4. Hypothesis scaffolding for each question type

Design: This is a READ-ONLY helper. It does not modify any existing
computation or reasoning paths. It only translates and annotates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Research question categories
# ---------------------------------------------------------------------------

class ResearchCategory(str, Enum):
    """Categories of research questions in the Acheron project."""

    VOLTAGE_CONTROL = "voltage_control"
    PATTERN_REWRITING = "pattern_rewriting"
    DAMAGE_RESPONSE = "damage_response"
    REGENERATION_TIMELINE = "regeneration_timeline"
    BIO_STORAGE = "bio_storage"
    TOPOLOGY = "topology"
    ENERGY = "energy"
    GENERAL = "general"


CATEGORY_DESCRIPTIONS: dict[ResearchCategory, str] = {
    ResearchCategory.VOLTAGE_CONTROL:
        "Finding the exact voltage switches that control what body parts grow",
    ResearchCategory.PATTERN_REWRITING:
        "Hijacking the worm's electrical blueprint to force specific outcomes",
    ResearchCategory.DAMAGE_RESPONSE:
        "How cells detect damage and trigger self-repair protocols",
    ResearchCategory.REGENERATION_TIMELINE:
        "How fast things happen — from damage detection to full rebuild",
    ResearchCategory.BIO_STORAGE:
        "Using living tissue as a data storage and transfer medium",
    ResearchCategory.TOPOLOGY:
        "How the wiring pattern of cell connections affects signal flow",
    ResearchCategory.ENERGY:
        "How much energy bioelectric computation costs and its limits",
    ResearchCategory.GENERAL:
        "General research questions about bioelectric memory",
}


# ---------------------------------------------------------------------------
# Category detection from casual queries
# ---------------------------------------------------------------------------

_CATEGORY_TRIGGERS: dict[ResearchCategory, list[str]] = {
    ResearchCategory.VOLTAGE_CONTROL: [
        "voltage", "on/off switch", "on off switch", "turn on", "turn off",
        "activate", "trigger", "threshold", "set_bit", "set bit",
        "depolariz", "hyperpolariz", "what voltage", "how many volts",
        "millivolt", "mv ", "which voltage", "correct voltage",
        "wake up", "vmem", "membrane potential",
    ],
    ResearchCategory.PATTERN_REWRITING: [
        "rewrite", "re-write", "re write", "override", "hijack",
        "reprogram", "re-program", "force", "on demand", "on-demand",
        "when we decide", "when i decide", "make it grow",
        "change the pattern", "modify the pattern", "control regeneration",
        "force regeneration", "trigger regeneration",
    ],
    ResearchCategory.DAMAGE_RESPONSE: [
        "damage", "damaged", "hurt", "injury", "wound",
        "auto heal", "auto-heal", "self heal", "self-heal",
        "self repair", "self-repair", "quarantine",
        "isolate", "backup", "transfer to neighbor",
        "alarm signal", "detect damage", "auto retire",
    ],
    ResearchCategory.REGENERATION_TIMELINE: [
        "how long", "how fast", "how much time", "timeline",
        "hours", "days", "minutes", "seconds",
        "time does it take", "time does the", "regeneration time",
        "speed of", "delay", "onset", "when does",
    ],
    ResearchCategory.BIO_STORAGE: [
        "hard drive", "usb", "store information", "carry information",
        "store data", "carry data", "memory", "remember",
        "pass it on", "pass data", "transfer information",
        "data storage", "information capacity", "bits",
        "fidelity", "persist", "persistence",
        "next generation", "next one", "clone",
    ],
    ResearchCategory.TOPOLOGY: [
        "wiring", "connection pattern", "network",
        "small-world", "small world", "scale-free", "scale free",
        "how cells connect", "gap junction network",
    ],
    ResearchCategory.ENERGY: [
        "energy cost", "how much energy", "power",
        "metabolic", "atp", "picojoule",
    ],
}


def detect_research_category(query: str) -> list[ResearchCategory]:
    """Detect research categories from a casual query.

    Returns categories sorted by match count (best match first).
    """
    lower = query.lower()
    scores: dict[ResearchCategory, int] = {}

    for category, triggers in _CATEGORY_TRIGGERS.items():
        count = 0
        for trigger in triggers:
            if trigger in lower:
                count += 1
        if count > 0:
            scores[category] = count

    if not scores:
        return [ResearchCategory.GENERAL]

    sorted_cats = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
    return sorted_cats


# ---------------------------------------------------------------------------
# Query translation
# ---------------------------------------------------------------------------

@dataclass
class TranslatedQuery:
    """A casual query translated to Nexus-ready scientific form."""

    original: str
    scientific_query: str
    plain_english_summary: str  # what we're really asking
    category: ResearchCategory
    related_categories: list[ResearchCategory] = field(default_factory=list)
    hypothesis_scaffold: str = ""  # suggested hypothesis structure
    key_variables: list[str] = field(default_factory=list)
    expected_output: str = ""  # what kind of answer to expect


# Pre-built translations for common Acheron research patterns
_TRANSLATION_TEMPLATES: dict[ResearchCategory, dict[str, str]] = {
    ResearchCategory.VOLTAGE_CONTROL: {
        "scientific_prefix":
            "What is the exact Vmem threshold (in mV) that triggers ",
        "scientific_suffix":
            " Provide the SET_BIT voltage range and the molecular mechanism "
            "(ionophore, optogenetics, or pharmacological agent) required.",
        "hypothesis_scaffold":
            "HYPOTHESIS: Vmem of [X] mV in [region] is sufficient to trigger "
            "[head/tail/organ] fate specification. "
            "KILL CONDITION: If Vmem manipulation of ±[Y] mV does not alter "
            "regeneration outcome, the bioelectric switch model is falsified.",
        "key_variables": [
            "Vmem threshold (mV)", "Target region",
            "Manipulation method", "Time to effect",
        ],
        "expected_output": "Specific voltage range in mV, manipulation protocol, "
                          "expected morphological outcome",
    },
    ResearchCategory.PATTERN_REWRITING: {
        "scientific_prefix":
            "Can exogenous bioelectric manipulation override endogenous Vmem "
            "patterning to achieve ",
        "scientific_suffix":
            " Specify the REWRITE protocol: agent, concentration, exposure "
            "duration, target Vmem, and verification via READ_BIT.",
        "hypothesis_scaffold":
            "HYPOTHESIS: Applying [agent] at [concentration] for [duration] "
            "to [region] will force [outcome] regardless of endogenous state. "
            "KILL CONDITION: If the tissue reverts to default pattern within "
            "[T] hours despite continuous manipulation, the REWRITE operation "
            "is not stable.",
        "key_variables": [
            "REWRITE agent", "Concentration",
            "Exposure duration", "Persistence (T_hold)",
        ],
        "expected_output": "Step-by-step manipulation protocol with dosing, "
                          "timing, and success verification method",
    },
    ResearchCategory.DAMAGE_RESPONSE: {
        "scientific_prefix":
            "What is the bioelectric damage detection and response cascade for ",
        "scientific_suffix":
            " Characterize: depolarization wave speed, gap junction signal "
            "propagation latency, QUARANTINE mechanism, and information "
            "transfer to adjacent healthy tissue.",
        "hypothesis_scaffold":
            "HYPOTHESIS: Damaged tissue depolarizes within [T1] seconds, "
            "triggering a gap-junction-mediated alarm signal that propagates "
            "at [V] mm/s and initiates QUARANTINE of the damaged region "
            "within [T2] minutes. "
            "KILL CONDITION: If signal propagation latency exceeds [T_max], "
            "the alarm system is too slow for functional self-repair.",
        "key_variables": [
            "Depolarization wave speed (mm/s)", "Signal propagation latency (ms)",
            "QUARANTINE onset time", "Information transfer fidelity",
        ],
        "expected_output": "Timeline of damage response with measured or "
                          "estimated values for each phase",
    },
    ResearchCategory.REGENERATION_TIMELINE: {
        "scientific_prefix":
            "What is the complete timeline for ",
        "scientific_suffix":
            " Break down by phase: wound healing, blastema formation, "
            "bioelectric pattern re-establishment, differentiation, and "
            "full morphological restoration. Provide time estimates per phase.",
        "hypothesis_scaffold":
            "HYPOTHESIS: Full regeneration completes in [T_total] days, "
            "with pattern re-establishment occurring by day [T_pattern]. "
            "KILL CONDITION: If bioelectric pattern is not measurably "
            "re-established by day [T_max], the pattern-first model is wrong.",
        "key_variables": [
            "Wound healing time", "Blastema formation time",
            "Pattern establishment time", "Full restoration time",
        ],
        "expected_output": "Phase-by-phase timeline with durations in "
                          "hours/days for each regeneration stage",
    },
    ResearchCategory.BIO_STORAGE: {
        "scientific_prefix":
            "Can bioelectric Vmem patterns serve as persistent, transferable "
            "data storage for ",
        "scientific_suffix":
            " Evaluate: (a) T_hold persistence of encoded patterns, "
            "(b) fidelity of pattern transfer through fission/regeneration, "
            "(c) maximum information capacity in bits per organism using "
            "Shannon entropy of measurable Vmem states.",
        "hypothesis_scaffold":
            "HYPOTHESIS: A planarian can store [N] bits of information as "
            "Vmem patterns persisting for [T] hours, with transfer fidelity "
            "of [F]% through regeneration. "
            "KILL CONDITION: If T_hold < 1 hour or transfer fidelity < 50%, "
            "bioelectric memory is not viable as a storage medium.",
        "key_variables": [
            "Storage capacity (bits)", "Persistence (T_hold hours)",
            "Transfer fidelity (%)", "Read/write energy (pJ)",
        ],
        "expected_output": "Capacity estimate in bits, persistence time, "
                          "transfer fidelity, and energy budget per read/write",
    },
}


def translate_query(
    casual_query: str,
    target_context: str = "planarian regeneration",
) -> TranslatedQuery:
    """Translate a casual research question into a Nexus-ready query.

    Takes plain English and returns a structured scientific query with
    hypothesis scaffold and expected output description.
    """
    categories = detect_research_category(casual_query)
    primary_cat = categories[0]
    related = categories[1:] if len(categories) > 1 else []

    template = _TRANSLATION_TEMPLATES.get(primary_cat)

    if template:
        scientific_query = (
            template["scientific_prefix"]
            + target_context + "?"
            + template["scientific_suffix"]
        )
        hypothesis = template["hypothesis_scaffold"]
        key_vars = template["key_variables"]
        expected = template["expected_output"]
    else:
        # General fallback
        scientific_query = (
            f"Regarding {target_context}: {casual_query} "
            "Provide quantitative analysis with all values tagged "
            "[MEASURED], [BOUNDED-INFERENCE], or UNKNOWN."
        )
        hypothesis = (
            "HYPOTHESIS: [Formulate specific, testable prediction]. "
            "KILL CONDITION: [Define what would prove this wrong]."
        )
        key_vars = ["Identify key measurable variables"]
        expected = "Quantitative answer with evidence tags and uncertainty bounds"

    plain_summary = CATEGORY_DESCRIPTIONS.get(
        primary_cat,
        "Investigating bioelectric mechanisms in living tissue",
    )

    return TranslatedQuery(
        original=casual_query,
        scientific_query=scientific_query,
        plain_english_summary=plain_summary,
        category=primary_cat,
        related_categories=related,
        hypothesis_scaffold=hypothesis,
        key_variables=key_vars,
        expected_output=expected,
    )


def format_translated_query(tq: TranslatedQuery) -> str:
    """Format a translated query for display."""
    lines = [
        "=" * 60,
        "RESEARCH QUESTION TRANSLATION",
        "=" * 60,
        "",
        f"YOUR QUESTION: {tq.original}",
        f"CATEGORY: {tq.category.value} — {tq.plain_english_summary}",
        "",
        f"NEXUS QUERY: {tq.scientific_query}",
        "",
        "HYPOTHESIS TEMPLATE:",
        f"  {tq.hypothesis_scaffold}",
        "",
        "KEY VARIABLES TO LOOK FOR:",
    ]
    for v in tq.key_variables:
        lines.append(f"  - {v}")
    lines.append("")
    lines.append(f"EXPECTED OUTPUT: {tq.expected_output}")

    if tq.related_categories:
        lines.append("")
        lines.append("RELATED RESEARCH AREAS:")
        for rc in tq.related_categories:
            desc = CATEGORY_DESCRIPTIONS.get(rc, "")
            lines.append(f"  - {rc.value}: {desc}")

    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plain English interpretation prompt injection
# ---------------------------------------------------------------------------

PLAIN_ENGLISH_SECTION = """

PLAIN ENGLISH INTERPRETATION (MANDATORY):
After your full scientific answer, you MUST include this section:

═══════════════════════════════════════════════════════════════
WHAT THIS MEANS (Plain English)
═══════════════════════════════════════════════════════════════

Write a 3-5 sentence summary of your answer that:
- Uses ZERO scientific jargon (no mV, no Vmem, no eigenvalues)
- Explains the answer like you're talking to a smart friend over coffee
- Focuses on WHAT we can do, HOW LONG it takes, and WHETHER it works
- Ends with one clear next step

Example: "We found that the worm's 'grow a head' signal is basically
a voltage switch — when cells are more negative than -20mV, they grow
a head; more positive, they grow a tail. We can flip this switch using
a chemical bath for about 2 hours. The worm remembers the new pattern
for at least 3 hours after we remove the chemical. Next step: test
whether the memory lasts through a full regeneration cycle."

THE NUMBERS THAT MATTER:
List 3-5 key numbers from your answer translated to intuitive units:
- "[Scientific value] → [What it means in plain English]"

Example:
- "Vmem = -20 mV → The 'grow a head' switch voltage"
- "T_hold = 3 hours → How long the worm remembers after treatment"
- "Signal speed = 0.5 ms → Faster than a human blink"

CONFIDENCE LEVEL:
Rate your answer: 🟢 SOLID (measured data) / 🟡 PROBABLE (good estimates) / 🔴 GUESS (needs experiments)
═══════════════════════════════════════════════════════════════
"""


def get_plain_english_injection(category: ResearchCategory) -> str:
    """Get category-specific plain English interpretation guidance.

    Injected into the tutor mode prompt to force plain-language output.
    """
    category_hints: dict[ResearchCategory, str] = {
        ResearchCategory.VOLTAGE_CONTROL:
            "Focus on: what voltage, what chemical/tool, how long to apply, "
            "what grows as a result. Think of it like a light switch with specific settings.",
        ResearchCategory.PATTERN_REWRITING:
            "Focus on: can we override nature's plan, what's the recipe "
            "(chemical + time + dose), does it stick or revert. "
            "Think of it like overwriting a file on a computer.",
        ResearchCategory.DAMAGE_RESPONSE:
            "Focus on: how fast the alarm rings, how the backup works, "
            "whether the repair is automatic. Think of it like a building's "
            "fire alarm and sprinkler system.",
        ResearchCategory.REGENERATION_TIMELINE:
            "Focus on: give specific times for each phase. Think of it like "
            "tracking a delivery — where is my package and when does it arrive.",
        ResearchCategory.BIO_STORAGE:
            "Focus on: how much data, how long it lasts, does the copy survive. "
            "Think of it like asking how big the USB drive is and if the data "
            "survives being copied.",
    }

    hint = category_hints.get(category, "")
    return PLAIN_ENGLISH_SECTION + (f"\nCATEGORY HINT: {hint}\n" if hint else "")


# ---------------------------------------------------------------------------
# Casual language detector
# ---------------------------------------------------------------------------

_CASUAL_MARKERS = [
    "can we", "can i", "how do we", "how do i",
    "is it possible", "what if we", "what happens if",
    "how long does", "how fast does", "how much time",
    "like a", "kind of like", "sort of like",
    "in simple", "plain english", "eli5", "explain like",
    "i don't understand", "i dont understand",
    "what does that mean", "in layman",
    "basically", "essentially", "so basically",
    "worm", "worms", "planarian",
    "hard drive", "usb", "computer",
    "on/off", "switch", "turn on", "turn off",
]


def is_casual_query(query: str) -> bool:
    """Detect if a query is written in casual/non-technical language.

    Used to auto-route to tutor mode with plain English interpretation.
    """
    lower = query.lower()
    match_count = sum(1 for marker in _CASUAL_MARKERS if marker in lower)
    # If 2+ casual markers, it's a casual query
    return match_count >= 2

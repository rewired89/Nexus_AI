"""Minimal Viable Experiment (MVE) designer for Science-First Mode.

Generates structured experiment proposals with:
  - Materials list
  - Step-by-step protocol (1-6 steps, 1-2 week timeline)
  - Expected outcomes
  - Failure modes

Designed for planarian-first bioelectric research (Project Acheron).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExperimentStep:
    """A single step in a minimal experiment protocol."""

    step_number: int
    description: str
    duration: str = ""
    materials: list[str] = field(default_factory=list)


@dataclass
class ExperimentProposal:
    """A minimal viable wet-lab experiment proposal."""

    title: str
    rationale: str
    organism: str = "Schmidtea mediterranea"
    steps: list[ExperimentStep] = field(default_factory=list)
    materials: list[str] = field(default_factory=list)
    expected_outcomes: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)
    timeline: str = "1-2 weeks"
    controls: list[str] = field(default_factory=list)
    readouts: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        """Format as readable text block for LLM output integration."""
        lines = [
            f"MINIMAL WET-LAB TEST: {self.title}",
            f"Organism: {self.organism}",
            f"Timeline: {self.timeline}",
            f"Rationale: {self.rationale}",
            "",
            "MATERIALS:",
        ]
        for m in self.materials:
            lines.append(f"  - {m}")

        lines.append("")
        lines.append("PROTOCOL:")
        for step in self.steps:
            dur = f" ({step.duration})" if step.duration else ""
            lines.append(f"  Step {step.step_number}: {step.description}{dur}")

        lines.append("")
        lines.append("CONTROLS:")
        for c in self.controls:
            lines.append(f"  - {c}")

        lines.append("")
        lines.append("READOUTS:")
        for r in self.readouts:
            lines.append(f"  - {r}")

        lines.append("")
        lines.append("EXPECTED OUTCOMES:")
        for o in self.expected_outcomes:
            lines.append(f"  - {o}")

        lines.append("")
        lines.append("FAILURE MODES:")
        for f_mode in self.failure_modes:
            lines.append(f"  - {f_mode}")

        return "\n".join(lines)


# ======================================================================
# Template experiments for common Acheron research patterns
# ======================================================================

def vmem_imaging_post_amputation() -> ExperimentProposal:
    """Template: Vmem imaging at key timepoints post-amputation."""
    return ExperimentProposal(
        title="Vmem Imaging Time-Course Post-Amputation",
        rationale=(
            "Establish baseline bioelectric state changes during "
            "planarian regeneration. Map Vmem dynamics at the wound "
            "site and across the body axis to identify the bioelectric "
            "signature of target morphology recall."
        ),
        organism="Schmidtea mediterranea (asexual CIW4)",
        timeline="1 week",
        materials=[
            "S. mediterranea colony (CIW4 asexual strain)",
            "DiBAC4(3) voltage-sensitive dye (1 μM working concentration)",
            "1x Montjuïch salts (planarian water)",
            "Fluorescence stereomicroscope with FITC filter",
            "Scalpel or razor blade for amputation",
            "6-well plates, transfer pipettes",
            "Image analysis software (ImageJ/FIJI)",
        ],
        steps=[
            ExperimentStep(
                step_number=1,
                description=(
                    "Starve planarians 7 days pre-experiment. "
                    "Select 20 size-matched animals."
                ),
                duration="Day -7 to 0",
            ),
            ExperimentStep(
                step_number=2,
                description=(
                    "Day 0: Amputate pre-pharyngeally (head vs tail fragments). "
                    "10 animals per condition + 5 intact controls."
                ),
                duration="Day 0, 30 min",
            ),
            ExperimentStep(
                step_number=3,
                description=(
                    "Incubate fragments in DiBAC4(3) (1 μM) for 30 min at "
                    "each timepoint: 0h, 6h, 24h, 48h, 72h, 7d post-amputation."
                ),
                duration="30 min per timepoint",
            ),
            ExperimentStep(
                step_number=4,
                description=(
                    "Image DiBAC fluorescence under FITC filter. "
                    "Capture anterior, posterior, and wound-site regions. "
                    "Use identical exposure settings across all timepoints."
                ),
                duration="15 min per timepoint",
            ),
            ExperimentStep(
                step_number=5,
                description=(
                    "Quantify mean fluorescence intensity per region "
                    "(anterior, mid, posterior, wound site) using ImageJ."
                ),
                duration="Day 8-9",
            ),
            ExperimentStep(
                step_number=6,
                description=(
                    "Score regeneration outcomes (head present/absent, "
                    "eyes present/absent, tail blastema) at day 7 and day 14."
                ),
                duration="Day 7 + Day 14",
            ),
        ],
        controls=[
            "Intact (unamputated) animals stained with DiBAC4(3)",
            "Amputated fragments in DMSO-only (dye vehicle control)",
        ],
        readouts=[
            "Vmem gradient (DiBAC fluorescence intensity) per body region",
            "Vmem dynamics (change over 0-7 days post-amputation)",
            "Regeneration morphology score (head/eyes/tail at day 14)",
        ],
        expected_outcomes=[
            "Wound site shows rapid depolarization (bright DiBAC signal) within 6h",
            "Anterior fragment re-establishes anterior-high hyperpolarization by 48-72h",
            "Posterior fragment remains more depolarized until blastema differentiates",
            "Intact controls show stable anterior-posterior Vmem gradient",
        ],
        failure_modes=[
            "DiBAC signal too weak: increase concentration to 5 μM or extend incubation",
            "High background: planarian pigment autofluorescence — use albino strain",
            "No Vmem difference detected: may need ratiometric dye (BeRST1) for "
            "single-cell resolution instead of population-level DiBAC",
            "Animals die post-amputation: check water quality, temperature (18-20°C)",
        ],
    )


def gap_junction_modulation_regeneration() -> ExperimentProposal:
    """Template: Gap junction blocker effects on regeneration outcome."""
    return ExperimentProposal(
        title="Gap Junction Modulation Effects on Regeneration",
        rationale=(
            "Test whether gap junction connectivity is required for "
            "faithful transmission of the bioelectric target morphology "
            "pattern during planarian regeneration."
        ),
        organism="Schmidtea mediterranea (asexual CIW4)",
        timeline="2 weeks",
        materials=[
            "S. mediterranea colony (CIW4 asexual strain)",
            "Octanol (0.5 mM — gap junction blocker)",
            "1x Montjuïch salts",
            "DiBAC4(3) voltage-sensitive dye (1 μM)",
            "Fluorescence stereomicroscope",
            "6-well plates",
        ],
        steps=[
            ExperimentStep(
                step_number=1,
                description="Starve planarians 7 days. Select 30 animals.",
                duration="Day -7 to 0",
            ),
            ExperimentStep(
                step_number=2,
                description=(
                    "Amputate pre-pharyngeally. Divide into 3 groups of 10: "
                    "(A) octanol 0.5 mM continuous, "
                    "(B) octanol 0.5 mM first 24h only, "
                    "(C) vehicle control."
                ),
                duration="Day 0",
            ),
            ExperimentStep(
                step_number=3,
                description=(
                    "Image DiBAC fluorescence at 0h, 24h, 72h, 7d, 14d "
                    "for all groups."
                ),
                duration="30 min per timepoint",
            ),
            ExperimentStep(
                step_number=4,
                description=(
                    "Score regeneration morphology at day 14: "
                    "normal head, cyclopic, headless, two-headed."
                ),
                duration="Day 14",
            ),
            ExperimentStep(
                step_number=5,
                description=(
                    "Optional: qPCR for polarity markers (notum, wnt1, "
                    "β-catenin) at 48h if RNA extraction available."
                ),
                duration="Day 2-3",
            ),
            ExperimentStep(
                step_number=6,
                description="Quantify and compare Vmem patterns and outcomes.",
                duration="Day 15-16",
            ),
        ],
        controls=[
            "Vehicle control (DMSO at equivalent volume)",
            "Intact unamputated controls",
        ],
        readouts=[
            "Regeneration outcome morphology (% normal vs abnormal heads)",
            "Vmem gradient disruption (DiBAC imaging)",
            "Optional: polarity gene expression (qPCR)",
        ],
        expected_outcomes=[
            "Continuous octanol: disrupted Vmem gradient, high rate of "
            "cyclopic or headless regeneration",
            "24h-only octanol: partially disrupted, intermediate outcomes",
            "Vehicle: normal regeneration, intact Vmem gradient restored by 72h",
        ],
        failure_modes=[
            "Octanol toxicity at 0.5 mM: reduce to 0.25 mM",
            "No phenotype: innexin redundancy — try lindane as alternative "
            "or combine with innexin-7 RNAi",
            "High mortality: reduce exposure window or concentration",
        ],
    )


# ======================================================================
# Experiment selection based on query analysis
# ======================================================================
_EXPERIMENT_TEMPLATES = {
    "vmem": vmem_imaging_post_amputation,
    "regeneration": vmem_imaging_post_amputation,
    "gj": gap_junction_modulation_regeneration,
    "gap_junction": gap_junction_modulation_regeneration,
}


def propose_experiment(
    required_measurements: list[str],
) -> ExperimentProposal | None:
    """Select the most appropriate experiment template.

    Returns None if no template matches.
    """
    for m in required_measurements:
        if m in _EXPERIMENT_TEMPLATES:
            return _EXPERIMENT_TEMPLATES[m]()

    # Default: if any bioelectric measurement requested, use Vmem imaging
    bioelectric_measures = {"vmem", "ef", "gj", "ion_flux"}
    if bioelectric_measures & set(required_measurements):
        return vmem_imaging_post_amputation()

    return None

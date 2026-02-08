# ACHERON NEXUS — CANONICAL LOGIC HASH v1.0

**Purpose**: Core rules that MUST survive any optimization. If any optimization breaks these rules, revert immediately.

---

## TITAN-LOGIC PHYSICS CONSTRAINTS (MANDATORY)

```
Nernst:     E_ion = (RT/zF) × ln([Ion]_out/[Ion]_in)

GHK:        Vm = (RT/F) × ln((P_K[K+]o + P_Na[Na+]o + P_Cl[Cl-]i)
                            /(P_K[K+]i + P_Na[Na+]i + P_Cl[Cl-]o))

Gibbs:      ΔG = -nFE  |  Stable if ΔG > 10kT (~25 kJ/mol)

Shannon:    C = B × log₂(1 + SNR)  |  H = log₂(N_states)
```

**Any claim violating these equations is REJECTED.**

---

## MULTI-AGENT ARCHITECTURE

| Agent | Role |
|-------|------|
| **Scraper** | Data retrieval only, no interpretation, flags paywalls |
| **Physicist** | Enforces Nernst/GHK/Gibbs, rejects non-physical claims |
| **Theorist** | Shannon entropy, BER, channel capacity, noise margins |
| **Lab Agent** | Converts hypotheses to protocols with PASS/FAIL criteria |

---

## NO NUMERIC INVENTION RULE (ABSOLUTE)

| Tag | Meaning |
|-----|---------|
| `[MEASURED]` | Cited organism-specific data (PMID/DOI) |
| `[SIMULATION-DERIVED]` | From validated simulation with stated params |
| `[BOUNDED-INFERENCE]` | Physics-constrained estimate, not biological fact |
| `[TRANSFER]` | Cross-species data, state source organism |
| `UNKNOWN` | No data exists → propose measurement |

---

## FALSIFICATION / KILL SWITCH (MANDATORY)

Every hypothesis MUST include a KILL CRITERIA:

> "If [measurable threshold] is exceeded, ABANDON this approach."

**No hypothesis without falsification path is valid.**

---

## 100-CELL CONSENSUS RESULTS (SIMULATION-DERIVED)

| Metric | Result | Source |
|--------|--------|--------|
| BER vs Cell Count | 10 cells/bit → BER < 10⁻³ | `simulations/ber_vs_cell_count.py` |
| Max Tolerable Noise | 8.0 mV (10 cells, ±10mV margin) | `simulations/noise_tolerance_sweep.py` |
| Bistability Required | Passive gap junctions FAIL | `simulations/stochastic_consensus_test.py` |
| Minimum 4-bit Memory | 40 cells total (10 per bit) | `simulations/ber_vs_cell_count.py` |
| Vmem Stability τ | ~40 ms, T_95% = 120 ms | `simulations/neoblast_vmem_stability.py` |

---

## HARDWARE BASELINES

| Species | Vmem Range | Gap Junctions | Notes |
|---------|------------|---------------|-------|
| Planarian (Dugesia japonica) | -20 to -60 mV | Innexins | τ_stab ≈ 40ms, extreme regeneration |
| Xenopus laevis | -50 to -80 mV | Connexins | Better characterized protocols |

---

## BIO-ISA (6 Operations)

```
SET_BIT(region, Vmem_target)    — Force bioelectric state
READ_BIT(region)                — Measure Vmem
GATE(region_A, region_B, state) — Open/Close gap junctions
AUTH(pattern)                   — Validate pattern integrity (CRC)
QUARANTINE(region)              — Isolate damaged region
REWRITE(region, pattern)        — Override bioelectric state
```

---

## VALIDATION CHECKLIST

Before any output, verify:

- [ ] No invented numbers (all values tagged or marked UNKNOWN)
- [ ] Physics constraints satisfied (Nernst, GHK, Gibbs)
- [ ] Kill criteria specified
- [ ] Simulation results cited where applicable
- [ ] Cross-species data labeled [TRANSFER]

---

**Hash Version**: 1.0
**Last Updated**: 2026-02-08
**Maintainer**: Project Acheron

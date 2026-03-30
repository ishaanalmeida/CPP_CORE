# Canon Ink Subsystem — Diagnostic Reasoner Engine

## Quick Start

```bash
# 1. Put all files in one directory
#    - Python files (*.py)
#    - action_catalogue_tuned.json (in same dir or specify path)
#    - KB files from project: observation_registry_v2.json, rc_registry_v2.json, 
#      cpt_store_v2.json, action_catalogue_v2.json, kb_metadata_v2.json

# 2. Automated demo (no interaction needed, simulates a scenario)
python demo.py --kb-dir ./kb --actions ./action_catalogue_tuned.json --error 250001 --auto

# 3. With XML datalog
python demo.py --kb-dir ./kb --actions ./action_catalogue_tuned.json --xml ./datalog.xml --error 250001 --auto

# 4. Interactive mode (you pick actions and outcomes)
python demo.py --kb-dir ./kb --actions ./action_catalogue_tuned.json --xml ./datalog.xml --error 250001

# 5. Multiple error codes
python demo.py --kb-dir ./kb --actions ./action_catalogue_tuned.json --error 250001 --error 290020 --error 290005 --auto
```

## Directory Setup

```
reasoner/
├── demo.py                         # CLI entry point
├── session_manager.py              # Orchestrates the diagnostic loop
├── reasoner.py                     # Naive Bayes posterior updating
├── action_selector.py              # EIG-based action ranking
├── kb_loader.py                    # Loads KB tables into indexed structures
├── datalog_parser.py               # Parses Canon XML datalog files
├── likelihood_tuner.py             # Three-tier outcome likelihood calibration
├── action_catalogue_tuned.json     # Action catalogue with Tier 1 RC mappings applied
├── tier1_mappings.json             # Tier 1 SDS mapping rationale (reference only)
└── kb/                             # KB JSON files (copy from project)
    ├── observation_registry_v2.json
    ├── rc_registry_v2.json
    ├── cpt_store_v2.json
    ├── action_catalogue_v2.json    # Original (unused if --actions is specified)
    └── kb_metadata_v2.json
```

## Usage Modes

### Interactive Mode (default)
Run without `--auto`. At each step you:
1. See ranked root cause hypotheses with probabilities
2. See recommended next actions ranked by expected information gain
3. Pick an action (enter its number)
4. Pick the observed outcome
5. See updated posteriors and next recommendations
6. Repeat until resolved or type 'q' to quit

Type 's' during the session for a full JSON summary.

### Automated Mode
Run with `--auto`. Simulates a technician: picks the top-ranked action each step,
selects "fail" for verification and "not_resolved" for resolution. Runs up to 5 steps.

## Arguments

| Argument    | Description |
|-------------|-------------|
| `--kb-dir`  | Directory containing KB JSON files (default: `/mnt/project`) |
| `--actions` | Path to tuned action catalogue (overrides `action_catalogue_v2.json` in kb-dir) |
| `--xml`     | Path to Canon datalog XML file (extracts ink errors, counters, expiry states) |
| `--error`   | Error code to inject (repeatable, e.g., `--error 250001 --error 290020`) |
| `--auto`    | Run automated non-interactive demo |

## Architecture

```
User Input (error codes, symptoms)
        │
        ▼
┌─────────────────┐     ┌──────────────────┐
│  DatalogParser   │────▶│  Observations    │
│  (XML extraction)│     │  (obs_id, value) │
└─────────────────┘     └────────┬─────────┘
                                 │
                                 ▼
                    ┌─────────────────────┐
                    │  NaiveBayesReasoner  │
                    │  P(RC|obs) update    │◀──────────────┐
                    └────────┬────────────┘               │
                             │                            │
                             ▼                            │
                    ┌─────────────────────┐    ┌──────────┴──────────┐
                    │   ActionSelector    │    │  Outcome registered │
                    │   EIG / cost rank   │    │  (user feedback)    │
                    └────────┬────────────┘    └─────────────────────┘
                             │
                             ▼
                    Recommended actions
                    (with outcome options)
```

## Likelihood Tiers

| Tier | Actions | Source | Description |
|------|---------|--------|-------------|
| 1    | 15 SDS tests | `calibrated_tier1` | Machine readings with mapped RCs. Near-deterministic. |
| 2    | 33 narrow checks | `calibrated_tier2` | 1-5 linked RCs, pass/fail/inconclusive. Calibrated by scope. |
| 3    | 77 remaining | `placeholder_discriminating` | Broad-scope or unlinked. Mildly discriminating placeholders. |

4 SDS tests remain `PENDING_DOMAIN_EXPERT` (flagged for Casper):
- 02-1-001 Mark ink tube as empty
- 02-1-007 White Ink Hardware Installed
- 02-6-002 Disable/Enable ink dosing
- 02-8-002 Number of inserted cartridges

## No External Dependencies
Pure Python 3.8+ standard library. No pip installs needed.

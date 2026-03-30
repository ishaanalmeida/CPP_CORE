"""
Knowledge Base Loader
Loads observation registry, RC registry, CPT store, and action catalogue
into indexed structures for fast lookup during reasoning.
"""

import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class Observation:
    obs_id: str
    obs_type: str  # error_code, symptom, counter, parameter
    obs_label: str
    subsystem: str
    domain: str
    bucketing_thresholds: Optional[Dict] = None
    source: str = ""
    notes: str = ""


@dataclass
class RootCause:
    rc_id: str
    rc_label: str
    subsystem: str
    prior_probability: float
    prior_count: int
    kb_validated: bool
    relevant_error_codes: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class CPTEntry:
    obs_id: str
    rc_id: str
    obs_value: str
    probability: float
    estimate_source: str
    sample_count: int
    notes: str = ""


@dataclass
class Action:
    action_id: str
    action_label: str
    action_type: str  # verification or resolution
    action_source: str
    subsystem: str
    linked_rc_ids: List[str]
    possible_outcomes: List[str]
    uses_global_outcome_set: bool
    outcome_likelihoods: List[Dict]  # [{rc_id, outcome_label, probability, ...}]
    cost_time: Optional[Dict] = None


class KnowledgeBase:
    """Indexed knowledge base for the ink subsystem reasoner."""

    def __init__(self, kb_dir: str, action_catalogue_path: str = None):
        self.kb_dir = Path(kb_dir)
        self.action_catalogue_path = action_catalogue_path
        
        # Core registries
        self.observations: Dict[str, Observation] = {}
        self.root_causes: Dict[str, RootCause] = {}
        self.actions: Dict[str, Action] = {}
        
        # CPT indexed as: cpt[(obs_id, rc_id, obs_value)] = probability
        self.cpt: Dict[tuple, float] = {}
        
        # Reverse indexes
        self.obs_by_type: Dict[str, List[str]] = defaultdict(list)
        self.obs_by_error_code: Dict[str, str] = {}  # error_code_value -> obs_id
        self.actions_by_rc: Dict[str, List[str]] = defaultdict(list)
        self.rc_by_error_code: Dict[str, List[str]] = defaultdict(list)
        
        # CPT coverage index: which (obs_id, obs_value) pairs exist for a given rc_id
        self.cpt_obs_for_rc: Dict[str, set] = defaultdict(set)
        # Which rc_ids have CPT entries for a given obs_id
        self.cpt_rcs_for_obs: Dict[str, set] = defaultdict(set)
        
        self._load()

    def _load(self):
        self._load_observations()
        self._load_root_causes()
        self._load_cpt()
        self._load_actions()

    def _load_observations(self):
        path = self.kb_dir / "observation_registry_v2.json"
        with open(path) as f:
            data = json.load(f)
        for item in data:
            obs = Observation(
                obs_id=item["obs_id"],
                obs_type=item["obs_type"],
                obs_label=item["obs_label"],
                subsystem=item.get("subsystem", ""),
                domain=item.get("domain", ""),
                bucketing_thresholds=item.get("bucketing_thresholds"),
                source=item.get("source", ""),
                notes=item.get("notes", ""),
            )
            self.observations[obs.obs_id] = obs
            self.obs_by_type[obs.obs_type].append(obs.obs_id)
            
            # Index error_code observations by their numeric code value
            # e.g., OBS_EC_290020 -> extract "290020"
            if obs.obs_type == "error_code":
                code = obs.obs_id.replace("OBS_EC_", "")
                self.obs_by_error_code[code] = obs.obs_id

    def _load_root_causes(self):
        path = self.kb_dir / "rc_registry_v2.json"
        with open(path) as f:
            data = json.load(f)
        for item in data:
            rc = RootCause(
                rc_id=item["rc_id"],
                rc_label=item["rc_label"],
                subsystem=item.get("subsystem", ""),
                prior_probability=item.get("prior_probability", 1.0 / len(data)),
                prior_count=item.get("prior_count", 0),
                kb_validated=item.get("kb_validated", False),
                relevant_error_codes=item.get("relevant_error_codes", []),
                notes=item.get("notes", ""),
            )
            self.root_causes[rc.rc_id] = rc
            
            # Index error codes -> root causes
            for code in rc.relevant_error_codes:
                self.rc_by_error_code[str(code)].append(rc.rc_id)

    def _load_cpt(self):
        path = self.kb_dir / "cpt_store_v2.json"
        with open(path) as f:
            data = json.load(f)
        for item in data:
            key = (item["obs_id"], item["rc_id"], str(item["obs_value"]))
            self.cpt[key] = item["probability"]
            self.cpt_obs_for_rc[item["rc_id"]].add((item["obs_id"], str(item["obs_value"])))
            self.cpt_rcs_for_obs[item["obs_id"]].add(item["rc_id"])

    def _load_actions(self):
        if self.action_catalogue_path:
            path = Path(self.action_catalogue_path)
        else:
            path = self.kb_dir / "action_catalogue_v2.json"
        with open(path) as f:
            data = json.load(f)
        for item in data:
            linked = item.get("linked_rc_ids", [])
            if isinstance(linked, str):
                # "PENDING_DOMAIN_EXPERT" - skip linking
                linked = []
            
            action = Action(
                action_id=item["action_id"],
                action_label=item["action_label"],
                action_type=item.get("action_type", ""),
                action_source=item.get("action_source", ""),
                subsystem=item.get("subsystem", ""),
                linked_rc_ids=linked,
                possible_outcomes=item.get("possible_outcomes", []),
                uses_global_outcome_set=item.get("uses_global_outcome_set", False),
                outcome_likelihoods=item.get("outcome_likelihoods", []),
                cost_time=item.get("cost_time"),
            )
            self.actions[action.action_id] = action
            
            for rc_id in action.linked_rc_ids:
                self.actions_by_rc[rc_id].append(action.action_id)

    def get_prior_distribution(self) -> Dict[str, float]:
        """Return normalized prior probability distribution over all root causes."""
        priors = {rc_id: rc.prior_probability for rc_id, rc in self.root_causes.items()}
        total = sum(priors.values())
        if total > 0:
            priors = {k: v / total for k, v in priors.items()}
        return priors

    def lookup_cpt(self, obs_id: str, rc_id: str, obs_value: str) -> Optional[float]:
        """Lookup P(obs_value | rc_id) from CPT. Returns None if not found."""
        key = (obs_id, rc_id, str(obs_value))
        return self.cpt.get(key)

    def get_actions_for_rcs(self, rc_ids: List[str]) -> List[Action]:
        """Get all actions linked to any of the given root cause IDs."""
        action_ids = set()
        for rc_id in rc_ids:
            action_ids.update(self.actions_by_rc.get(rc_id, []))
        return [self.actions[aid] for aid in action_ids if aid in self.actions]

    def summary(self) -> str:
        return (
            f"KnowledgeBase loaded:\n"
            f"  Observations: {len(self.observations)} "
            f"({', '.join(f'{t}:{len(ids)}' for t, ids in self.obs_by_type.items())})\n"
            f"  Root causes:  {len(self.root_causes)}\n"
            f"  CPT entries:  {len(self.cpt)}\n"
            f"  Actions:      {len(self.actions)} "
            f"(verification: {sum(1 for a in self.actions.values() if a.action_type=='verification')}, "
            f"resolution: {sum(1 for a in self.actions.values() if a.action_type=='resolution')})\n"
            f"  Action-RC links: {sum(len(a.linked_rc_ids) for a in self.actions.values())}"
        )

    def tune_likelihoods(self) -> Dict[str, int]:
        """
        Apply calibrated outcome likelihoods using the three-tier scheme.
        
        Returns dict with counts per tier: {"tier1": N, "tier2": N, "tier3": N}
        """
        from likelihood_tuner import tune_action_likelihoods
        
        # Reconstruct action dicts for the tuner (it works on raw dicts)
        action_dicts = []
        for action in self.actions.values():
            d = {
                "action_id": action.action_id,
                "action_label": action.action_label,
                "action_type": action.action_type,
                "linked_rc_ids": action.linked_rc_ids,
                "possible_outcomes": action.possible_outcomes,
                "uses_global_outcome_set": action.uses_global_outcome_set,
                "outcome_likelihoods": action.outcome_likelihoods,
            }
            action_dicts.append(d)
        
        all_rc_ids = set(self.root_causes.keys())
        tuned_count = tune_action_likelihoods(action_dicts, all_rc_ids)
        
        # Write tuned likelihoods back into Action dataclass instances
        tier_counts = {"tier1a": 0, "tier1b": 0, "tier2": 0, "tier3": 0}
        for d in action_dicts:
            action = self.actions.get(d["action_id"])
            if action is None:
                continue
            source = d.get("likelihood_source", "")
            if source:
                action.outcome_likelihoods = d["outcome_likelihoods"]
                if "tier1a" in source:
                    tier_counts["tier1a"] += 1
                elif "tier1b" in source:
                    tier_counts["tier1b"] += 1
                elif "tier2" in source:
                    tier_counts["tier2"] += 1
                elif "placeholder" in source:
                    tier_counts["tier3"] += 1
        
        return tier_counts

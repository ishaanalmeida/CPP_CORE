"""
Naive Bayes Reasoner
Computes posterior P(RC | observations) using:
- Prior distribution from RC registry
- CPT-based likelihood P(obs | RC) for each observation
- Sequential Bayesian updating as new evidence arrives

Handles:
- Missing CPT entries via uniform fallback (per kb_metadata policy)
- Unregistered observations via marginalisation (skip)
- Action outcome integration for sequential diagnosis
"""

import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from kb_loader import KnowledgeBase


@dataclass
class ReasonerState:
    """Tracks the current state of the diagnostic reasoning session."""
    # Current posterior distribution over root causes
    posterior: Dict[str, float] = field(default_factory=dict)
    
    # Evidence accumulated so far: list of (obs_id, obs_value) or (action_id, outcome)
    evidence_log: List[Dict] = field(default_factory=list)
    
    # Actions already performed (to avoid repeats)
    performed_actions: set = field(default_factory=set)
    
    # Root causes that have been ruled out (posterior < threshold)
    eliminated_rcs: set = field(default_factory=set)
    
    # Session status
    step_count: int = 0
    resolved: bool = False
    resolved_rc: Optional[str] = None


class NaiveBayesReasoner:
    """
    Naive Bayes diagnostic reasoner with sequential posterior updating.
    
    Core equation:
        P(RC_i | obs_1, ..., obs_n) ∝ P(RC_i) × ∏ P(obs_j | RC_i)
    
    With Bayesian updating per step:
        P(RC_i | obs_new, obs_prev) ∝ P(obs_new | RC_i) × P(RC_i | obs_prev)
    """

    # Root causes below this posterior get flagged as effectively eliminated
    ELIMINATION_THRESHOLD = 0.001
    
    # When a CPT entry is missing, use this uniform fallback
    # (per kb_metadata: missing_cpt_behaviour = uniform_fallback)
    UNIFORM_FALLBACK = 0.5
    
    # Smoothing factor to prevent zero probabilities
    LAPLACE_SMOOTHING = 1e-6

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def initialize(self, initial_observations: List[Tuple[str, str]] = None) -> ReasonerState:
        """
        Initialize a new reasoning session.
        
        Args:
            initial_observations: Optional list of (obs_id, obs_value) from user input + datalog
            
        Returns:
            ReasonerState with prior (or posterior if observations given)
        """
        state = ReasonerState()
        state.posterior = self.kb.get_prior_distribution()
        
        if initial_observations:
            self.update_with_observations(state, initial_observations)
        
        return state

    def update_with_observations(
        self, state: ReasonerState, observations: List[Tuple[str, str]]
    ) -> Dict[str, float]:
        """
        Update posterior with a batch of new observations.
        
        Args:
            state: Current reasoner state
            observations: List of (obs_id, obs_value) pairs
            
        Returns:
            Updated posterior distribution
        """
        for obs_id, obs_value in observations:
            self._update_single(state, obs_id, obs_value, source="observation")
        
        return state.posterior

    def update_with_action_outcome(
        self, state: ReasonerState, action_id: str, outcome: str
    ) -> Dict[str, float]:
        """
        Update posterior based on an action's outcome.
        
        Uses the action's outcome_likelihoods: P(outcome | RC) for each linked RC.
        
        Args:
            state: Current reasoner state
            action_id: The action that was performed
            outcome: The observed outcome label
            
        Returns:
            Updated posterior distribution
        """
        action = self.kb.actions.get(action_id)
        if action is None:
            return state.posterior
        
        state.performed_actions.add(action_id)
        
        # Build likelihood lookup: P(outcome | RC)
        outcome_probs = {}
        for entry in action.outcome_likelihoods:
            if entry["outcome_label"] == outcome:
                outcome_probs[entry["rc_id"]] = entry["probability"]
        
        # Special case for resolution actions: if outcome is "resolved",
        # the linked RCs get a strong boost
        if action.action_type == "resolution" and outcome in ("resolved", "pass"):
            for rc_id in action.linked_rc_ids:
                if rc_id not in outcome_probs:
                    outcome_probs[rc_id] = 0.9  # strong signal
        
        if action.action_type == "resolution" and outcome in ("not_resolved", "fail"):
            for rc_id in action.linked_rc_ids:
                if rc_id not in outcome_probs:
                    outcome_probs[rc_id] = 0.1  # strong negative signal
        
        # Update posterior
        new_posterior = {}
        for rc_id, prior_p in state.posterior.items():
            if rc_id in state.eliminated_rcs:
                new_posterior[rc_id] = self.LAPLACE_SMOOTHING
                continue
            
            likelihood = outcome_probs.get(rc_id, self.UNIFORM_FALLBACK)
            new_posterior[rc_id] = prior_p * likelihood
        
        # Normalize
        state.posterior = self._normalize(new_posterior)
        
        # Log evidence
        state.evidence_log.append({
            "type": "action_outcome",
            "action_id": action_id,
            "outcome": outcome,
            "step": state.step_count,
        })
        state.step_count += 1
        
        # Check for eliminated RCs
        self._update_eliminations(state)
        
        # Check if resolved
        if outcome in ("resolved",) and action.action_type == "resolution":
            top_rc = max(state.posterior, key=state.posterior.get)
            if state.posterior[top_rc] > 0.5:
                state.resolved = True
                state.resolved_rc = top_rc
        
        return state.posterior

    def _update_single(
        self, state: ReasonerState, obs_id: str, obs_value: str, source: str = "observation"
    ):
        """Update posterior with a single observation."""
        # Check if observation is registered
        if obs_id not in self.kb.observations:
            # Unregistered: marginalise (skip) per kb_metadata policy
            state.evidence_log.append({
                "type": source,
                "obs_id": obs_id,
                "obs_value": obs_value,
                "status": "skipped_unregistered",
                "step": state.step_count,
            })
            return
        
        new_posterior = {}
        for rc_id, prior_p in state.posterior.items():
            if rc_id in state.eliminated_rcs:
                new_posterior[rc_id] = self.LAPLACE_SMOOTHING
                continue
            
            # Look up P(obs_value | rc_id)
            likelihood = self.kb.lookup_cpt(obs_id, rc_id, obs_value)
            
            if likelihood is None:
                # Missing CPT entry: use uniform fallback
                likelihood = self.UNIFORM_FALLBACK
            
            # Apply Laplace smoothing to prevent zero
            likelihood = max(likelihood, self.LAPLACE_SMOOTHING)
            
            new_posterior[rc_id] = prior_p * likelihood
        
        # Normalize
        state.posterior = self._normalize(new_posterior)
        
        # Log
        state.evidence_log.append({
            "type": source,
            "obs_id": obs_id,
            "obs_value": obs_value,
            "step": state.step_count,
        })
        state.step_count += 1
        
        # Update eliminations
        self._update_eliminations(state)

    def _normalize(self, distribution: Dict[str, float]) -> Dict[str, float]:
        """Normalize a probability distribution to sum to 1."""
        total = sum(distribution.values())
        if total <= 0:
            # Fallback to uniform
            n = len(distribution)
            return {k: 1.0 / n for k in distribution}
        return {k: v / total for k, v in distribution.items()}

    def _update_eliminations(self, state: ReasonerState):
        """Mark root causes with very low posterior as eliminated."""
        for rc_id, prob in state.posterior.items():
            if prob < self.ELIMINATION_THRESHOLD:
                state.eliminated_rcs.add(rc_id)

    def get_ranked_hypotheses(
        self, state: ReasonerState, top_n: int = 10
    ) -> List[Dict]:
        """
        Get ranked list of root cause hypotheses.
        
        Returns:
            List of {rc_id, rc_label, probability, rank} dicts, sorted by probability desc.
        """
        ranked = sorted(
            state.posterior.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        results = []
        for rank, (rc_id, prob) in enumerate(ranked[:top_n], 1):
            rc = self.kb.root_causes.get(rc_id)
            results.append({
                "rank": rank,
                "rc_id": rc_id,
                "rc_label": rc.rc_label if rc else rc_id,
                "probability": round(prob, 6),
                "eliminated": rc_id in state.eliminated_rcs,
            })
        
        return results

    def entropy(self, state: ReasonerState) -> float:
        """
        Compute Shannon entropy of the current posterior.
        Lower entropy = more certainty about the root cause.
        """
        h = 0.0
        for p in state.posterior.values():
            if p > 0:
                h -= p * math.log2(p)
        return h

    def max_posterior(self, state: ReasonerState) -> Tuple[str, float]:
        """Return the RC with highest posterior probability."""
        best_rc = max(state.posterior, key=state.posterior.get)
        return best_rc, state.posterior[best_rc]

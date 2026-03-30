"""
Action Selector
Selects the next best action based on:
1. Expected Information Gain (EIG) - how much the action reduces uncertainty
2. Cost awareness - prefer cheaper/faster actions when EIG is similar

Strategy:
- For each candidate action, compute expected posterior entropy after each possible outcome
- EIG = current_entropy - expected_posterior_entropy
- Score = EIG / cost (or EIG alone if cost data is missing)
- Prioritize verification actions over resolution actions (verify first, then fix)
"""

import math
from typing import Dict, List, Optional, Tuple
from copy import deepcopy

from kb_loader import KnowledgeBase, Action
from reasoner import NaiveBayesReasoner, ReasonerState


class ActionSelector:
    """Selects the next best diagnostic/repair action."""

    # Default cost for actions without cost data (minutes)
    DEFAULT_COST_MINUTES = 15
    
    # Minimum probability threshold for a RC to be "confident enough" to attempt resolution
    RESOLUTION_CONFIDENCE_THRESHOLD = 0.4

    def __init__(self, kb: KnowledgeBase, reasoner: NaiveBayesReasoner):
        self.kb = kb
        self.reasoner = reasoner

    def select_next_action(
        self, state: ReasonerState, top_n: int = 5
    ) -> List[Dict]:
        """
        Select and rank the best next actions.
        
        Strategy:
        1. If top RC has high confidence -> suggest resolution for that RC
        2. Otherwise -> suggest verification actions that maximize information gain
        
        Args:
            state: Current reasoner state
            top_n: Number of actions to return
            
        Returns:
            Ranked list of {action_id, action_label, score, eig, cost, rationale, outcomes}
        """
        current_entropy = self.reasoner.entropy(state)
        
        # Get active (non-eliminated) RCs with meaningful probability
        active_rcs = [
            rc_id for rc_id, prob in state.posterior.items()
            if prob > self.reasoner.ELIMINATION_THRESHOLD and rc_id not in state.eliminated_rcs
        ]
        
        # Get candidate actions linked to active RCs
        candidates = self._get_candidate_actions(state, active_rcs)
        
        if not candidates:
            return [{
                "action_id": None,
                "action_label": "No more actions available",
                "score": 0,
                "eig": 0,
                "cost": 0,
                "rationale": "All available actions have been performed or no actions link to remaining hypotheses.",
                "outcomes": [],
            }]
        
        # Score each candidate
        scored = []
        for action in candidates:
            eig = self._compute_eig(state, action, current_entropy)
            cost = self._get_cost(action)
            
            # Score = EIG / cost, with bonus for verification actions
            type_bonus = 1.2 if action.action_type == "verification" else 1.0
            
            # Check if this is a resolution action for the top RC
            top_rc, top_prob = self.reasoner.max_posterior(state)
            is_top_resolution = (
                action.action_type == "resolution"
                and top_rc in action.linked_rc_ids
                and top_prob >= self.RESOLUTION_CONFIDENCE_THRESHOLD
            )
            
            if is_top_resolution:
                # Boost resolution actions when confidence is high
                type_bonus = 1.5
            
            score = (eig * type_bonus) / max(cost, 1)
            
            rationale = self._build_rationale(action, eig, state)
            
            scored.append({
                "action_id": action.action_id,
                "action_label": action.action_label,
                "action_type": action.action_type,
                "score": round(score, 6),
                "eig": round(eig, 6),
                "cost_minutes": cost,
                "rationale": rationale,
                "outcomes": action.possible_outcomes,
                "linked_rcs": action.linked_rc_ids[:5],  # Truncate for display
            })
        
        # Sort by score descending
        scored.sort(key=lambda x: x["score"], reverse=True)
        
        return scored[:top_n]

    def _get_candidate_actions(
        self, state: ReasonerState, active_rcs: List[str]
    ) -> List[Action]:
        """Get actions that haven't been performed yet and link to active RCs."""
        candidates = []
        seen = set()
        
        for rc_id in active_rcs:
            for action_id in self.kb.actions_by_rc.get(rc_id, []):
                if action_id in state.performed_actions:
                    continue
                if action_id in seen:
                    continue
                seen.add(action_id)
                
                action = self.kb.actions.get(action_id)
                if action and len(action.possible_outcomes) >= 2:
                    candidates.append(action)
        
        return candidates

    def _compute_eig(
        self, state: ReasonerState, action: Action, current_entropy: float
    ) -> float:
        """
        Compute Expected Information Gain for an action.
        
        EIG = H(RC | current evidence) - E[H(RC | current evidence, action outcome)]
            = current_entropy - Σ_outcome P(outcome) × H(RC | outcome)
        
        Where P(outcome) = Σ_rc P(outcome | rc) × P(rc | current evidence)
        """
        expected_posterior_entropy = 0.0
        
        for outcome in action.possible_outcomes:
            # Compute P(outcome) = Σ_rc P(outcome | rc) × P(rc)
            p_outcome = 0.0
            outcome_likelihoods = {}
            
            for entry in action.outcome_likelihoods:
                if entry["outcome_label"] == outcome:
                    rc_id = entry["rc_id"]
                    p_out_given_rc = entry["probability"]
                    outcome_likelihoods[rc_id] = p_out_given_rc
                    p_outcome += p_out_given_rc * state.posterior.get(rc_id, 0)
            
            if p_outcome <= 0:
                continue
            
            # Compute posterior given this outcome
            hypothetical_posterior = {}
            for rc_id, prior_p in state.posterior.items():
                if rc_id in state.eliminated_rcs:
                    hypothetical_posterior[rc_id] = 1e-10
                    continue
                
                likelihood = outcome_likelihoods.get(rc_id, 0.5)  # uniform fallback
                hypothetical_posterior[rc_id] = prior_p * likelihood
            
            # Normalize
            total = sum(hypothetical_posterior.values())
            if total > 0:
                hypothetical_posterior = {k: v / total for k, v in hypothetical_posterior.items()}
            
            # Compute entropy of this hypothetical posterior
            h = 0.0
            for p in hypothetical_posterior.values():
                if p > 0:
                    h -= p * math.log2(p)
            
            expected_posterior_entropy += p_outcome * h
        
        eig = current_entropy - expected_posterior_entropy
        return max(eig, 0)  # EIG should be non-negative

    def _get_cost(self, action: Action) -> float:
        """Extract cost in minutes from action, with default fallback."""
        if action.cost_time and isinstance(action.cost_time, dict):
            return action.cost_time.get("minutes", self.DEFAULT_COST_MINUTES)
        return self.DEFAULT_COST_MINUTES

    def _build_rationale(self, action: Action, eig: float, state: ReasonerState) -> str:
        """Build human-readable rationale for why this action is recommended."""
        parts = []
        
        if action.action_type == "verification":
            parts.append("Diagnostic check")
        else:
            parts.append("Repair action")
        
        # Which RCs does this action help discriminate?
        relevant_rcs = []
        for rc_id in action.linked_rc_ids[:3]:
            rc = self.kb.root_causes.get(rc_id)
            prob = state.posterior.get(rc_id, 0)
            if rc and prob > 0.01:
                relevant_rcs.append(f"{rc.rc_label} ({prob:.1%})")
        
        if relevant_rcs:
            parts.append(f"targets: {', '.join(relevant_rcs)}")
        
        if eig > 0.5:
            parts.append("high information gain")
        elif eig > 0.1:
            parts.append("moderate information gain")
        else:
            parts.append("low information gain")
        
        return " | ".join(parts)

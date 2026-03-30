"""
Session Manager
Orchestrates the full diagnostic reasoning loop:
1. Accept initial observations (user input + XML datalog)
2. Compute initial posterior over root causes
3. Recommend next best action
4. Accept action outcome from user
5. Update posterior and repeat until resolved or exhausted

This is the main interface for the demo.
"""

import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from kb_loader import KnowledgeBase
from datalog_parser import DatalogParser, DatalogExtraction
from reasoner import NaiveBayesReasoner, ReasonerState
from action_selector import ActionSelector


@dataclass
class SessionSnapshot:
    """A snapshot of the session state at a given step, for logging/replay."""
    step: int
    event_type: str  # "init", "observation", "action_outcome"
    event_data: Dict
    top_hypotheses: List[Dict]
    recommended_actions: List[Dict]
    entropy: float
    max_posterior_rc: str
    max_posterior_prob: float


class DiagnosticSession:
    """Manages a single diagnostic reasoning session."""

    def __init__(self, kb_dir: str, lookback_days: int = 30, action_catalogue_path: str = None):
        # Load knowledge base
        self.kb = KnowledgeBase(kb_dir, action_catalogue_path=action_catalogue_path)
        
        # Apply calibrated outcome likelihoods (3-tier scheme)
        self.tuning_stats = self.kb.tune_likelihoods()
        
        # Initialize components
        self.parser = DatalogParser(lookback_days=lookback_days)
        self.reasoner = NaiveBayesReasoner(self.kb)
        self.selector = ActionSelector(self.kb, self.reasoner)
        
        # Session state
        self.state: Optional[ReasonerState] = None
        self.datalog: Optional[DatalogExtraction] = None
        self.history: List[SessionSnapshot] = []
        
        # All observations collected
        self.all_observations: List[Tuple[str, str]] = []

    def start(
        self,
        user_observations: List[Tuple[str, str]] = None,
        xml_path: str = None,
    ) -> Dict:
        """
        Start a new diagnostic session.
        
        Args:
            user_observations: Manual observations as (obs_id, obs_value) pairs.
                              Can also be raw error codes like [("290020", "present")]
                              which will be mapped to obs_ids.
            xml_path: Path to datalog XML file for automatic observation extraction.
            
        Returns:
            Dict with initial hypotheses and recommended actions.
        """
        observations = []
        
        # Process user-entered observations
        if user_observations:
            for obs_id, obs_value in user_observations:
                # Try to map raw error codes to obs_ids
                mapped_id = self.kb.obs_by_error_code.get(obs_id, obs_id)
                # Normalize obs_value: "present" -> "1" for error codes
                if mapped_id.startswith("OBS_EC_") and obs_value in ("present", "true", "yes", "1"):
                    obs_value = "1"
                elif mapped_id.startswith("OBS_EC_") and obs_value in ("absent", "false", "no", "0"):
                    obs_value = "0"
                observations.append((mapped_id, obs_value))
        
        # Process XML datalog
        if xml_path:
            self.datalog = self.parser.parse(xml_path)
            xml_obs = self.parser.extract_observations(self.datalog, self.kb)
            observations.extend(xml_obs)
        
        # Deduplicate observations (keep first occurrence)
        seen = set()
        unique_obs = []
        for obs in observations:
            if obs[0] not in seen:
                seen.add(obs[0])
                unique_obs.append(obs)
        
        self.all_observations = unique_obs
        
        # Initialize reasoner with observations
        self.state = self.reasoner.initialize(unique_obs)
        
        # Get initial results
        hypotheses = self.reasoner.get_ranked_hypotheses(self.state)
        actions = self.selector.select_next_action(self.state)
        entropy = self.reasoner.entropy(self.state)
        top_rc, top_prob = self.reasoner.max_posterior(self.state)
        
        # Log snapshot
        action_step = len([h for h in self.history if h.event_type == "action_outcome"])
        snapshot = SessionSnapshot(
            step=action_step,
            event_type="init",
            event_data={
                "observations": [(o[0], o[1]) for o in unique_obs],
                "datalog_parsed": xml_path is not None,
            },
            top_hypotheses=hypotheses,
            recommended_actions=actions,
            entropy=entropy,
            max_posterior_rc=top_rc,
            max_posterior_prob=top_prob,
        )
        self.history.append(snapshot)
        
        return {
            "status": "session_started",
            "observations_loaded": len(unique_obs),
            "observation_details": [(o[0], o[1]) for o in unique_obs],
            "hypotheses": hypotheses,
            "recommended_actions": actions,
            "entropy": round(entropy, 4),
            "top_rc": top_rc,
            "top_rc_label": self.kb.root_causes[top_rc].rc_label if top_rc in self.kb.root_causes else top_rc,
            "top_rc_prob": round(top_prob, 4),
            "datalog_summary": self.parser.summary(self.datalog) if self.datalog else None,
        }

    def register_outcome(self, action_id: str, outcome: str) -> Dict:
        """
        Register the outcome of a performed action and get updated recommendations.
        
        Args:
            action_id: The action that was performed
            outcome: The observed outcome (must be one of action's possible_outcomes)
            
        Returns:
            Dict with updated hypotheses and next recommended actions.
        """
        if self.state is None:
            return {"error": "Session not started. Call start() first."}
        
        # Validate action and outcome
        action = self.kb.actions.get(action_id)
        if action is None:
            return {"error": f"Unknown action: {action_id}"}
        
        if outcome not in action.possible_outcomes:
            return {
                "error": f"Invalid outcome '{outcome}' for action {action_id}. "
                         f"Valid outcomes: {action.possible_outcomes}"
            }
        
        # Update posterior
        self.reasoner.update_with_action_outcome(self.state, action_id, outcome)
        
        # Get updated results
        hypotheses = self.reasoner.get_ranked_hypotheses(self.state)
        next_actions = self.selector.select_next_action(self.state)
        entropy = self.reasoner.entropy(self.state)
        top_rc, top_prob = self.reasoner.max_posterior(self.state)
        
        # Log snapshot
        action_step = len([h for h in self.history if h.event_type == "action_outcome"]) + 1
        snapshot = SessionSnapshot(
            step=action_step,
            event_type="action_outcome",
            event_data={"action_id": action_id, "outcome": outcome},
            top_hypotheses=hypotheses,
            recommended_actions=next_actions,
            entropy=entropy,
            max_posterior_rc=top_rc,
            max_posterior_prob=top_prob,
        )
        self.history.append(snapshot)
        
        result = {
            "status": "resolved" if self.state.resolved else "in_progress",
            "step": action_step,
            "action_performed": action.action_label,
            "outcome": outcome,
            "hypotheses": hypotheses,
            "recommended_actions": next_actions,
            "entropy": round(entropy, 4),
            "entropy_reduction": round(self.history[-2].entropy - entropy, 4) if len(self.history) > 1 else 0,
            "top_rc": top_rc,
            "top_rc_label": self.kb.root_causes[top_rc].rc_label if top_rc in self.kb.root_causes else top_rc,
            "top_rc_prob": round(top_prob, 4),
            "active_hypotheses": sum(1 for p in self.state.posterior.values() if p > 0.01),
            "eliminated_count": len(self.state.eliminated_rcs),
        }
        
        if self.state.resolved:
            result["resolution"] = {
                "root_cause": self.state.resolved_rc,
                "label": self.kb.root_causes[self.state.resolved_rc].rc_label 
                         if self.state.resolved_rc in self.kb.root_causes else "",
                "confidence": round(top_prob, 4),
                "steps_taken": self.state.step_count,
            }
        
        return result

    def get_action_outcomes(self, action_id: str) -> Optional[List[str]]:
        """Get the possible outcomes for an action (for user selection)."""
        action = self.kb.actions.get(action_id)
        if action:
            return action.possible_outcomes
        return None

    def get_session_summary(self) -> Dict:
        """Get a summary of the entire session for logging/review."""
        if self.state is None:
            return {"error": "No active session"}
        
        return {
            "steps": self.state.step_count,
            "resolved": self.state.resolved,
            "resolved_rc": self.state.resolved_rc,
            "final_entropy": round(self.reasoner.entropy(self.state), 4),
            "observations_count": len(self.all_observations),
            "actions_performed": list(self.state.performed_actions),
            "eliminated_rcs": len(self.state.eliminated_rcs),
            "evidence_log": self.state.evidence_log,
            "history": [
                {
                    "step": s.step,
                    "type": s.event_type,
                    "entropy": round(s.entropy, 4),
                    "top_rc": s.max_posterior_rc,
                    "top_prob": round(s.max_posterior_prob, 4),
                }
                for s in self.history
            ],
        }

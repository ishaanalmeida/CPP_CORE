"""
Interactive CLI Demo
Runs the full diagnostic reasoning loop with the Canon ink subsystem KB.

Usage:
    python demo.py [--kb-dir ./kb] [--actions ./action_catalogue_tuned.json] [--xml datalog.xml] [--error 290005]
"""

import sys
import json
import argparse
from typing import List, Tuple

from session_manager import DiagnosticSession


def print_header(text: str):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")


def print_hypotheses(hypotheses: List[dict], max_show: int = 10):
    print(f"\n  {'Rank':<5} {'Probability':>11}  {'Root Cause'}")
    print(f"  {'─'*5} {'─'*11}  {'─'*50}")
    for h in hypotheses[:max_show]:
        marker = " ✗" if h["eliminated"] else ""
        print(f"  {h['rank']:<5} {h['probability']:>10.4%}  {h['rc_label']}{marker}")


def print_action(action: dict, index: int = 1):
    """Print a single action with full detail."""
    if action["action_id"] is None:
        print(f"\n  No more actions available.")
        print(f"  {action['rationale']}")
        return

    print(f"\n  Recommended action:")
    print(f"  ┌─────────────────────────────────────────────────────────────────")
    print(f"  │  [{action['action_type'].upper()}] {action['action_label']}")
    print(f"  │")
    print(f"  │  EIG score: {action['score']:.4f}   |   Cost: {action['cost_minutes']} min")
    print(f"  │  {action['rationale']}")
    print(f"  │")
    print(f"  │  Possible outcomes:")
    for i, o in enumerate(action["outcomes"], 1):
        print(f"  │    {i}. {o}")
    print(f"  └─────────────────────────────────────────────────────────────────")


def run_demo(kb_dir: str, xml_path: str = None, error_codes: List[str] = None, action_catalogue_path: str = None):
    """Run the interactive diagnostic demo."""
    
    print_header("Canon Ink Subsystem — Diagnostic Reasoner")
    
    session = DiagnosticSession(kb_dir=kb_dir, lookback_days=90, action_catalogue_path=action_catalogue_path)
    print(f"\n{session.kb.summary()}")
    ts = session.tuning_stats
    print(f"  Likelihood tuning: tier1a={ts.get('tier1a',0)}, tier1b={ts.get('tier1b',0)}, tier2={ts.get('tier2',0)}, tier3={ts.get('tier3',0)}")
    
    # Prepare initial observations
    user_obs: List[Tuple[str, str]] = []
    if error_codes:
        for code in error_codes:
            user_obs.append((code, "present"))
    
    # Start session
    print_header("Initializing Session")
    
    result = session.start(
        user_observations=user_obs if user_obs else None,
        xml_path=xml_path,
    )
    
    if result.get("datalog_summary"):
        lines = result["datalog_summary"].split("\n")
        for line in lines[:5]:
            print(f"    {line}")
    
    print(f"\n  Observations loaded: {result['observations_loaded']}")
    if result["observation_details"]:
        for obs_id, obs_val in result["observation_details"][:15]:
            obs = session.kb.observations.get(obs_id)
            label = obs.obs_label if obs else obs_id
            print(f"    • {obs_id} = {obs_val}  ({label})")
    
    print(f"\n  Initial entropy: {result['entropy']:.4f} bits")
    print(f"  Top hypothesis: {result['top_rc_label']} ({result['top_rc_prob']:.2%})")
    
    print_hypotheses(result["hypotheses"])
    
    # Interactive loop
    user_step = 0
    while not session.state.resolved:
        user_step += 1
        actions = result["recommended_actions"]
        top_action = actions[0] if actions else None
        
        if top_action is None or top_action["action_id"] is None:
            print(f"\n  No more actions available. Hypothesis set exhausted.")
            break
        
        print_header(f"Step {user_step}")
        print_action(top_action)
        
        print(f"\n  [Enter outcome number]  [s = skip]  [q = quit]  [d = details]")
        
        try:
            choice = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        
        if choice == 'q':
            break
        
        if choice == 'd':
            # Show full hypothesis table and alternative actions
            print_hypotheses(result["hypotheses"])
            print(f"\n  Alternative actions:")
            for i, a in enumerate(actions[1:5], 2):
                if a["action_id"]:
                    print(f"    {i}. [{a['action_type']}] {a['action_label']}")
                    print(f"       EIG={a['score']:.4f}")
            continue
        
        if choice == 's':
            # Skip = register as inconclusive (or the neutral outcome)
            outcomes = top_action["outcomes"]
            skip_outcome = None
            for o in outcomes:
                if o.lower() in ("inconclusive", "pass"):
                    skip_outcome = o
                    break
            if skip_outcome is None:
                # For 2-outcome tests, pick the "normal" one (index 0 or adjusted)
                skip_outcome = outcomes[0]
            
            print(f"  Skipped → registered as '{skip_outcome}'")
            result = session.register_outcome(top_action["action_id"], skip_outcome)
        else:
            try:
                outcome_idx = int(choice) - 1
            except ValueError:
                print("  Invalid input. Enter outcome number, 's' to skip, or 'q' to quit.")
                user_step -= 1
                continue
            
            outcomes = top_action["outcomes"]
            if outcome_idx < 0 or outcome_idx >= len(outcomes):
                print(f"  Invalid. Enter 1-{len(outcomes)}.")
                user_step -= 1
                continue
            
            selected_outcome = outcomes[outcome_idx]
            result = session.register_outcome(top_action["action_id"], selected_outcome)
        
        # Show result
        outcome_display = result.get("outcome", "")
        print(f"\n  Result: {result['action_performed']}")
        print(f"          → {outcome_display}")
        print(f"\n  Entropy: {result['entropy']:.4f} bits (Δ = {result['entropy_reduction']:+.4f})")
        print(f"  Top hypothesis: {result['top_rc_label']} ({result['top_rc_prob']:.2%})")
        print(f"  Active: {result['active_hypotheses']}  |  Eliminated: {result['eliminated_count']}")
        
        print_hypotheses(result["hypotheses"], max_show=5)
        
        if result["status"] == "resolved":
            res = result["resolution"]
            print_header("RESOLVED")
            print(f"  Root cause: {res['label']}")
            print(f"  Confidence: {res['confidence']:.2%}")
            print(f"  Steps taken: {user_step}")
            break
    
    # Session summary
    print_header("Session Summary")
    summary = session.get_session_summary()
    print(f"  Steps:           {user_step}")
    print(f"  Resolved:        {summary['resolved']}")
    print(f"  Final entropy:   {summary['final_entropy']:.4f} bits")
    print(f"  Actions taken:   {len(summary['actions_performed'])}")
    print(f"  Eliminated RCs:  {summary['eliminated_rcs']}")
    
    print(f"\n  Entropy trajectory:")
    for h in summary["history"]:
        print(f"    Step {h['step']:>2}: H={h['entropy']:.4f}  top={h['top_rc'][:40]} ({h['top_prob']:.2%})")


def run_automated_demo(kb_dir: str, xml_path: str = None, error_codes: List[str] = None, action_catalogue_path: str = None):
    """Run a non-interactive demo."""
    
    print_header("Canon Ink Subsystem — Automated Demo")
    
    session = DiagnosticSession(kb_dir=kb_dir, lookback_days=90, action_catalogue_path=action_catalogue_path)
    print(f"\n{session.kb.summary()}")
    ts = session.tuning_stats
    print(f"  Likelihood tuning: tier1a={ts.get('tier1a',0)}, tier1b={ts.get('tier1b',0)}, tier2={ts.get('tier2',0)}, tier3={ts.get('tier3',0)}")
    
    user_obs = [(code, "present") for code in (error_codes or [])]
    
    result = session.start(
        user_observations=user_obs if user_obs else None,
        xml_path=xml_path,
    )
    
    print(f"\n  Observations loaded: {result['observations_loaded']}")
    for obs_id, obs_val in result["observation_details"][:10]:
        obs = session.kb.observations.get(obs_id)
        label = obs.obs_label if obs else obs_id
        print(f"    • {obs_id} = {obs_val}  ({label})")
    
    print(f"\n  Entropy: {result['entropy']:.4f} bits")
    print(f"  Top: {result['top_rc_label']} ({result['top_rc_prob']:.2%})")
    print_hypotheses(result["hypotheses"], max_show=5)
    
    # Simulate: always top action, "fail" for verification, "not_resolved" for resolution
    outcome_pick = {
        "verification": lambda o: next((x for x in o if x.lower() in ("fail", "expired", "not safe", "not supported", "not functioning", "not installed")), o[1] if len(o) == 2 else o[0]),
        "resolution": lambda o: next((x for x in o if "not_resolved" in x.lower()), o[-1]),
    }
    
    for step in range(1, 8):
        actions = result["recommended_actions"]
        if not actions or actions[0]["action_id"] is None:
            break
        
        top = actions[0]
        picker = outcome_pick.get(top.get("action_type", ""), lambda o: o[0])
        outcome = picker(top["outcomes"])
        
        print(f"\n  Step {step}: {top['action_label']}")
        print(f"    → {outcome}")
        
        result = session.register_outcome(top["action_id"], outcome)
        
        print(f"    Entropy: {result['entropy']:.4f} (Δ={result['entropy_reduction']:+.4f})")
        print(f"    Top: {result['top_rc_label']} ({result['top_rc_prob']:.2%})")
        print(f"    Active: {result['active_hypotheses']} | Eliminated: {result['eliminated_count']}")
        
        if result["status"] == "resolved":
            print_header("RESOLVED")
            print(f"  Root cause: {result['resolution']['label']}")
            print(f"  Confidence: {result['resolution']['confidence']:.2%}")
            print(f"  Steps: {step}")
            break
    
    print_header("Final")
    s = session.get_session_summary()
    print(f"  Steps: {len(s['actions_performed'])}, Resolved: {s['resolved']}, Entropy: {s['final_entropy']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canon Ink Subsystem Diagnostic Reasoner")
    parser.add_argument("--kb-dir", default="./kb", help="Path to KB JSON files")
    parser.add_argument("--actions", default=None, help="Path to tuned action catalogue")
    parser.add_argument("--xml", default=None, help="Path to datalog XML file")
    parser.add_argument("--error", action="append", default=[], help="Error code(s)")
    parser.add_argument("--auto", action="store_true", help="Automated demo")
    
    args = parser.parse_args()
    
    if args.auto:
        run_automated_demo(args.kb_dir, args.xml, args.error, args.actions)
    else:
        run_demo(args.kb_dir, args.xml, args.error, args.actions)

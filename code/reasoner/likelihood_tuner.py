"""
Likelihood Tuner
Applies calibrated P(outcome | RC) values to the action catalogue
based on a tiered scheme:

Tier 1a — Deterministic parameter reads (expiry date, safe mode, ink type):
  P(normal | linked_RC) = 0.01   — machine says "not expired", it's not expired
  P(abnormal | linked_RC) = 0.95
  P(normal | unlinked_RC) = 0.98
  P(abnormal | unlinked_RC) = 0.02

Tier 1b — Sensor/actuator tests (leak sensor, level sensors, valve activation):
  1 RC:    P(abnormal|linked)=0.95, P(normal|linked)=0.05
  2-3 RCs: P(abnormal|linked)=0.85, P(normal|linked)=0.10
  4+ RCs:  P(abnormal|linked)=0.75, P(normal|linked)=0.15
  All:     P(abnormal|unlinked)=0.03, P(normal|unlinked)=0.95

Tier 2 — Narrow-scope manual checks (1-5 linked RCs, pass/fail/inconclusive):
  1 RC:  P(fail|linked)=0.90, P(pass|linked)=0.05, P(inconcl|linked)=0.05
         P(fail|unlinked)=0.03, P(pass|unlinked)=0.92, P(inconcl|unlinked)=0.05
  2 RCs: P(fail|linked)=0.80, P(pass|linked)=0.10, P(inconcl|linked)=0.10
         P(fail|unlinked)=0.05, P(pass|unlinked)=0.87, P(inconcl|unlinked)=0.08
  3-5:   P(fail|linked)=0.70, P(pass|linked)=0.15, P(inconcl|linked)=0.15
         P(fail|unlinked)=0.05, P(pass|unlinked)=0.87, P(inconcl|unlinked)=0.08

Tier 3 — Everything else: mildly discriminating placeholders.
"""

from typing import Dict, List, Set


# Explicit polarity: which outcome index is ABNORMAL per SDS test.
# Default is index 1. Override here when index 0 is the abnormal finding.
SDS_ABNORMAL_INDEX_OVERRIDE = {
    # Leak sensor: "Detects Ink" (idx 0) = ink in leak tray = abnormal
    "ACT_SDS_006_022102_Check_INK_LEAK_SE_02B18_Sens": 0,
    # HIGH/LOW level sensors: "Ink detected; Refill not allowed" (idx 0)
    # Flagged for Casper — ambiguous without knowing actual tank state.
    # Treating idx 0 as abnormal for now (unexpected reading when sensor defective).
    "ACT_SDS_007_022103_Check_INK_x_HIGH_SE_C_M_Y_K_": 0,
    "ACT_SDS_008_022104_Check_INK_x_LOW_SE_C_M_Y_K_o": 0,
}

# Tier 1a: deterministic parameter reads (direct register values)
TIER_1A_ACTIONS = {
    "ACT_SDS_001_021003_Expire_date_of_the_intermedi",
    "ACT_SDS_002_021004_Expire_state_of_the_ink_in_t",
    "ACT_SDS_003_021005_Ink_safe_mode_state_C_M_Y_K_",
    "ACT_SDS_004_021006_The_supported_ink_type_Param",
}

TIER2_MAX_LINKED_RCS = 5


def tune_action_likelihoods(actions: List[Dict], all_rc_ids: Set[str]) -> int:
    tuned = 0
    for action in actions:
        linked = action.get("linked_rc_ids", [])
        if isinstance(linked, str):
            continue
        linked_set = set(linked)
        outcomes = action.get("possible_outcomes", [])
        if len(outcomes) < 2:
            continue

        aid = action["action_id"]
        is_sds = aid.startswith("ACT_SDS_")
        is_chk = aid.startswith("ACT_CHK_")
        is_global = action.get("uses_global_outcome_set", False)
        n_linked = len(linked_set)

        if is_sds and n_linked > 0 and aid in TIER_1A_ACTIONS:
            action["outcome_likelihoods"] = _build_tier1a(aid, outcomes, linked_set, all_rc_ids)
            action["likelihood_source"] = "calibrated_tier1a"
            tuned += 1
        elif is_sds and n_linked > 0:
            action["outcome_likelihoods"] = _build_tier1b(aid, outcomes, linked_set, all_rc_ids, n_linked)
            action["likelihood_source"] = "calibrated_tier1b"
            tuned += 1
        elif is_chk and is_global and 0 < n_linked <= TIER2_MAX_LINKED_RCS and set(outcomes) == {"pass", "fail", "inconclusive"}:
            action["outcome_likelihoods"] = _build_tier2(linked_set, all_rc_ids, n_linked)
            action["likelihood_source"] = "calibrated_tier2"
            tuned += 1
        else:
            has_uniform = any(e.get("estimate_source") == "uniform_placeholder" for e in action.get("outcome_likelihoods", []))
            if has_uniform and n_linked > 0:
                action["outcome_likelihoods"] = _build_tier3(outcomes, linked_set, all_rc_ids, action.get("action_type", ""))
                action["likelihood_source"] = "placeholder_discriminating"
                tuned += 1
    return tuned


def _abnormal_idx(aid: str) -> int:
    return SDS_ABNORMAL_INDEX_OVERRIDE.get(aid, 1)


def _build_tier1a(aid, outcomes, linked, all_rcs):
    """Deterministic parameter reads. Normal outcome nearly eliminates linked RCs."""
    idx_abn = _abnormal_idx(aid)
    idx_nor = 1 - idx_abn
    if len(outcomes) == 2:
        map_linked = {outcomes[idx_nor]: 0.01, outcomes[idx_abn]: 0.95}
        map_unlinked = {outcomes[idx_nor]: 0.98, outcomes[idx_abn]: 0.02}
    else:
        n = len(outcomes)
        map_linked = {o: 1.0/n for o in outcomes}
        map_unlinked = {o: 1.0/n for o in outcomes}
    return _expand(map_linked, map_unlinked, linked, all_rcs, "calibrated_tier1a")


def _build_tier1b(aid, outcomes, linked, all_rcs, n_linked):
    """Sensor/actuator tests. Softer than 1a."""
    if n_linked == 1:
        p_abn_l, p_nor_l = 0.95, 0.05
    elif n_linked <= 3:
        p_abn_l, p_nor_l = 0.85, 0.10
    else:
        p_abn_l, p_nor_l = 0.75, 0.15
    p_abn_u, p_nor_u = 0.03, 0.95

    idx_abn = _abnormal_idx(aid)
    if len(outcomes) == 2:
        idx_nor = 1 - idx_abn
        map_linked = {outcomes[idx_nor]: p_nor_l, outcomes[idx_abn]: p_abn_l}
        map_unlinked = {outcomes[idx_nor]: p_nor_u, outcomes[idx_abn]: p_abn_u}
    elif len(outcomes) == 3 and "fail" in outcomes:
        p_inc_l = max(1.0 - p_abn_l - p_nor_l, 0.01)
        p_inc_u = max(1.0 - p_abn_u - p_nor_u, 0.01)
        map_linked = {"pass": p_nor_l, "fail": p_abn_l, "inconclusive": p_inc_l}
        map_unlinked = {"pass": p_nor_u, "fail": p_abn_u, "inconclusive": p_inc_u}
    else:
        n = len(outcomes)
        map_linked = {o: 1.0/n for o in outcomes}
        map_unlinked = {o: 1.0/n for o in outcomes}
    return _expand(map_linked, map_unlinked, linked, all_rcs, "calibrated_tier1b")


def _build_tier2(linked, all_rcs, n_linked):
    """Narrow-scope manual checks."""
    if n_linked <= 1:
        ml = {"fail": 0.90, "pass": 0.05, "inconclusive": 0.05}
        mu = {"fail": 0.03, "pass": 0.92, "inconclusive": 0.05}
    elif n_linked == 2:
        ml = {"fail": 0.80, "pass": 0.10, "inconclusive": 0.10}
        mu = {"fail": 0.05, "pass": 0.87, "inconclusive": 0.08}
    else:
        ml = {"fail": 0.70, "pass": 0.15, "inconclusive": 0.15}
        mu = {"fail": 0.05, "pass": 0.87, "inconclusive": 0.08}
    return _expand(ml, mu, linked, all_rcs, "calibrated_tier2")


def _build_tier3(outcomes, linked, all_rcs, action_type):
    """Mildly discriminating placeholders."""
    n = len(outcomes)
    if n < 2:
        return []
    if action_type == "verification" and set(outcomes) == {"pass", "fail", "inconclusive"}:
        ml = {"fail": 0.55, "pass": 0.30, "inconclusive": 0.15}
        mu = {"fail": 0.10, "pass": 0.80, "inconclusive": 0.10}
    elif action_type == "resolution" and "resolved" in outcomes and "not_resolved" in outcomes:
        if "partially_resolved" in outcomes:
            ml = {"resolved": 0.60, "partially_resolved": 0.20, "not_resolved": 0.20}
            mu = {"resolved": 0.08, "partially_resolved": 0.07, "not_resolved": 0.85}
        else:
            ml = {"resolved": 0.65, "not_resolved": 0.35}
            mu = {"resolved": 0.05, "not_resolved": 0.95}
    else:
        ml, mu = {}, {}
        for i, o in enumerate(outcomes):
            if i == 0:
                ml[o], mu[o] = 0.5, 0.2
            else:
                ml[o] = 0.5 / (n-1)
                mu[o] = 0.8 / (n-1)
    return _expand(ml, mu, linked, all_rcs, "placeholder_discriminating")


def _expand(map_linked, map_unlinked, linked, all_rcs, source):
    """Expand probability maps into per-RC likelihood entries."""
    out = []
    for rc_id in all_rcs:
        probs = map_linked if rc_id in linked else map_unlinked
        for olabel, prob in probs.items():
            out.append({"rc_id": rc_id, "outcome_label": olabel, "probability": round(prob, 4), "estimate_source": source})
    return out

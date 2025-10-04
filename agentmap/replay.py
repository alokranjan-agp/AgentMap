
from __future__ import annotations
import json, math
from typing import Dict, Any, List, Tuple

from .ontology import AgentMap
from .planner_ao import AOPlanner, Stage
from .router import CostLens

def default_demo_context() -> Dict[str, Any]:
    return {"email":"patient@example.com","patient_id":"P-12345","specialty":"endocrinology","amount":1200,"country":"IN"}

def default_stages() -> List[Stage]:
    return [
        Stage(name="auth", candidates=["email.send_otp"]),
        Stage(name="schedule", candidates=["clinic.schedule_followup"]),
        Stage(name="charge", candidates=["pg_a.charge","pg_b.charge"]),
        Stage(name="notify", candidates=["email.notify"]),
    ]

def compare_plans(saved: Dict[str, Any], recomputed: Dict[str, Any]) -> Dict[str, Any]:
    saved_actions = [s["action_id"] for s in saved.get("steps", [])]
    new_actions = [s["action_id"] for s in recomputed.get("steps", [])]
    equal = saved_actions == new_actions
    diffs = []
    for i, (a, b) in enumerate(zip(saved_actions, new_actions)):
        if a != b:
            diffs.append({"index": i, "saved": a, "recomputed": b})
    if len(saved_actions) != len(new_actions):
        diffs.append({"index": "length", "saved_len": len(saved_actions), "recomputed_len": len(new_actions)})
    return {"equal": equal and len(diffs)==0, "diffs": diffs, "saved_actions": saved_actions, "recomputed_actions": new_actions}

def replay_verify(map_model: AgentMap, saved_plan_json: Dict[str, Any], ctx: Dict[str, Any] | None = None, stages: List[Stage] | None = None) -> Dict[str, Any]:
    """
    Recompute a plan for the same goal using the same deterministic planner and
    compare action sequences.
    """
    ctx = ctx or default_demo_context()
    stages = stages or default_stages()
    planner = AOPlanner(map_model, CostLens(weights=dict(latency=0.5, price=0.5)))
    new_plan = planner.plan(goal=saved_plan_json["goal"], context=ctx, stages=stages, model_version="replay")
    saved_slim = {"goal": saved_plan_json["goal"], "steps": [{"action_id": s["action_id"]} for s in saved_plan_json["steps"]]}
    new_slim = {"goal": new_plan.goal, "steps": [{"action_id": s.action_id} for s in new_plan.steps]}
    result = compare_plans(saved_slim, new_slim)
    result["recomputed_plan"] = json.loads(new_plan.to_json())
    return result

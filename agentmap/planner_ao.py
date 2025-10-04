
from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib

from .ontology import AgentMap, Plan, PlanStep, Action, canonical_json
from .policy import evaluate_policy
from .schema import validate_inputs
from .router import CostLens
from .llm_adapter import stable_tiebreak_key

@dataclass(frozen=True)
class Stage:
    name: str
    candidates: List[str]

class AOPlanner:
    def __init__(self, map_model: AgentMap, lens: CostLens):
        self.map = map_model
        self.lens = lens

    def _policies_pass(self, action: Action, ctx: Dict[str, Any]) -> Tuple[bool, List[str]]:
        passed = []
        for pid in action.policy_ids or []:
            pol = next((p for p in self.map.policies if p.id == pid), None)
            if not pol:
                continue
            if not evaluate_policy(pol.expr, ctx):
                return False, passed
            passed.append(pol.id)
        return True, passed

    def _score_action(self, action: Action, ctx: Dict[str, Any]) -> Tuple[float, str]:
        score = self.lens.score(action, ctx)
        tie = stable_tiebreak_key(ctx.get("__goal__", "goal"), action.id)
        return score, tie

    def plan(self, goal: str, context: Dict[str, Any], stages: List[Stage], model_version: Optional[str]=None) -> Plan:
        ctx = dict(context)
        ctx["__goal__"] = goal
        steps: List[PlanStep] = []
        for st in stages:
            viable: List[Tuple[float, str, Action, List[str]]] = []
            for aid in st.candidates:
                a = self.map.action_by_id(aid)
                if not a:
                    continue
                ok, passed = self._policies_pass(a, ctx)
                if not ok:
                    continue
                inputs = {k: ctx.get(k) for k in a.inputs.keys() if k in ctx}
                try:
                    if a.input_schema:
                        validate_inputs(a.input_schema, inputs)
                except Exception:
                    continue
                score, tie = self._score_action(a, ctx)
                viable.append((score, tie, a, passed))
            if not viable:
                raise RuntimeError(f"No feasible actions for stage '{st.name}'")
            viable.sort(key=lambda t: (t[0], t[1]))
            best = viable[0][2]
            passed = viable[0][3]
            inputs = {k: ctx.get(k) for k in best.inputs.keys() if k in ctx}
            step = PlanStep(
                action_id=best.id,
                inputs=inputs,
                expected_effects=[],
                cost=self.lens.score(best, ctx),
                latency_ms=best.latency_ms,
                passed_policies=passed
            )
            step.fingerprint = hashlib.sha256(canonical_json(step).encode("utf-8")).hexdigest()[:16]
            steps.append(step)
        return Plan(goal=goal, steps=steps, map_version=self.map.map_version(), model_version=model_version)

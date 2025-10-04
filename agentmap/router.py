
from __future__ import annotations
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass
import hashlib

from .ontology import AgentMap, Plan, PlanStep, Action, canonical_json
from .policy import evaluate_policy
from .llm_adapter import stable_tiebreak_key
from .schema import validate_inputs

@dataclass
class CostLens:
    weights: Dict[str, float]
    def score(self, action: Action, ctx: Dict[str, Any]) -> float:
        price = action.cost_base * float(ctx.get("amount", 1.0)) if "amount" in ctx else action.cost_base
        latency = action.latency_ms
        risk = action.risk
        return (
            self.weights.get("price", 0.0) * price +
            self.weights.get("latency", 0.0) * latency +
            self.weights.get("risk", 0.0) * risk
        )

class Router:
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

    def plan(self, goal: str, context: Dict[str, Any], allowed_actions: List[str], model_version: str|None=None) -> Plan:
        actions: List[Action] = []
        for aid in allowed_actions:
            a = self.map.action_by_id(aid)
            if a:
                actions.append(a)
        scored = []
        for a in actions:
            ok, passed = self._policies_pass(a, context)
            if not ok:
                continue
            inputs = {k: context.get(k) for k in a.inputs.keys() if k in context}
            if a.input_schema:
                validate_inputs(a.input_schema, inputs)
            s = self.lens.score(a, context)
            tie = stable_tiebreak_key(goal, a.id)
            scored.append((s, tie, a, passed))
        if not scored:
            raise RuntimeError("No feasible actions.")
        scored.sort(key=lambda t: (t[0], t[1]))
        best = scored[0][2]
        passed = scored[0][3]
        inputs = {k: context.get(k) for k in best.inputs.keys() if k in context}
        step = PlanStep(
            action_id=best.id, inputs=inputs, expected_effects=[], cost=self.lens.score(best, context),
            latency_ms=best.latency_ms, passed_policies=passed
        )
        step.fingerprint = hashlib.sha256(canonical_json(step).encode("utf-8")).hexdigest()[:16]
        return Plan(goal=goal, steps=[step], map_version=self.map.map_version(), model_version=model_version)

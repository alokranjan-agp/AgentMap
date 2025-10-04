
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import time, random

from .ontology import Plan, PlanStep, AgentMap, Action

class ExecutionResult:
    def __init__(self, step: PlanStep, ok: bool, info: Dict[str, Any]):
        self.step = step
        self.ok = ok
        self.info = info

class MockExecutor:
    def __init__(self, map_model: AgentMap, seed: int = 123):
        self.map = map_model
        self.rnd = random.Random(seed)

    def _simulate(self, action: Action) -> Tuple[bool, Dict[str, Any]]:
        jitter = self.rnd.uniform(-0.05, 0.05)
        simulated_ms = max(0.0, action.latency_ms * (1.0 + jitter))
        time.sleep(min(0.001, simulated_ms/1e6))  # very small sleep for demo
        p_success = max(0.85, 1.0 - action.risk)
        ok = (self.rnd.random() < p_success)
        return ok, {"simulated_latency_ms": simulated_ms, "p_success": p_success}

    def run(self, plan: Plan) -> Dict[str, Any]:
        results: List[ExecutionResult] = []
        for step in plan.steps:
            act = self.map.action_by_id(step.action_id)
            ok, info = self._simulate(act)
            results.append(ExecutionResult(step, ok, info))
            if not ok:
                break
        report = {
            "goal": plan.goal,
            "map_version": plan.map_version,
            "router_version": plan.router_version,
            "model_version": plan.model_version,
            "steps": [
                {
                    "action_id": r.step.action_id,
                    "inputs": r.step.inputs,
                    "passed_policies": r.step.passed_policies,
                    "planned_cost": r.step.cost,
                    "planned_latency_ms": r.step.latency_ms,
                    "fingerprint": r.step.fingerprint,
                    "ok": r.ok,
                    "exec_info": r.info
                } for r in results
            ],
            "success": all(r.ok for r in results)
        }
        return report

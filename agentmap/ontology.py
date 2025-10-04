
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import json, hashlib

def canonical_json(obj: Any) -> str:
    def default(o):
        if hasattr(o, "model_dump"):
            return o.model_dump()
        raise TypeError(f"Not serializable: {type(o)}")
    return json.dumps(obj, default=default, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

class Action(BaseModel):
    id: str
    inputs: Dict[str, str] = Field(default_factory=dict)
    input_schema: Dict[str, Any] | None = None
    cost_base: float = 0.0
    latency_ms: float = 0.0
    risk: float = 0.0
    policy_ids: List[str] = Field(default_factory=list)

class Tile(BaseModel):
    id: str
    actions: List[Action] = Field(default_factory=list)

class Policy(BaseModel):
    id: str
    expr: str

class PlanStep(BaseModel):
    action_id: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    expected_effects: List[str] = Field(default_factory=list)
    cost: float = 0.0
    latency_ms: float = 0.0
    passed_policies: List[str] = Field(default_factory=list)
    fingerprint: str = ""

class Plan(BaseModel):
    goal: str
    steps: List[PlanStep] = Field(default_factory=list)
    map_version: str = ""
    router_version: str = "router@0.3.0"
    model_version: Optional[str] = None

    def to_json(self) -> str:
        return canonical_json(self)

class AgentMap(BaseModel):
    tiles: List[Tile] = Field(default_factory=list)
    policies: List[Policy] = Field(default_factory=list)

    def map_version(self) -> str:
        h = hashlib.sha256()
        h.update(canonical_json(self).encode("utf-8"))
        return h.hexdigest()[:16]

    def action_by_id(self, aid: str) -> Optional[Action]:
        for t in self.tiles:
            for a in t.actions:
                if a.id == aid:
                    return a
        return None

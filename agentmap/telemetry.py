
from __future__ import annotations
from typing import Optional
import random

from .ontology import AgentMap

class TelemetryConfig:
    def __init__(self, outage_prob: float=0.0, latency_jitter: float=0.2, price_jitter: float=0.1, ewma_alpha: float=0.3):
        self.outage_prob = outage_prob
        self.latency_jitter = latency_jitter
        self.price_jitter = price_jitter
        self.ewma_alpha = ewma_alpha

class TelemetrySimulator:
    def __init__(self, map_model: AgentMap, cfg: TelemetryConfig):
        self.map = map_model
        self.cfg = cfg
        self._baseline = {}
        for t in self.map.tiles:
            for a in t.actions:
                self._baseline[a.id] = dict(latency=a.latency_ms, cost=a.cost_base)

    def step(self, seed: Optional[int]=None) -> None:
        rnd = random.Random(seed)
        for t in self.map.tiles:
            for a in t.actions:
                base = self._baseline[a.id]
                if rnd.random() < self.cfg.outage_prob:
                    obs_latency = base["latency"] * (5.0 + rnd.random()*2.0)
                    obs_cost = base["cost"] * (1.0 + rnd.random()*0.5)
                else:
                    obs_latency = base["latency"] * (1.0 + (rnd.random()*2-1)*self.cfg.latency_jitter)
                    obs_cost = base["cost"] * (1.0 + (rnd.random()*2-1)*self.cfg.price_jitter)

                a.latency_ms = self.cfg.ewma_alpha * obs_latency + (1 - self.cfg.ewma_alpha) * a.latency_ms
                a.cost_base  = self.cfg.ewma_alpha * obs_cost   + (1 - self.cfg.ewma_alpha) * a.cost_base

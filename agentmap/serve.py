
from __future__ import annotations
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
import threading

from .ontology import AgentMap
from .telemetry import TelemetrySimulator, TelemetryConfig

# Jinja environment using in-package templates
env = Environment(
    loader=FileSystemLoader(searchpath="agentmap/templates"),
    autoescape=select_autoescape(['html'])
)

class DashboardState:
    def __init__(self, amap: AgentMap, outage: float = 0.05):
        self.map = amap
        self.sim = TelemetrySimulator(amap, TelemetryConfig(outage_prob=outage, ewma_alpha=0.3))
        self.lock = threading.Lock()

    def step(self):
        with self.lock:
            self.sim.step()

def create_app(state: DashboardState) -> FastAPI:
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        template = env.get_template("index.html")
        # collect tiles/actions
        tiles = []
        for t in state.map.tiles:
            actions = []
            for a in t.actions:
                actions.append({
                    "id": a.id,
                    "latency_ms": round(a.latency_ms, 2),
                    "cost_base": a.cost_base,
                    "risk": a.risk,
                    "policies": a.policy_ids
                })
            tiles.append({"id": t.id, "actions": actions})
        html = template.render(tiles=tiles)
        return HTMLResponse(html)

    @app.post("/api/telemetry/step")
    def telemetry_step():
        state.step()
        return JSONResponse({"ok": True})

    @app.get("/api/map")
    def api_map():
        data = {"tiles": []}
        for t in state.map.tiles:
            data["tiles"].append({
                "id": t.id,
                "actions": [{"id": a.id, "latency_ms": a.latency_ms, "cost_base": a.cost_base, "risk": a.risk, "policies": a.policy_ids} for a in t.actions]
            })
        return JSONResponse(data)

    return app

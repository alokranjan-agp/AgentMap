import argparse, os, json
import matplotlib.pyplot as plt

from .ontology import AgentMap, Tile, Action, Policy
from .planner_ao import AOPlanner, Stage
from .router import CostLens
from .telemetry import TelemetrySimulator, TelemetryConfig
from .executor import MockExecutor

def build_demo_map(ablate_policies=False, ablate_tiles=False):
    # Tiles
    email = Tile(
        id="email",
        actions=[
            Action(id="email.send_otp", inputs={"email":"string"},
                   input_schema={"type":"object","properties":{"email":{"type":"string"}},"required":["email"]},
                   cost_base=0.0, latency_ms=120, risk=0.02, policy_ids=[]),
            Action(id="email.notify", inputs={"email":"string"},
                   input_schema={"type":"object","properties":{"email":{"type":"string"}},"required":["email"]},
                   cost_base=0.0, latency_ms=90, risk=0.01, policy_ids=[]),
        ]
    )
    clinic = Tile(
        id="clinic",
        actions=[
            Action(id="clinic.schedule_followup", inputs={"patient_id":"string","specialty":"string"},
                   input_schema={"type":"object","properties":{"patient_id":{"type":"string"},"specialty":{"type":"string"}},"required":["patient_id","specialty"]},
                   cost_base=0.0005, latency_ms=300, risk=0.05, policy_ids=["phi_guard"]),
        ]
    )
    payments = Tile(
        id="payments",
        actions=[
            Action(id="pg_a.charge", inputs={"amount":"number","country":"string"},
                   input_schema={"type":"object","properties":{"amount":{"type":"number","minimum":0},"country":{"type":"string"}},"required":["amount","country"]},
                   cost_base=0.00025, latency_ms=420, risk=0.08, policy_ids=["cap","india_only"]),
            Action(id="pg_b.charge", inputs={"amount":"number","country":"string"},
                   input_schema={"type":"object","properties":{"amount":{"type":"number","minimum":0},"country":{"type":"string"}},"required":["amount","country"]},
                   cost_base=0.00015, latency_ms=560, risk=0.10, policy_ids=["cap","india_only"]),
        ]
    )
    policies = []
    if not ablate_policies:
        policies = [
            Policy(id="cap", expr="amount <= 20000"),
            Policy(id="india_only", expr="country == 'IN'"),
            Policy(id="phi_guard", expr="country == 'IN'"),
        ]
    tiles = [email, clinic, payments]
    if ablate_tiles:
        # Flatten into one tile (no composition)
        all_actions = []
        for t in tiles:
            all_actions.extend(t.actions)
        tiles = [Tile(id="flat", actions=all_actions)]
    return AgentMap(tiles=tiles, policies=policies)

def default_stages():
    return [
        Stage(name="auth", candidates=["email.send_otp"]),
        Stage(name="schedule", candidates=["clinic.schedule_followup"]),
        Stage(name="charge", candidates=["pg_a.charge","pg_b.charge"]),
        Stage(name="notify", candidates=["email.notify"]),
    ]

def cmd_plan(args):
    os.makedirs(args.out, exist_ok=True)
    amap = build_demo_map(ablate_policies=("policies" in args.ablate), ablate_tiles=("tiles" in args.ablate))
    planner = AOPlanner(amap, CostLens(weights=dict(latency=0.5, price=0.5)))
    stages = default_stages()
    ctx = {"email":"patient@example.com","patient_id":"P-12345","specialty":"endocrinology","amount":1200,"country":"IN"}
    plan = planner.plan(goal="renew_and_book_followup", context=ctx, stages=stages, model_version="cli-ao*")
    with open(os.path.join(args.out,"plan.json"),"w") as f: f.write(plan.to_json())
    print("Wrote", os.path.join(args.out,"plan.json"))

def cmd_simulate(args):
    os.makedirs(args.out, exist_ok=True)
    amap = build_demo_map(ablate_policies=("policies" in args.ablate), ablate_tiles=("tiles" in args.ablate))
    with open(args.plan) as f:
        plan_in = json.load(f)
    planner = AOPlanner(amap, CostLens(weights=dict(latency=0.5, price=0.5)))
    stages = default_stages()
    ctx = {"email":"patient@example.com","patient_id":"P-12345","specialty":"endocrinology","amount":1200,"country":"IN"}

    sim = TelemetrySimulator(amap, TelemetryConfig(outage_prob=args.outage, ewma_alpha=0.3))
    sim.step(seed=1)
    plan2 = planner.plan(goal=plan_in["goal"], context=ctx, stages=stages, model_version="cli-ao*")
    with open(os.path.join(args.out,"plan_after_telemetry.json"),"w") as f: f.write(plan2.to_json())
    print("Wrote", os.path.join(args.out,"plan_after_telemetry.json"))

def cmd_run(args):
    os.makedirs(args.out, exist_ok=True)
    amap = build_demo_map(ablate_policies=("policies" in args.ablate), ablate_tiles=("tiles" in args.ablate))
    with open(args.plan) as f:
        plan_in = json.load(f)
    # Build Plan object ad-hoc isn't necessary; we replan using same goal for determinism
    from .ontology import Plan
    from .planner_ao import Stage
    planner = AOPlanner(amap, CostLens(weights=dict(latency=0.5, price=0.5)))
    stages = default_stages()
    ctx = {"email":"patient@example.com","patient_id":"P-12345","specialty":"endocrinology","amount":1200,"country":"IN"}
    plan = planner.plan(goal=plan_in["goal"], context=ctx, stages=stages, model_version="cli-ao*")
    ex = MockExecutor(amap, seed=args.seed)
    report = ex.run(plan)
    with open(os.path.join(args.out,"execution_report.json"),"w") as f: json.dump(report, f, indent=2)
    # Chart
    planned = [s["planned_latency_ms"] for s in report["steps"]]
    observed = [s["exec_info"]["simulated_latency_ms"] for s in report["steps"]]
    labels = [s["action_id"] for s in report["steps"]]
    x = range(len(labels)); width=0.35
    plt.figure()
    plt.bar([i-width/2 for i in x], planned, width, label="Planned")
    plt.bar([i+width/2 for i in x], observed, width, label="Observed")
    plt.title("Per-step Latency (ms)")
    plt.xlabel("Action"); plt.ylabel("ms")
    plt.xticks(list(x), labels, rotation=25, ha="right")
    plt.legend(); plt.tight_layout()
    fig = os.path.join(args.out,"latency_comparison.png")
    plt.savefig(fig); plt.close()
    print("Wrote", fig)

def cmd_bench(args):
    import time, math, random
    import pandas as pd
    os.makedirs(args.out, exist_ok=True)
    workloads = []
    wl_map = {"steady":0.0, "burst":0.05, "storm":0.2}
    for w in args.workloads.split(","):
        if w.strip() in wl_map:
            workloads.append((w.strip(), wl_map[w.strip()]))

    rows = []
    for name, outage in workloads:
        amap = build_demo_map(ablate_policies=("policies" in args.ablate), ablate_tiles=("tiles" in args.ablate))
        planner = AOPlanner(amap, CostLens(weights=dict(latency=0.5, price=0.5)))
        sim = TelemetrySimulator(amap, TelemetryConfig(outage_prob=outage, ewma_alpha=0.3)) if ("telemetry" not in args.ablate) else None
        for i in range(args.trials):
            if sim: sim.step(seed=i)
            start = time.perf_counter()
            stages = default_stages()
            ctx = {"email":"patient@example.com","patient_id":"P-12345","specialty":"endocrinology","amount":1200,"country":"IN"}
            try:
                plan = planner.plan(goal="bench_goal", context=ctx, stages=stages, model_version="cli-ao*")
                elapsed_ms = (time.perf_counter() - start)*1000
                total_cost = sum(s.cost for s in plan.steps)
                total_latency = sum(s.latency_ms for s in plan.steps)
                rows.append(dict(workload=name, success=1, ttfa_ms=elapsed_ms, total_latency_ms=total_latency, total_cost=total_cost))
            except Exception:
                elapsed_ms = (time.perf_counter() - start)*1000
                rows.append(dict(workload=name, success=0, ttfa_ms=elapsed_ms, total_latency_ms=math.nan, total_cost=math.nan))
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.DataFrame(rows)
    csvp = os.path.join(args.out, "results.csv")
    df.to_csv(csvp, index=False)

    # Plots
    plt.figure()
    df.boxplot(column="ttfa_ms", by="workload")
    plt.title("Time-to-first-plan (ms)"); plt.suptitle(""); plt.xlabel("Workload"); plt.ylabel("ms")
    plt.tight_layout(); plt.savefig(os.path.join(args.out,"ttfa_boxplot.png")); plt.close()

    plt.figure()
    sr = df.groupby("workload")["success"].mean().reset_index()
    plt.bar(sr["workload"], sr["success"])
    plt.title("Success rate"); plt.xlabel("Workload"); plt.ylabel("rate")
    plt.tight_layout(); plt.savefig(os.path.join(args.out,"success_rate.png")); plt.close()

    plt.figure()
    agg = df[df["success"]==1].groupby("workload")[["total_cost","total_latency_ms"]].mean().reset_index()
    plt.bar(agg["workload"], agg["total_cost"])
    plt.title("Mean total cost (successful plans)"); plt.xlabel("Workload"); plt.ylabel("cost units")
    plt.tight_layout(); plt.savefig(os.path.join(args.out,"mean_cost.png")); plt.close()

    plt.figure()
    plt.bar(agg["workload"], agg["total_latency_ms"])
    plt.title("Mean total latency (ms) — successful plans"); plt.xlabel("Workload"); plt.ylabel("ms")
    plt.tight_layout(); plt.savefig(os.path.join(args.out,"mean_latency.png")); plt.close()

    print("CSV:", csvp)

from .cli_extra import _extend_parser_for_replay_and_serve, cmd_replay, cmd_serve
from .cli_llm import _extend_parser_for_llm_demo

def main():
    p = argparse.ArgumentParser(prog="agentmap", description="Deterministic agent maps CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    _extend_parser_for_replay_and_serve(sub)
    _extend_parser_for_llm_demo(sub)

    sp = sub.add_parser("plan", help="Create AO* plan for demo map")
    sp.add_argument("--out", required=True)
    sp.add_argument("--ablate", default="", help="Comma-separated: policies,tiles")
    sp.set_defaults(func=cmd_plan)

    ss = sub.add_parser("simulate", help="Apply telemetry and re-plan")
    ss.add_argument("--plan", required=True)
    ss.add_argument("--out", required=True)
    ss.add_argument("--outage", type=float, default=0.05)
    ss.add_argument("--ablate", default="", help="Comma-separated: policies,tiles")
    ss.set_defaults(func=cmd_simulate)

    sr = sub.add_parser("run", help="Execute a plan and emit report + chart")
    sr.add_argument("--plan", required=True)
    sr.add_argument("--out", required=True)
    sr.add_argument("--seed", type=int, default=999)
    sr.add_argument("--ablate", default="", help="Comma-separated: policies,tiles")
    sr.set_defaults(func=cmd_run)

    sb = sub.add_parser("bench", help="Run synthetic benchmarks and plots")
    sb.add_argument("--out", required=True)
    sb.add_argument("--tiles", type=int, default=3)           # reserved for future
    sb.add_argument("--actions", type=int, default=100)       # reserved for future
    sb.add_argument("--trials", type=int, default=200)
    sb.add_argument("--workloads", default="steady,burst,storm")
    sb.add_argument("--ablate", default="", help="Comma-separated: policies,telemetry,tiles")
    sb.set_defaults(func=cmd_bench)

    args = p.parse_args()
    # normalize ablate list (only for commands that have it)
    if hasattr(args, 'ablate'):
        args.ablate = [a.strip() for a in args.ablate.split(",") if a.strip()]
    args.func(args)

if __name__ == "__main__":
    main()

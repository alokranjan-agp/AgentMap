from .replay import replay_verify, default_demo_context, default_stages
from .serve import create_app, DashboardState
from .cli import build_demo_map
import uvicorn
import os
import json

def cmd_replay(args):
    import json
    amap = build_demo_map(ablate_policies=("policies" in args.ablate), ablate_tiles=("tiles" in args.ablate))
    with open(args.plan) as f:
        saved = json.load(f)
    ctx = default_demo_context() if not args.ctx else json.load(open(args.ctx))
    res = replay_verify(amap, saved, ctx=ctx, stages=default_stages())
    outp = args.out or os.path.dirname(args.plan) or "."
    os.makedirs(outp, exist_ok=True)
    with open(os.path.join(outp, "replay_result.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("Replay equal:", res["equal"])
    if res["diffs"]:
        print("Diffs:", res["diffs"])
    print("Wrote", os.path.join(outp, "replay_result.json"))

def cmd_serve(args):
    amap = build_demo_map(ablate_policies=("policies" in args.ablate), ablate_tiles=("tiles" in args.ablate))
    state = DashboardState(amap, outage=args.outage)
    app = create_app(state)
    uvicorn.run(app, host=args.host, port=args.port)

# Register subcommands
def _extend_parser_for_replay_and_serve(sub):
    rp = sub.add_parser("replay", help="Verify determinism by recomputing plan and comparing")
    rp.add_argument("--plan", required=True)
    rp.add_argument("--out", default="")
    rp.add_argument("--ctx", default="", help="Optional JSON file with context")
    rp.add_argument("--ablate", default="", help="Comma-separated: policies,tiles")
    rp.set_defaults(func=cmd_replay)

    sv = sub.add_parser("serve", help="Web dashboard for tiles & live traffic")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--outage", type=float, default=0.05)
    sv.add_argument("--ablate", default="", help="Comma-separated: policies,tiles")
    sv.set_defaults(func=cmd_serve)

# Inject into existing main() by replacing the subparser creation block

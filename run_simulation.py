"""
run_simulation.py -- thin driver: parameters -> mesh -> solver -> output.

Usage:
    python run_simulation.py
    python run_simulation.py --zeta_eff 0 --d0 0       # sanity check 1
    python run_simulation.py --d0 0                   # sanity check 2
"""
import argparse

import mesh_gen
from solver import Params, run


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--E", type=float, default=1.0, help="Young's modulus")
    p.add_argument("--nu", type=float, default=0.3, help="Poisson ratio")
    p.add_argument("--xi", type=float, default=1.0, help="drag/friction coefficient")
    p.add_argument("--A", type=float, default=1.0, help="nematic bulk coefficient")
    p.add_argument("--K", type=float, default=0.05, help="Frank elastic constant")
    p.add_argument("--Gamma", type=float, default=1.0, help="nematic mobility")
    p.add_argument(
        "--gamma",
        type=float,
        default=0.0,
        help="strain–orientation coupling coefficient",
    )

    activity = p.add_mutually_exclusive_group()
    activity.add_argument(
        "--zeta_eff",
        type=float,
        default=None,
        help="effective activity (< 0 contractile; > 0 extensile)",
    )
    activity.add_argument(
        "--zeta",
        type=float,
        default=None,
        help="deprecated alias for --zeta_eff",
    )

    loading = p.add_mutually_exclusive_group()
    loading.add_argument(
        "--epsilon0",
        type=float,
        default=None,
        help="nominal strain amplitude epsilon0 = d0/(2L)",
    )
    loading.add_argument(
        "--d0",
        type=float,
        default=None,
        help="wall displacement amplitude; alternative to --epsilon0",
    )
    p.add_argument("--T", type=float, default=5.0, help="wall actuation pulse duration")
    p.add_argument("--dt", type=float, default=0.02, help="time step")
    p.add_argument("--t_end", type=float, default=15.0, help="total simulated time")
    p.add_argument("--save_every", type=int, default=5, help="save every N steps")
    p.add_argument("--outdir", type=str, default="output", help="output directory")
    p.add_argument(
        "--lambda_coupling",
        type=float,
        default=0.0,
        help="corotation coefficient; must be zero in the current model",
    )

    args = p.parse_args()

    if args.lambda_coupling != 0.0:
        p.error("--lambda_coupling must be 0 in the current model.")

    return args


def main():
    args = parse_args()
    defaults = Params()

    # Resolve effective activity. Do not subtract gamma again.
    if args.zeta is not None:
        print("Note: --zeta is deprecated and is interpreted as --zeta_eff.")
        zeta_eff = args.zeta
    elif args.zeta_eff is not None:
        zeta_eff = args.zeta_eff
    else:
        zeta_eff = defaults.zeta_eff

    # Resolve loading amplitude.
    if args.epsilon0 is not None:
        d0 = 2.0 * mesh_gen.L * args.epsilon0
    elif args.d0 is not None:
        d0 = args.d0
    else:
        d0 = defaults.d0

    params = Params(
        E=args.E, nu=args.nu, xi=args.xi,
        A=args.A, K=args.K, Gamma=args.Gamma,
        gamma=args.gamma, zeta_eff=zeta_eff,
        d0=d0, T=args.T,
        dt=args.dt, t_end=args.t_end, save_every=args.save_every,
        outdir=args.outdir,
        lambda_coupling=args.lambda_coupling,
    )

    tau_el, tau_Q = params.derived_timescales()
    print(f"L = {mesh_gen.L}, H = {mesh_gen.H}")
    print(f"tau_el (elastic relaxation) ~ {tau_el:.4g}")
    print(f"tau_Q  (nematic relaxation) ~ {tau_Q:.4g}")
    print(f"T (actuation pulse)        = {params.T:.4g}")
    print(f"gamma                     = {params.gamma:.4g}")
    print(f"zeta_eff                  = {params.zeta_eff:.4g}")
    print(f"lambda_coupling           = {params.lambda_coupling:.4g}")
    print(f"d0                        = {params.d0:.4g}")
    print(f"epsilon0 = d0/(2L)         = {params.d0 / (2.0 * mesh_gen.L):.4g}")

    mesh, cell_tags, facet_tags = mesh_gen.build_mesh()
    run(params, mesh=mesh, cell_tags=cell_tags, facet_tags=facet_tags)


if __name__ == "__main__":
    main()

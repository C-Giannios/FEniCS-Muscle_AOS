"""
run_simulation.py -- thin driver: parameters -> mesh -> solver -> output.

Usage:
    python run_simulation.py
    python run_simulation.py --zeta 0 --d0 0          # sanity check 1
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
    p.add_argument("--zeta", type=float, default=-0.5, help="activity (< 0 contractile)")
    p.add_argument("--d0", type=float, default=0.1, help="wall actuation amplitude")
    p.add_argument("--T", type=float, default=5.0, help="wall actuation pulse duration")
    p.add_argument("--dt", type=float, default=0.02, help="time step")
    p.add_argument("--t_end", type=float, default=15.0, help="total simulated time")
    p.add_argument("--save_every", type=int, default=5, help="save every N steps")
    p.add_argument("--outdir", type=str, default="output", help="output directory")
    return p.parse_args()


def main():
    args = parse_args()
    params = Params(
        E=args.E, nu=args.nu, xi=args.xi,
        A=args.A, K=args.K, Gamma=args.Gamma, zeta=args.zeta,
        d0=args.d0, T=args.T,
        dt=args.dt, t_end=args.t_end, save_every=args.save_every,
        outdir=args.outdir,
    )

    tau_el, tau_Q = params.derived_timescales()
    print(f"L = {mesh_gen.L}, H = {mesh_gen.H}")
    print(f"tau_el (elastic relaxation) ~ {tau_el:.4g}")
    print(f"tau_Q  (nematic relaxation) ~ {tau_Q:.4g}")
    print(f"T (actuation pulse)        = {params.T:.4g}")

    mesh, cell_tags, facet_tags = mesh_gen.build_mesh()
    run(params, mesh=mesh, cell_tags=cell_tags, facet_tags=facet_tags)


if __name__ == "__main__":
    main()

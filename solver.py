"""
solver.py -- staggered, semi-implicit (IMEX) time-stepping solver for a
contractile active nematic field embedded in an isotropic linear-elastic
solid (overdamped mechanics, plane stress).

Each time step:
  1. Mechanics: backward-Euler solve for u^{n+1}, using Q^n (lagged) for the
     active stress and the current wall displacement d(t^{n+1}) as a
     Dirichlet BC on the actuated wall. Purely linear elasticity solve.
  2. Corotation: v^{n+1} = (u^{n+1} - u^n)/dt, W^{n+1} = skew(grad v^{n+1}).
  3. Nematic: backward-Euler on the Frank-elastic (Laplacian) term, explicit
     on the bulk (Landau-de Gennes) term and the corotation term (both
     evaluated with Q^n). Also linear.

All physical parameters are passed in via a Params dataclass (see
run_simulation.py) so this module has no hard-coded constants.
"""
from dataclasses import dataclass, field
import numpy as np
import ufl
import basix.ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io
from dolfinx.fem.petsc import LinearProblem

import mesh_gen


@dataclass
class Params:
    # Elasticity
    E: float = 1.0
    nu: float = 0.3
    xi: float = 1.0

    # Nematic
    A: float = 1.0
    K: float = 0.05
    Gamma: float = 1.0
    zeta: float = -0.5          # ζ < 0 => contractile
    lambda_coupling: float = 1.0  # coefficient of WQ-QW; -1 gives legacy rotation

    # Actuation
    d0: float = 0.1
    T: float = 5.0
    # Time stepping
    dt: float = 0.02
    t_end: float = 15.0
    save_every: int = 5

    # Output
    outdir: str = "output"

    def derived_timescales(self):
        L = mesh_gen.L
        tau_el = self.xi * L**2 / self.E
        tau_Q = 1.0 / (self.Gamma * self.A)
        return tau_el, tau_Q


def wall_displacement(t: float, p: Params) -> float:
    """d(t) = d0 sin(pi t / T) for t in [0, T], 0 afterward."""
    if 0.0 <= t <= p.T:
        return p.d0 * np.sin(np.pi * t / p.T)
    return 0.0


def run(p: Params, mesh=None, cell_tags=None, facet_tags=None, verbose=True):
    comm = MPI.COMM_WORLD

    if mesh is None:
        mesh, cell_tags, facet_tags = mesh_gen.build_mesh(comm)

    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(fdim, tdim)

    if comm.rank == 0 and verbose:
        tau_el, tau_Q = p.derived_timescales()
        print(f"tau_el ~ {tau_el:.4g}, tau_Q ~ {tau_Q:.4g}, "
              f"T = {p.T:.4g}, dt = {p.dt:.4g}, t_end = {p.t_end:.4g}")

    # ------------------------------------------------------------------
    # Function spaces
    # ------------------------------------------------------------------
    u_elem = basix.ufl.element("Lagrange", mesh.basix_cell(), 2, shape=(2,))
    V = fem.functionspace(mesh, u_elem)

    q_elem = basix.ufl.element("Lagrange", mesh.basix_cell(), 1, shape=(2,))
    QS = fem.functionspace(mesh, q_elem)

    # CG1 scalar/vector spaces purely for output/interpolation
    s1_elem = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
    S1 = fem.functionspace(mesh, s1_elem)
    v1_elem = basix.ufl.element("Lagrange", mesh.basix_cell(), 1, shape=(2,))
    V1 = fem.functionspace(mesh, v1_elem)

    # ------------------------------------------------------------------
    # Elastic parameters (plane-stress effective Lame constants)
    # ------------------------------------------------------------------
    lam_star = p.E * p.nu / (1.0 - p.nu**2)
    mu = p.E / (2.0 * (1.0 + p.nu))

    def eps(u):
        return ufl.sym(ufl.grad(u))

    def sigma_passive(u):
        return lam_star * ufl.tr(eps(u)) * ufl.Identity(2) + 2.0 * mu * eps(u)

    def Q_tensor(Q):
        """Build the full traceless-symmetric 2x2 tensor from (Qxx, Qxy)."""
        return ufl.as_matrix([[Q[0], Q[1]], [Q[1], -Q[0]]])

    zeta_c = fem.Constant(mesh, PETSc.ScalarType(p.zeta))

    def sigma_active(Q):
        return -zeta_c * Q_tensor(Q)

    # ------------------------------------------------------------------
    # Boundary conditions -- mechanics
    # ------------------------------------------------------------------
    actuated_facets = facet_tags.find(mesh_gen.TAG_ACTUATED)
    fixed_facets = facet_tags.find(mesh_gen.TAG_FIXED)

    dofs_act_u = fem.locate_dofs_topological(V, fdim, actuated_facets)
    dofs_fix_u = fem.locate_dofs_topological(V, fdim, fixed_facets)

    d_const = fem.Constant(mesh, PETSc.ScalarType((0.0, 0.0)))
    zero_vec = fem.Constant(mesh, PETSc.ScalarType((0.0, 0.0)))

    bc_act_u = fem.dirichletbc(d_const, dofs_act_u, V)
    bc_fix_u = fem.dirichletbc(zero_vec, dofs_fix_u, V)
    bcs_u = [bc_act_u, bc_fix_u]

    # ------------------------------------------------------------------
    # Boundary conditions -- nematic (both walls anchored at 45 deg)
    # ------------------------------------------------------------------
    dofs_act_Q = fem.locate_dofs_topological(QS, fdim, actuated_facets)
    dofs_fix_Q = fem.locate_dofs_topological(QS, fdim, fixed_facets)

    Q_anchor = fem.Constant(mesh, PETSc.ScalarType((0.0, 1.0)))
    bc_act_Q = fem.dirichletbc(Q_anchor, dofs_act_Q, QS)
    bc_fix_Q = fem.dirichletbc(Q_anchor, dofs_fix_Q, QS)
    bcs_Q = [bc_act_Q, bc_fix_Q]

    # ------------------------------------------------------------------
    # State functions
    # ------------------------------------------------------------------
    u_n = fem.Function(V, name="u")
    u_n.x.array[:] = 0.0

    Q_n = fem.Function(QS, name="Q")
    Q_n.sub(0).interpolate(lambda x: np.zeros(x.shape[1]))
    Q_n.sub(1).interpolate(lambda x: np.ones(x.shape[1]))
    Q_n.x.scatter_forward()

    dt_c = fem.Constant(mesh, PETSc.ScalarType(p.dt))
    xi_c = fem.Constant(mesh, PETSc.ScalarType(p.xi))
    A_c = fem.Constant(mesh, PETSc.ScalarType(p.A))
    K_c = fem.Constant(mesh, PETSc.ScalarType(p.K))
    Gamma_c = fem.Constant(mesh, PETSc.ScalarType(p.Gamma))

    # ------------------------------------------------------------------
    # Mechanics weak form
    # ------------------------------------------------------------------
    u_trial = ufl.TrialFunction(V)
    w_test = ufl.TestFunction(V)

    a_u = (xi_c / dt_c) * ufl.inner(u_trial, w_test) * ufl.dx \
        + ufl.inner(sigma_passive(u_trial), eps(w_test)) * ufl.dx
    L_u = (xi_c / dt_c) * ufl.inner(u_n, w_test) * ufl.dx \
        + ufl.inner(sigma_active(Q_n), eps(w_test)) * ufl.dx

    # ------------------------------------------------------------------
    # Nematic weak form
    # ------------------------------------------------------------------
    Q_trial = ufl.TrialFunction(QS)
    q_test = ufl.TestFunction(QS)

    u_np1_holder = fem.Function(V)  # updated with the mechanics solution each step

    v_expr = (u_np1_holder - u_n) / dt_c
    grad_v = ufl.grad(v_expr)
    w_spin = 0.5 * (grad_v[0, 1] - grad_v[1, 0])  # scalar spin: W = [[0, w],[-w, 0]]

    S2_n = Q_n[0] * Q_n[0] + Q_n[1] * Q_n[1]
    Hb_xx = A_c * (1.0 - S2_n) * Q_n[0]
    Hb_xy = A_c * (1.0 - S2_n) * Q_n[1]


    # lambda*(WQ-QW) gives d(Qxx,Qxy)/dt = 2*lambda*w*(Qxy,-Qxx).
    # Hence the component phase 2*theta rotates at rate -2*lambda*w.
    # Apply the exact frozen-spin rotation, preserving Qxx^2 + Qxy^2
    # pointwise in this substep. lambda=-1 reproduces the legacy rotation.


    rot_angle = -2.0 * p.lambda_coupling * w_spin * dt_c
    cos_r = ufl.cos(rot_angle)
    sin_r = ufl.sin(rot_angle)
    Q_rot_xx = cos_r * Q_n[0] - sin_r * Q_n[1]
    Q_rot_xy = sin_r * Q_n[0] + cos_r * Q_n[1]

    rhs_xx = Q_rot_xx + dt_c * Gamma_c * Hb_xx
    rhs_xy = Q_rot_xy + dt_c * Gamma_c * Hb_xy

    a_Q = ufl.inner(Q_trial, q_test) * ufl.dx \
        + dt_c * Gamma_c * K_c * ufl.inner(ufl.grad(Q_trial), ufl.grad(q_test)) * ufl.dx
    L_Q = (rhs_xx * q_test[0] + rhs_xy * q_test[1]) * ufl.dx

    # ------------------------------------------------------------------
    # Output fields (CG1 interpolants of derived quantities)
    # ------------------------------------------------------------------
    u1 = fem.Function(V1, name="displacement")
    vm1 = fem.Function(S1, name="von_Mises_stress")
    Qxx1 = fem.Function(S1, name="Qxx")
    Qxy1 = fem.Function(S1, name="Qxy")
    theta1 = fem.Function(S1, name="theta")
    S_mag1 = fem.Function(S1, name="S")

    sigma_total = sigma_passive(u_n) + sigma_active(Q_n)
    s_dev = sigma_total - (ufl.tr(sigma_total) / 2.0) * ufl.Identity(2)
    von_mises_expr = ufl.sqrt(
        sigma_total[0, 0]**2 - sigma_total[0, 0] * sigma_total[1, 1]
        + sigma_total[1, 1]**2 + 3.0 * sigma_total[0, 1]**2
    )
    vm_expr_compiled = fem.Expression(von_mises_expr, S1.element.interpolation_points())

    import os
    os.makedirs(p.outdir, exist_ok=True)
    xdmf_path = os.path.join(p.outdir, "fields.xdmf")
    xdmf = io.XDMFFile(comm, xdmf_path, "w")
    xdmf.write_mesh(mesh)

    snap_dir = os.path.join(p.outdir, "snapshots")
    os.makedirs(snap_dir, exist_ok=True)
    x1 = V1.tabulate_dof_coordinates()[:, :2]
    x1_topology_saved = False

    def save_state(t: float, step: int):
        nonlocal x1_topology_saved
        u1.interpolate(u_n)
        Qxx1.interpolate(fem.Expression(Q_n[0], S1.element.interpolation_points()))
        Qxy1.interpolate(fem.Expression(Q_n[1], S1.element.interpolation_points()))
        theta1.x.array[:] = 0.5 * np.arctan2(Qxy1.x.array, Qxx1.x.array)
        S_mag1.x.array[:] = np.sqrt(Qxx1.x.array**2 + Qxy1.x.array**2)
        vm1.interpolate(vm_expr_compiled)

        xdmf.write_function(u1, t)
        xdmf.write_function(vm1, t)
        xdmf.write_function(Qxx1, t)
        xdmf.write_function(Qxy1, t)
        xdmf.write_function(theta1, t)
        xdmf.write_function(S_mag1, t)

        conn = mesh.geometry.dofmap.reshape(-1, 3) if not x1_topology_saved else None
        npz_kwargs = dict(
            t=t,
            coords=x1,
            u=u1.x.array.reshape(-1, 2).copy(),
            von_mises=vm1.x.array.copy(),
            Qxx=Qxx1.x.array.copy(),
            Qxy=Qxy1.x.array.copy(),
            theta=theta1.x.array.copy(),
            S=S_mag1.x.array.copy(),
        )
        if conn is not None:
            npz_kwargs["connectivity"] = conn
            x1_topology_saved = True
        np.savez(os.path.join(snap_dir, f"step_{step:05d}.npz"), **npz_kwargs)

    save_state(0.0, 0)

    # ------------------------------------------------------------------
    # Time-stepping loop
    # ------------------------------------------------------------------
    nsteps = int(round(p.t_end / p.dt))
    t = 0.0
    for step in range(1, nsteps + 1):
        t = step * p.dt
        d_const.value[1] = wall_displacement(t, p)

        problem_u = LinearProblem(
            a_u, L_u, bcs=bcs_u,
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        u_np1 = problem_u.solve()
        u_np1_holder.x.array[:] = u_np1.x.array

        problem_Q = LinearProblem(
            a_Q, L_Q, bcs=bcs_Q,
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        Q_np1 = problem_Q.solve()

        u_n.x.array[:] = u_np1.x.array
        Q_n.x.array[:] = Q_np1.x.array

        if step % p.save_every == 0 or step == nsteps:
            save_state(t, step)
            if comm.rank == 0 and verbose:
                print(f"step {step}/{nsteps}  t={t:.3f}  "
                      f"|u|_max={np.max(np.abs(u_n.x.array)):.4e}  "
                      f"d(t)={wall_displacement(t, p):.4e}")

    xdmf.close()
    if comm.rank == 0 and verbose:
        print(f"Done. Fields written to {xdmf_path}, snapshots in {snap_dir}")

    return mesh, u_n, Q_n

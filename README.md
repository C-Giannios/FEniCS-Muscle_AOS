# Active nematic in an elastic solid

A 2D FEniCSx (dolfinx) finite-element simulation of a contractile active
nematic field embedded in an isotropic linear-elastic solid, on a
triangle-plus-rectangle domain, actuated by a time-dependent wall
displacement.

## Geometry

A 45-45-90 right triangle glued to a long rectangle along one leg, with the
right-angle vertex at the origin:

- `A = (0, 0)` — 90° vertex, on the rectangle's near edge
- `B = (L, 0)` — 45° vertex, on the rectangle's near edge
- `C = (0, -L)` — 45° vertex, the free tip (sharp point of the whole shape)
- Rectangle: `x in [0, L]`, `y in [0, H]`, `H >> L` (default `H = 5L`)

`A` lies exactly on the straight segment between `(0, H)` and `C` (both on
`x = 0`), so the union of the triangle and rectangle is a single
simply-connected quadrilateral `C -> B -> (L,H) -> (0,H) -> C` — one Gmsh
plane surface, no boolean fragment/union needed.

Boundary tags:

| tag        | edge(s)                                  | role                        |
|------------|-------------------------------------------|-----------------------------|
| `actuated` | `(0,H) -> C`, length `H+L`                 | moving wall (Dirichlet)     |
| `fixed`    | `B -> (L,H)`                               | fixed wall (Dirichlet u=0)  |
| `free`     | hypotenuse `C -> B` and far edge `(L,H)->(0,H)` | traction-free (Neumann) |

The mesh is refined near the tip `C` with a Gmsh `Distance` + `Threshold`
field (see `mesh_gen.py` for the exact target sizes).

## Governing equations

Unknowns: displacement `u(x,t)` (vector), nematic tensor via its two
independent components `Qxx(x,t)`, `Qxy(x,t)` (`Qyy = -Qxx`, `Qyx = Qxy` —
traceless and symmetric by construction, `Q = S(cos2θ, sin2θ)`).

**Mechanics** — overdamped, linear elasticity, plane stress:

$$
\xi \, \partial_t u = \nabla \cdot \sigma, \qquad
\sigma = \sigma_{\text{passive}} + \sigma_{\text{active}}
$$

$$
\sigma_{\text{passive}} = \lambda^* \, \mathrm{tr}(\varepsilon(u)) \, I + 2\mu \, \varepsilon(u),
\qquad \varepsilon(u) = \mathrm{sym}(\nabla u)
$$

$$
\lambda^* = \frac{E\nu}{1-\nu^2}, \qquad \mu = \frac{E}{2(1+\nu)}
\qquad \text{(plane-stress effective Lam\'e constants)}
$$

$$
\sigma_{\text{active}} = -\zeta Q, \qquad \zeta < 0 \text{ for contractile activity}
$$

**Nematic** — relaxational (Landau-de Gennes) dynamics + corotation with the
local material spin:

$$
\partial_t Q = \Gamma H + (W \cdot Q - Q \cdot W), \qquad
H = A\left[Q - \tfrac{1}{2}(Q\!:\!Q)\,Q\right] + K \nabla^2 Q
$$

$$
W = \mathrm{skew}(\nabla v), \qquad v = \partial_t u
$$

Writing `S² = Qxx² + Qxy²` and `w` the scalar spin with
`W = [[0, w], [-w, 0]]`, `w = ½(∂vx/∂y − ∂vy/∂x)`:

$$
H_{xx} = A(1-S^2)Q_{xx} + K\nabla^2 Q_{xx}, \qquad
H_{xy} = A(1-S^2)Q_{xy} + K\nabla^2 Q_{xy}
$$

$$
(W\!\cdot\!Q - Q\!\cdot\!W)_{xx} = 2wQ_{xy}, \qquad
(W\!\cdot\!Q - Q\!\cdot\!W)_{xy} = -2wQ_{xx}
$$

i.e. corotation rotates the director's phase `2θ` at rate `2w` — the two
scalar components decouple into two Helmholtz-type problems (no tensor
solve needed). The equilibrium `S=1` matches the 45° anchoring
`(Qxx,Qxy)=(0,1)` exactly.

### Boundary / initial conditions

- **Mechanics**: `u = (0, d(t))` on `actuated` (uniform translation, so the
  edge stays straight); `u = 0` on `fixed`; traction-free on `free`.
  `d(t) = d0 · sin(πt/T)` for `t ∈ [0,T]`, `0` afterward.
- **Nematic**: `(Qxx,Qxy) = (0,1)` (45° anchoring) on both `actuated` and
  `fixed`; homogeneous Neumann on `free`.
- **Initial**: `u(x,0)=0`, `Q(x,0)=(0,1)` uniform.

## Numerical scheme

Vector CG2 for `u`; a 2-component (CG1) blocked space for `(Qxx,Qxy)`
(reconstructed as a full tensor only for stress evaluation/output via
`ufl.as_matrix`). Staggered, semi-implicit (IMEX) per step:

1. **Mechanics** (backward Euler, using lagged `Q^n` for active stress):

$$
a(u,w) = \int \left[\tfrac{\xi}{\Delta t}u\cdot w + \lambda^*\mathrm{tr}(\varepsilon(u))\mathrm{tr}(\varepsilon(w)) + 2\mu\,\varepsilon(u)\!:\!\varepsilon(w)\right]dx
$$

$$
L(w) = \int \left[\tfrac{\xi}{\Delta t}u^n\cdot w + \zeta Q^n\!:\!\varepsilon(w)\right] dx
$$

2. `v^{n+1} = (u^{n+1}-u^n)/\Delta t`, `w^{n+1}` from `grad(v)` (UFL
   expression, no separate `Function`).
3. **Nematic** (implicit Laplacian, explicit bulk + corotation):

$$
a(Q,q) = \int\left[Q\cdot q + \Delta t\,\Gamma K\,\nabla Q\!:\!\nabla q\right]dx
$$

   The corotation part of the right-hand side is **not** the naive
   forward-Euler increment `Q^n + 2\Delta t\, w[Q_{xy}^n,-Q_{xx}^n]`: that
   linearization of a rotation is not magnitude-preserving, and inflates
   `|Q|` every step once `w\Delta t = O(1)` — an explicit-Euler-on-a-rotation
   instability that shows up as a hard blow-up at higher activity or on
   finer meshes (where `w` is locally larger, e.g. near the tip). Instead
   the exact rotation by angle `2w\Delta t` is applied:

$$
Q^{\text{rot}}_{xx} = \cos(2w\Delta t)\,Q_{xx}^n - \sin(2w\Delta t)\,Q_{xy}^n, \qquad
Q^{\text{rot}}_{xy} = \sin(2w\Delta t)\,Q_{xx}^n + \cos(2w\Delta t)\,Q_{xy}^n
$$

$$
L(q) = \int\left[\left(Q^{\text{rot}}_{xx} + \Delta t\,\Gamma H_{xx}\right)q_{xx} + \left(Q^{\text{rot}}_{xy} + \Delta t\,\Gamma H_{xy}\right)q_{xy}\right]dx
$$

   which exactly preserves `S²` regardless of `w\Delta t`, unconditionally.

## Parameters

All exposed as named constants/dataclass fields (`mesh_gen.py`,
`solver.py::Params`): `L, H` (geometry); `E, nu` (elasticity); `xi`
(drag); `A, K, Gamma, zeta` (nematic); `d0, T` (wall actuation); `dt,
t_end`. Derived timescales `tau_el ~ xi*L^2/E` and `tau_Q ~ 1/(Gamma*A)`
are printed at the start of each run.

**Activity guidance**: with the default `A=1, K=0.05, Gamma=1`, `zeta`
around `-0.5` to `-0.6` gives a clean, single-region contractile effect
(order parameter `S` stays above ~0.85 everywhere, deformation and stress
concentrate smoothly at the tip). Beyond `zeta ~ -0.7` the stress and
director fields start developing multiple small disordered patches
(defect nucleation), and by `zeta ~ -0.8` this becomes a genuine
multi-defect, active-turbulence-like regime (the spec's own stability
criterion — Frank constant `K` large enough that `sqrt(K/A)` is comparable
to the domain size — is not satisfied by the default `K`). That regime is
intentionally not the target operating point here; keep `|zeta| ≲ 0.6-0.7`
for clean runs, or raise `K` substantially if higher activity is needed.

## Repo layout

- `mesh_gen.py` — builds and saves the tagged mesh via the Gmsh Python API.
- `solver.py` — assembles and time-steps the coupled system, writing `u`,
  von Mises stress, `Q` (both components plus derived director angle
  `θ = ½atan2(Qxy,Qxx)` and magnitude `S = sqrt(Qxx²+Qxy²)`) to XDMF and
  `.npz` snapshots.
- `run_simulation.py` — thin CLI driver: parameters → mesh → solver → output.
- `postprocess.ipynb` — loads the `.npz` snapshots and produces snapshot
  plots (displacement, stress, director field) and two animations
  (director field, stress), both on the deforming mesh. The director field
  is coarse-grained onto a grid (`NBINS_X`/`NBINS_Y`) by averaging the
  Cartesian `Q`-tensor components per bin before plotting — the correct way
  to downsample a headless nematic field (naive angle-averaging breaks near
  the `±π/2` branch cut).

## Running

Requires dolfinx (targets the modern API: `basix.ufl.element`,
`fem.functionspace`, `fem.petsc.LinearProblem` — developed against dolfinx
0.9.0). On a cluster without a dolfinx module, pull the official image with
Apptainer:

```bash
apptainer pull dolfinx_v0.9.0.sif docker://dolfinx/dolfinx:v0.9.0
apptainer exec dolfinx_v0.9.0.sif python3 run_simulation.py \
    --zeta -0.6 --d0 0.1 --t_end 15 --outdir output_zeta0.6
```

Then open `postprocess.ipynb` (point `OUTDIR` at the run's output
directory) with a kernel that has `numpy`/`matplotlib` — these don't need
to be the dolfinx environment.
